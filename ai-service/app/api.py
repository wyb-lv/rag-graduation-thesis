"""API HTTP cho FE goi RAG hybrid (dense + bm25 + rerank), dung lai logic trong query_model.py.

Chay (trong thu muc Model):
    uvicorn app.api:app --reload --port 8000

FE goi:
    POST /ask  { "question": "...", "kho": "...", "reference_date": "2026-01-31", "loai_van_ban": ["thong_bao"] }
      ->  { "answer": "...", "in_scope": true, "sources": [...] }
    Chi `question` la bat buoc. Khong tim thay quy dinh phu hop -> in_scope=false, sources=[], khong goi LLM.
    Them tuy chon: "session_id" (UUID), "reset"; response luon co "session_id" (xem API.md).
    POST /ask/stream  (cung body)  ->  text/event-stream (SSE), xem API.md:
      data: {"event": "sources", "in_scope": true, "sources": [...]}
      data: {"event": "delta", "text": "..."}      (lap lai)
      data: [DONE]                                   (hoac data: {"event": "error", "detail": "..."})
BGE-M3 va reranker nap ngay khi server khoi dong (thread nen) va nam trong bo nho cho cac
request sau. Request toi khi dang nap se cho nap xong roi chay tiep. GET /health -> "models": pending/ready/error.
"""
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool

from . import config as cfg
from . import conversations, qdrant_setup, session, speech
from .ingest import embed_consumer, embed_service
from .models import encoders
from .ocr import consumer as ocr_consumer
from .rag import query_model as qm
from .router import guard, router_llm
from .router.router_llm import Action
from .router.small_talk import Intent

log = logging.getLogger("uvicorn.error")
_warmup = {"status": "pending"}
_warmed = threading.Event()  # set = khong co warm-up dang chay; lifespan clear() ngay truoc khi bat thread, warm_up() set lai khi xong
_warmed.set()

WARMUP_TEXT = "Thí sinh được đăng ký tối đa bao nhiêu nguyện vọng?"


def warm_up():
    """Nap model va chay thu mot lan (CUDA khoi tao kernel o lan chay dau): ~16s tren RTX 3050."""
    start = time.perf_counter()
    try:
        encoders.encode_texts([WARMUP_TEXT])
        encoders.rerank_scores(WARMUP_TEXT, [WARMUP_TEXT])
    except Exception:
        _warmup["status"] = "error"
        log.exception("Warmup model that bai, model se duoc nap lai o request dau tien")
    else:
        _warmup["status"] = "ready"
        log.info("Warmup model xong sau %.1fs", time.perf_counter() - start)
    finally:
        _warmed.set()  # thanh cong hay that bai deu tha cac request dang doi


@asynccontextmanager
async def lifespan(app):
    # Nhanh trong HYBRID_LEGS phai co vector trong collection; thieu thi RuntimeError, server khong khoi dong
    qdrant_setup.validate_legs(qm.get_qdrant())
    # Thread nen: server nhan request ngay (/health tra loi duoc), khong bat FE cho tren connection refused
    _warmed.clear()
    threading.Thread(target=warm_up, name="model-warmup", daemon=True).start()
    if not (cfg.GEMINI_API_KEY and cfg.ROUTER_MODEL):
        log.warning("thieu GEMINI_API_KEY / GEMINI_MODEL: router se loi o moi request va cau hoi di thang vao RAG")
    try:
        session.get_store()  # nap LlamaIndex + tao engine SQLite luc khoi dong, khong doi den request dau tien
    except Exception as e:  # service van khoi dong; request dau tien se thu lai
        log.warning("khong khoi tao duoc session store (%s): se thu lai o request dau tien", type(e).__name__)
    try:
        await conversations.ensure_indexes()
    except Exception as e:  # service van khoi dong; ghi luot hoi dap se loi (va chi ghi log) cho den khi MongoDB len
        log.warning("khong tao duoc index MongoDB (%s): kiem tra MONGODB_URI", type(e).__name__)
    # Khoi dong RabbitMQ consumer de nhan job embed tu file-service; broker chua len thi consumer tu thu ket noi lai
    embed_consumer.start_consumer()
    ocr_consumer.start_consumer()  # OCR (PDF / DOCX / ảnh) thay cho OCR cũ của file-service
    yield


app = FastAPI(title="RAG QA API", lifespan=lifespan)
# Không thêm CORSMiddleware: api-gateway đã gắn CORS, thêm ở đây thì response có 2 header
# Access-Control-Allow-Origin và trình duyệt chặn fetch.


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=None, ge=1, le=10)  # so doan lam ngu canh; mac dinh top_k_rerank
    kho: Optional[str] = None
    reference_date: Optional[date] = None  # loc van ban con hieu luc tai ngay nay; mac dinh hom nay
    khoa: Optional[int] = Field(default=None, ge=1990, le=2100)  # khoa cua sinh vien; chi lay van ban ap dung cho khoa nay
    loai_van_ban: Optional[List[str]] = None  # chi tim trong cac loai van ban nay (vd ["thong_bao", "quy_che"])
    session_id: Optional[str] = None  # UUID cua cuoc tro chuyen; bo trong thi server tao moi
    reset: bool = False  # xoa lich su session truoc khi xu ly request nay


class Source(BaseModel):
    source: str
    text: str
    score: float  # diem rerank neu co rerank, khong thi diem truy van (cosine / RRF)
    rerank_score: Optional[float] = None
    payload: Dict[str, Any] = {}


class AskResponse(BaseModel):
    answer: str
    in_scope: bool
    sources: List[Source]
    # rag | greeting | thanks | goodbye | capability | meta | blocked | off_topic | clarify (do router_llm.py quyet dinh)
    intent: str = Intent.RAG.value
    session_id: str  # UUID cua cuoc tro chuyen (do client gui hoac server tao)


@app.get("/health")
def health():
    return {"status": "ok", "models": _warmup["status"]}


def _question(payload):
    question = guard.normalize_question(payload.question.strip())  # chi con zero width / ky tu dieu khien cung la cau hoi rong
    if not question:
        raise HTTPException(status_code=400, detail="question khong duoc de trong")
    return question


@contextmanager
def _pipeline_errors():
    try:
        yield
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Thieu bien moi truong: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Loi khi truy van Qdrant/goi LLM: {e}")


def _sources(hits):
    return [
        Source(
            source=h["source"] or "",
            text=h["text"],
            score=h["rerank_score"] if h["rerank_score"] is not None else h["retrieval_score"],
            rerank_score=h["rerank_score"],
            payload=h["payload"],
        )
        for h in hits
    ]


@dataclass
class _Turn:
    """Moi thu cua mot request: cau hoi chuan hoa, session dang mo va khoa / kho hieu luc."""
    question: str
    session: Any = None  # session.Session; giu lock cho den _end()
    sid: Optional[str] = None
    user_id: Optional[str] = None  # X-User-Id do gateway gan; None = khong xac thuc
    khoa: Optional[int] = None
    kho: Optional[str] = None


def _valid_facts(raw):
    """khoa / kho da luu, kiem lai bang chinh bo validator cua AskRequest; gia tri khong hop le bi bo."""
    facts = {}
    for name, value in raw.items():
        try:
            facts[name] = getattr(AskRequest(question="-", **{name: value}), name)
        except ValidationError:
            pass
    return facts


async def _begin(payload, request):
    """Chuan hoa cau hoi, mo session va gop khoa / kho: gia tri trong request de len gia tri da luu.
    reference_date khong bao gio duoc luu."""
    t = _Turn(_question(payload))
    if not _warmed.is_set():  # dang nap model: doi xong roi moi chay router / Groq, tranh timeout do tranh CPU voi luc nap; chua giu lock session
        if not await run_in_threadpool(_warmed.wait, cfg.WARMUP_WAIT_S):
            log.warning("warm-up chua xong sau %.0fs: chay tiep, router / viet lai co the timeout", cfg.WARMUP_WAIT_S)
    t.sid = str(uuid.uuid4()) if payload.session_id is None else session.parse_id(payload.session_id)
    if t.sid is None:
        raise HTTPException(status_code=400, detail="session_id phai la UUID")
    t.user_id = request.headers.get(cfg.SESSION_USER_HEADER)
    key = session.key_for(t.sid, t.user_id)
    try:
        t.session = await (await session.store()).open(key, reset=payload.reset)
    except Exception as e:  # loi session store khong duoc lam hong cau hoi: tra loi khong co lich su, chi ghi loai loi
        log.warning("session %s: khong mo duoc session (%s), tra loi khong co lich su", session.digest(key), type(e).__name__)
    stored = _valid_facts(t.session.facts) if t.session else {}
    t.khoa = payload.khoa if payload.khoa is not None else stored.get("khoa")
    t.kho = payload.kho or stored.get("kho")
    return t


def _end(t):
    if t.session:
        t.session.release()


def _log_router(t, routed, action):
    d = routed.decision
    log.info(
        "router intent=%s conf=%s action=%s model=%s cached=%s fallback=%s failure=%s latency_ms=%d session=%s history_turns=%d note=%s",
        d.intent if d else "-", f"{d.confidence:.2f}" if d else "-", action.intent.value, routed.model, routed.cached,
        "rag" if d is None else "-", routed.failure or "-", routed.latency_ms,
        session.digest(t.sid), len(t.session.history) // 2 if t.session else 0, action.note or "-",
    )


async def _plan(t):
    """Quyet dinh cho request: Action co `answer` = tra loi co dinh (khong truy van / goi LLM), khong thi di tiep RAG voi `action.query`.

    Mot lan goi router (cau hoi + lich su + khoa / kho + co previous_was_clarify) tu suy luan y dinh; khong co
    luat tu khoa nao. Router loi (timeout, quota, JSON hong, het ngan sach RPM, thieu khoa / model) thi cau hoi di thang vao RAG,
    van duoc chuan hoa, quet doan trich, prompt co rao va guard dau ra bao ve; khong bao gio tra off_topic / clarify khi router loi."""
    history = t.session.history if t.session else []
    facts, previous_was_clarify = {"khoa": t.khoa, "kho": t.kho}, bool(t.session and t.session.previous_was_clarify)
    routed = await router_llm.resolve(t.question, history, facts, previous_was_clarify)
    action = (router_llm.apply_policy(routed.decision, t.question, facts, previous_was_clarify)
              if routed.decision is not None else Action(Intent.RAG, query=t.question))
    _log_router(t, routed, action)
    return action


def _pipeline_kwargs(t, payload):
    return dict(top_k=payload.top_k, kho=t.kho, reference_date=payload.reference_date, khoa=t.khoa,
                loai_van_ban=payload.loai_van_ban)


def _history(t):
    return t.session.history if t.session else []


async def _remember(t, answer, intent, store=True):
    """Luu luot da hoan tat. Loi luu khong lam hong cau tra loi da co.
    `store`=False (bi chan / lac de): khong vao lich su prompt cua session nhung van vao kho luu tru, vi nguoi dung da thay luot nay."""
    if t.session and store:
        try:
            await t.session.add_turn(t.question, answer, intent.value, {"khoa": t.khoa, "kho": t.kho})
        except Exception as e:
            log.warning("session %s: khong luu duoc luot nay (%s)", session.digest(t.sid), type(e).__name__)
    try:
        await conversations.add_turn(t.sid, t.user_id, t.question, answer, intent.value)
    except Exception as e:
        log.warning("session %s: khong luu duoc vao kho hoi thoai (%s)", session.digest(t.sid), type(e).__name__)


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request):
    t = await _begin(payload, request)
    try:
        action = await _plan(t)
        if action.answer is not None:
            answer, in_scope, hits, intent, store = action.answer, action.in_scope, [], action.intent, action.store
        else:
            kw = _pipeline_kwargs(t, payload)
            if action.query != t.question:  # chi khi router viet lai: khong thi loi goi y het nhu truoc
                kw["search_query"] = action.query
            if _history(t):
                kw["history"] = _history(t)
            with _pipeline_errors():
                answer, hits, in_scope = await run_in_threadpool(qm.ask, t.question, **kw)
            intent = Intent.BLOCKED if answer == cfg.GUARD_REFUSAL_ANSWER else Intent.RAG  # guard dau ra da chan cau tra loi
            store = intent is Intent.RAG
        await _remember(t, answer, intent, store)
        return AskResponse(answer=answer, in_scope=in_scope, sources=_sources(hits), intent=intent.value, session_id=t.sid)
    finally:
        _end(t)


def _sse(data):
    """Mot su kien SSE. json.dumps escape xuong dong nen moi su kien chi co mot dong `data:`."""
    return f"data: {data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request):
    """Nhu /ask nhung tra cau tra loi dang SSE theo tung doan token.

    Truy van + rerank chay truoc khi gui header, nen loi o buoc nay van tra ma HTTP (400/500/502) nhu /ask.
    Loi khi dang sinh cau tra loi thi gui su kien `error` (status 200 da gui roi) va dong stream, khong co [DONE].
    Luot hoi dap chi duoc luu vao session sau delta cuoi, truoc `[DONE]`; stream loi / bi chan thi khong luu gi.
    Lock session giu den khi stream ket thuc (hoac client ngat), de request ke tiep cung session doc lich su da day du.
    """
    t = await _begin(payload, request)
    try:
        action = await _plan(t)
        if action.answer is not None:  # chao hoi / bi chan / off_topic / clarify: cung hop dong SSE, sources rong + dung mot delta
            hits, chunks, in_scope, intent, store = [], iter([action.answer]), action.in_scope, action.intent, action.store
        else:
            with _pipeline_errors():
                hits = await run_in_threadpool(qm.find_hits, action.query or t.question, **_pipeline_kwargs(t, payload))
                chunks = (
                    qm.generate_answer_stream(t.question, hits, **({"history": _history(t)} if _history(t) else {}))
                    if hits else iter([qm.NO_RESULT_ANSWER])
                )
            in_scope, intent, store = bool(hits), Intent.RAG, True
    except BaseException:
        _end(t)
        raise
    sources = [s.model_dump() for s in _sources(hits)]
    # `intent` chi them vao su kien khi khac rag, de su kien cua cac cau hoi RAG giu nguyen tung byte
    extra = {"session_id": t.sid, **({} if intent is Intent.RAG else {"intent": intent.value})}

    async def events():
        try:
            yield _sse({"event": "sources", "in_scope": in_scope, "sources": sources, **extra})
            parts = []
            try:
                async for text in iterate_in_threadpool(chunks):
                    parts.append(text)
                    yield _sse({"event": "delta", "text": text})
            except guard.OutputBlocked:
                yield _sse({"event": "error", "detail": cfg.GUARD_REFUSAL_ANSWER})
                return
            except Exception as e:
                yield _sse({"event": "error", "detail": f"Loi khi goi LLM: {e}"})
                return
            # truoc [DONE]: client dong ket noi ngay khi nhan [DONE] thi doan sau yield cuoi khong chay
            await _remember(t, "".join(parts), intent, store)
            yield _sse("[DONE]")
        finally:
            _end(t)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(_end, t),  # events() chua kip chay (client ngat som) thi van nha lock
    )


# ── Conversation endpoints (MongoDB) ─────────────────────────────────────────

class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    intent: Optional[str] = None  # chi co o tin nhan cua tro ly
    ts: datetime


class ConversationSummary(BaseModel):
    session_id: str
    title: str  # cau hoi dau tien, bi cat
    created_at: datetime
    updated_at: datetime


class Conversation(ConversationSummary):
    messages: List[ConversationMessage]


def _conversation_owner(request, *, required=False):
    """X-User-Id do gateway gan sau khi xac thuc; `required`: thieu thi 401 (liet ke can biet la cua ai)."""
    user_id = request.headers.get(cfg.SESSION_USER_HEADER)
    if required and not user_id:
        raise HTTPException(status_code=401, detail=f"Thieu header {cfg.SESSION_USER_HEADER} (dang nhap qua api-gateway)")
    return user_id


def _conversation_id(session_id):
    sid = session.parse_id(session_id)
    if sid is None:
        raise HTTPException(status_code=400, detail="session_id phai la UUID")
    return sid


@contextmanager
def _archive_errors():
    try:
        yield
    except HTTPException:
        raise
    except Exception as e:  # chi ghi loai loi: thong bao loi cua driver co the chua MONGODB_URI (mat khau)
        log.warning("conversations: %s", type(e).__name__)
        raise HTTPException(status_code=502, detail="Khong truy cap duoc kho luu tru hoi thoai")


@app.get("/conversations", response_model=List[ConversationSummary], summary="Danh sách cuộc hội thoại của người dùng (mới nhất trước)")
async def list_conversations(request: Request, limit: int = Query(50, ge=1, le=cfg.CONVERSATIONS_LIST_MAX)):
    user_id = _conversation_owner(request, required=True)
    with _archive_errors():
        return await conversations.list_for(user_id, limit)


@app.get("/conversations/{session_id}", response_model=Conversation, summary="Toàn bộ tin nhắn của một cuộc hội thoại")
async def get_conversation(session_id: str, request: Request):
    user_id, sid = _conversation_owner(request), _conversation_id(session_id)
    with _archive_errors():
        found = await conversations.get(sid, user_id)
    if found is None:
        raise HTTPException(status_code=404, detail="Khong co cuoc hoi thoai nay")
    return found


@app.delete("/conversations/{session_id}", summary="Xóa một cuộc hội thoại (cả lịch sử ngắn hạn của session)")
async def delete_conversation(session_id: str, request: Request):
    user_id, sid = _conversation_owner(request), _conversation_id(session_id)
    with _archive_errors():
        (await (await session.store()).open(session.key_for(sid, user_id), reset=True)).release()  # khong de cuoc hoi thoai da xoa van nam trong prompt cua luot sau
        if not await conversations.delete(sid, user_id):
            raise HTTPException(status_code=404, detail="Khong co cuoc hoi thoai nay")
    return {"session_id": sid, "message": "Da xoa cuoc hoi thoai."}


# ── Embed endpoints ──────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    """Yêu cầu embed một tài liệu vào Qdrant."""
    document_id: int = Field(..., description="ID tài liệu trong file-service")
    text: str = Field(..., min_length=1, description="Toàn bộ văn bản đã OCR")
    source: str = Field(..., min_length=1, description="Tên tài liệu hiển thị (VD: Quy chế 2024)")
    kho: Optional[str] = Field(default=None, description="Nhãn kho (VD: tuyen_sinh)")
    hieu_luc_tu: Optional[str] = Field(default=None, description="Ngày hiệu lực từ ISO-8601")
    hieu_luc_den: Optional[str] = Field(default=None, description="Ngày hiệu lực đến ISO-8601")


class EmbedResponse(BaseModel):
    document_id: int
    paragraphs: int
    upserted: int
    message: str


@app.post("/embed", response_model=EmbedResponse, summary="Embed tài liệu vào Qdrant")
def embed_document(payload: EmbedRequest):
    """Embed văn bản của một tài liệu vào Qdrant để phục vụ RAG.

    - Tách thành đoạn → BGE-M3 (dense) + BM25 → upsert Qdrant.
    - Idempotent: xóa vector cũ của `document_id` trước khi thêm mới.
    - Thường được gọi tự động từ RabbitMQ consumer sau khi file-service OCR xong.
    - Có thể gọi thủ công để re-index hoặc kiểm thử.
    """
    try:
        result = embed_service.embed_document(
            document_id=payload.document_id,
            text=payload.text,
            source=payload.source,
            kho=payload.kho,
            hieu_luc_tu=payload.hieu_luc_tu,
            hieu_luc_den=payload.hieu_luc_den,
        )
        return EmbedResponse(
            document_id=payload.document_id,
            paragraphs=result["paragraphs"],
            upserted=result["upserted"],
            message=f"Đã index {result['upserted']}/{result['paragraphs']} đoạn vào Qdrant.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi embed: {e}")


@app.delete("/embed/{document_id}", summary="Xóa vector của tài liệu khỏi Qdrant")
def delete_document_vectors(document_id: int):
    """Xóa toàn bộ vector liên quan đến `document_id` khỏi Qdrant.

    Nên gọi khi tài liệu bị xóa ở file-service.
    """
    try:
        embed_service.delete_document_vectors(document_id)
        return {"document_id": document_id, "message": "Đã xóa vector khỏi Qdrant."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xóa vector: {e}")

# ==============================================================================
# 4. VOICE INTERACTION ENDPOINTS (STT, VOICE-TO-RAG STREAM & TTS)
# ==============================================================================


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000, description="Văn bản câu trả lời cần đọc thành tiếng")
    voice: str = cfg.TTS_VOICE


def _read_and_validate_audio(file: UploadFile) -> tuple[bytes, str]:
    """Đọc giới hạn MAX_AUDIO_BYTES + 1 để chống tràn RAM (OOM) và kiểm tra định dạng âm thanh."""
    filename = file.filename or "audio.wav"
    content = file.file.read(cfg.MAX_AUDIO_BYTES + 1)
    try:
        speech.validate_audio_bytes(content, filename=filename)
    except speech.AudioTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except speech.UnsupportedAudioFormatError as e:
        raise HTTPException(status_code=415, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return content, filename


def _transcribe(file: UploadFile) -> dict:
    """Kiểm tra file âm thanh rồi chuyển thành văn bản; lỗi nào cũng thành HTTPException."""
    content, filename = _read_and_validate_audio(file)
    try:
        return speech.transcribe_speech(content, filename=filename)
    except Exception as e:
        log.exception("Lỗi xử lý STT")
        raise HTTPException(status_code=500, detail=f"Lỗi nhận diện giọng nói: {str(e)}")


@app.post("/stt", summary="Chuyển giọng nói thành văn bản (Dành cho nút Mic ghi âm)")
def speech_to_text(
    file: UploadFile = File(...),
):
    """
    Nhận file ghi âm từ người dùng, kiểm tra bảo mật (<= 5MB & Magic Bytes),
    thực hiện VAD lọc khoảng lặng + Groq Whisper + HSU Phonetic Post-Correction.
    """
    return _transcribe(file)


@app.post(
    "/ask/voice/stream",
    summary="Upload file âm thanh câu hỏi và nhận trực tiếp luồng câu trả lời RAG (SSE)",
)
def ask_voice_stream(
    file: UploadFile = File(...),
    top_k: int = Form(cfg.TOP_K_RERANK),
    kho: Optional[str] = Form(None),
    reference_date: Optional[date] = Form(None),
):
    """
    Luồng 1 (Upload File Âm thanh -> Hỏi trực tiếp):
      1. Kiểm tra bảo mật file âm thanh (<= 5MB, đúng định dạng audio).
      2. Chuyển âm thanh thành văn bản qua `speech.transcribe_speech`.
      3. Đưa thẳng văn bản câu hỏi vào pipeline RAG (`find_hits` + `generate_answer_stream`).
      4. Trả về luồng SSE gồm: `transcription` -> `sources` -> `delta` (nhiều lần) -> `[DONE]`.
    """
    stt_result = _transcribe(file)
    question = (stt_result.get("text") or "").strip()
    if not stt_result.get("is_speech") or not question:
        raise HTTPException(
            status_code=400,
            detail="Không nhận diện được giọng nói trong file âm thanh. Vui lòng chọn file có tiếng nói rõ ràng.",
        )

    with _pipeline_errors():
        hits = qm.find_hits(
            question,
            top_k=top_k,
            kho=kho,
            reference_date=reference_date,
        )
        chunks = qm.generate_answer_stream(question, hits) if hits else iter([qm.NO_RESULT_ANSWER])
    sources = [s.model_dump() for s in _sources(hits)]

    def events():
        yield _sse({"event": "transcription", "text": question})
        yield _sse({"event": "sources", "in_scope": bool(hits), "sources": sources})
        try:
            for text in chunks:
                yield _sse({"event": "delta", "text": text})
        except guard.OutputBlocked:
            yield _sse({"event": "error", "detail": cfg.GUARD_REFUSAL_ANSWER})
            return
        except Exception as e:
            yield _sse({"event": "error", "detail": f"Loi khi goi LLM: {e}"})
            return
        yield _sse("[DONE]")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/tts", summary="Đọc nội dung câu trả lời của AI thành giọng nói Tiếng Việt (MP3 Stream)")
async def text_to_speech(payload: TTSRequest):
    """
    Luồng 3: Chuẩn hóa các ký hiệu viết tắt học vụ HSU (QĐ-ĐHHS, HSU, SV, >= 2.5/4.0...)
    và stream trực tiếp dữ liệu MP3 về cho trình duyệt phát.
    """
    raw_text = (payload.text or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Văn bản cần đọc không được để trống.")

    try:
        processed_text = speech.normalize_academic_text(raw_text)
        if not processed_text:
            raise HTTPException(status_code=400, detail="Văn bản sau chuẩn hóa rỗng.")

        mp3_bytes = await speech.synthesize_speech_bytes(
            text=processed_text,
            voice=payload.voice or cfg.TTS_VOICE,
        )
        return Response(
            content=mp3_bytes,
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-store"},
        )
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Lỗi xử lý TTS")
        raise HTTPException(status_code=500, detail=f"Lỗi tổng hợp giọng nói: {str(e)}")