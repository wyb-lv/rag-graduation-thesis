"""API HTTP cho FE goi RAG hybrid (dense + sparse + rerank), dung lai logic trong query_model.py.

Chay (trong thu muc Model):
    uvicorn api:app --reload --port 8000

FE goi:
    POST /ask  { "question": "...", "mode": "hybrid_rerank", "kho": "...", "reference_date": "2026-01-31",
                 "loai_van_ban": ["thong_bao"] }
      ->  { "answer": "...", "in_scope": true, "sources": [...] }
    Chi `question` la bat buoc. Khong tim thay quy dinh phu hop -> in_scope=false, sources=[], khong goi LLM.
    Them tuy chon (chi khi SESSION_ENABLED): "session_id" (UUID), "reset"; response them "session_id" (xem API.md).
    POST /ask/stream  (cung body)  ->  text/event-stream (SSE), xem API.md:
      data: {"event": "sources", "in_scope": true, "sources": [...]}
      data: {"event": "delta", "text": "..."}      (lap lai)
      data: [DONE]                                   (hoac data: {"event": "error", "detail": "..."})
BGE-M3 va reranker nap ngay khi server khoi dong (thread nen, tat bang WARMUP=0) va nam trong bo nho cho cac
request sau. Request toi khi dang nap se cho nap xong roi chay tiep. GET /health -> "models": pending/ready/error.
"""
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import iterate_in_threadpool

from . import config as cfg
from . import qdrant_setup, session
from .ingest import embed_consumer, embed_service
from .models import encoders
from .rag import query_model as qm
from .router import guard, router_llm
from .router.router_llm import Action
from .router.small_talk import Intent

log = logging.getLogger("uvicorn.error")
_warmup = {"status": "disabled" if not cfg.WARMUP else "pending"}

WARMUP_TEXT = "Thí sinh được đăng ký tối đa bao nhiêu nguyện vọng?"


def warm_up():
    """Nap model va chay thu mot lan (CUDA khoi tao kernel o lan chay dau): ~16s tren RTX 3050."""
    start = time.perf_counter()
    try:
        encoders.encode_texts([WARMUP_TEXT])
        if cfg.RAG_MODE == "hybrid_rerank":  # mode khac khong can reranker, de danh VRAM
            encoders.rerank_scores(WARMUP_TEXT, [WARMUP_TEXT])
    except Exception:
        _warmup["status"] = "error"
        log.exception("Warmup model that bai, model se duoc nap lai o request dau tien")
        return
    _warmup["status"] = "ready"
    log.info("Warmup model xong sau %.1fs", time.perf_counter() - start)


@asynccontextmanager
async def lifespan(app):
    # Nhanh trong HYBRID_LEGS phai co vector trong collection; thieu thi RuntimeError, server khong khoi dong
    qdrant_setup.validate_legs(qm.get_qdrant())
    # Thread nen: server nhan request ngay (/health tra loi duoc), khong bat FE cho tren connection refused
    if cfg.WARMUP:
        threading.Thread(target=warm_up, name="model-warmup", daemon=True).start()
    if cfg.ROUTER_ENABLED and not cfg.GEMINI_API_KEY:
        log.warning("ROUTER_ENABLED nhung thieu GEMINI_API_KEY: router se loi o moi request va cau hoi di thang vao RAG")
    if cfg.SESSION_ENABLED:
        try:
            session.get_store()  # nap LlamaIndex + tao engine SQLite luc khoi dong, khong doi den request dau tien
        except Exception as e:  # service van khoi dong; request dau tien se thu lai
            log.warning("khong khoi tao duoc session store (%s): se thu lai o request dau tien", type(e).__name__)
    # Khoi dong RabbitMQ consumer de nhan job embed tu file-service (chi chay khi RABBITMQ_URL duoc cau hinh)
    embed_consumer.start_consumer()
    yield


app = FastAPI(title="RAG QA API", lifespan=lifespan)
# Không thêm CORSMiddleware: api-gateway đã gắn CORS, thêm ở đây thì response có 2 header
# Access-Control-Allow-Origin và trình duyệt chặn fetch.


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=None, ge=1, le=10)  # so doan lam ngu canh; mac dinh top_k_rerank
    mode: Optional[Literal["dense", "hybrid", "hybrid_rerank"]] = None  # mac dinh RAG_MODE
    kho: Optional[str] = None
    reference_date: Optional[date] = None  # loc van ban con hieu luc tai ngay nay; mac dinh hom nay
    khoa: Optional[int] = Field(default=None, ge=1990, le=2100)  # khoa cua sinh vien; chi lay van ban ap dung cho khoa nay
    loai_van_ban: Optional[List[str]] = None  # chi tim trong cac loai van ban nay (vd ["thong_bao", "quy_che"])
    session_id: Optional[str] = None  # UUID cua cuoc tro chuyen; bo trong thi server tao moi. Bi bo qua khi SESSION_ENABLED=false
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
    session_id: Optional[str] = None  # chi co khi SESSION_ENABLED


@app.get("/health")
def health():
    return {"status": "ok", "models": _warmup["status"]}


def _question(payload):
    question = payload.question.strip()
    if cfg.GUARD_ENABLED:
        question = guard.normalize_question(question)  # chi con zero width / ky tu dieu khien cung la cau hoi rong
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
    """Moi thu cua mot request: cau hoi chuan hoa, session dang mo (neu bat) va khoa / kho hieu luc."""
    question: str
    session: Any = None  # session.Session; giu lock cho den _end()
    sid: Optional[str] = None
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
    """Chuan hoa cau hoi, mo session (khi SESSION_ENABLED) va gop khoa / kho: gia tri trong request de len gia tri da luu.
    reference_date khong bao gio duoc luu."""
    t = _Turn(_question(payload))
    stored = {}
    if cfg.SESSION_ENABLED:
        t.sid = str(uuid.uuid4()) if payload.session_id is None else session.parse_id(payload.session_id)
        if t.sid is None:
            raise HTTPException(status_code=400, detail="session_id phai la UUID")
        key = session.key_for(t.sid, request.headers.get(cfg.SESSION_USER_HEADER))
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
        session.digest(t.sid) if t.sid else "-", len(t.session.history) // 2 if t.session else 0, action.note or "-",
    )


async def _plan(t):
    """Quyet dinh cho request: Action co `answer` = tra loi co dinh (khong truy van / goi LLM), khong thi di tiep RAG voi `action.query`.

    ROUTER_ENABLED: mot lan goi router (cau hoi + lich su + khoa / kho + co previous_was_clarify) tu suy luan y dinh; khong co
    luat tu khoa nao. Router tat hoac loi (timeout, quota, JSON hong, het ngan sach RPM) thi cau hoi di thang vao RAG, van duoc
    chuan hoa, quet doan trich, prompt co rao va guard dau ra bao ve; khong bao gio tra off_topic / clarify khi router loi."""
    if not cfg.ROUTER_ENABLED:
        return Action(Intent.RAG, query=t.question)
    history = t.session.history if t.session else []
    facts, previous_was_clarify = {"khoa": t.khoa, "kho": t.kho}, bool(t.session and t.session.previous_was_clarify)
    routed = await router_llm.resolve(t.question, history, facts, previous_was_clarify)
    action = (router_llm.apply_policy(routed.decision, t.question, facts, previous_was_clarify)
              if routed.decision is not None else Action(Intent.RAG, query=t.question))
    _log_router(t, routed, action)
    return action


def _pipeline_kwargs(t, payload):
    return dict(top_k=payload.top_k, mode=payload.mode, kho=t.kho, reference_date=payload.reference_date, khoa=t.khoa,
                loai_van_ban=payload.loai_van_ban)


def _history(t):
    return t.session.history if t.session else []


async def _remember(t, answer, intent):
    """Luu luot da hoan tat. Loi luu khong lam hong cau tra loi da co."""
    if t.session:
        try:
            await t.session.add_turn(t.question, answer, intent.value, {"khoa": t.khoa, "kho": t.kho})
        except Exception as e:
            log.warning("session %s: khong luu duoc luot nay (%s)", session.digest(t.sid), type(e).__name__)


@app.post("/ask", response_model=AskResponse, response_model_exclude_unset=True)  # exclude_unset: khong co session thi khong co khoa session_id
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
        if store:
            await _remember(t, answer, intent)
        return AskResponse(answer=answer, in_scope=in_scope, sources=_sources(hits), intent=intent.value,
                           **({"session_id": t.sid} if t.sid else {}))
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
    extra = {**({"session_id": t.sid} if t.sid else {}), **({} if intent is Intent.RAG else {"intent": intent.value})}

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
            if store:  # truoc [DONE]: client dong ket noi ngay khi nhan [DONE] thi doan sau yield cuoi khong chay
                await _remember(t, "".join(parts), intent)
            yield _sse("[DONE]")
        finally:
            _end(t)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(_end, t),  # events() chua kip chay (client ngat som) thi van nha lock
    )


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

    - Tách thành đoạn → BGE-M3 (dense + sparse) → upsert Qdrant.
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

