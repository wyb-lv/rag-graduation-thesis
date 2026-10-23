"""API HTTP cho FE goi RAG hybrid (dense + sparse + rerank), dung lai logic trong query_model.py.

Chay (trong thu muc Model):
    uvicorn api:app --reload --port 8000

FE goi:
    POST /ask  { "question": "...", "mode": "hybrid_rerank", "kho": "...", "reference_date": "2026-01-31",
                 "loai_van_ban": ["thong_bao"] }
      ->  { "answer": "...", "in_scope": true, "sources": [...] }
    Chi `question` la bat buoc. Khong tim thay quy dinh phu hop -> in_scope=false, sources=[], khong goi LLM.
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
from contextlib import asynccontextmanager, contextmanager
from datetime import date
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import config as cfg
from . import guard, qdrant_setup, small_talk
from .ingest import embed_consumer, embed_service
from .models import encoders
from .rag import query_model as qm
from .small_talk import Intent

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
    # rag | greeting | thanks | goodbye | capability | meta | blocked (guard.py / small_talk.py)
    intent: str = Intent.RAG.value


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


def _gate(question):
    """Guard dau vao + small talk router, dung chung cho /ask va /ask/stream (truoc moi buoc RAG).

    Tra ve (cau hoi dung cho RAG, reply); reply = (answer, in_scope, intent) khi khong can truy van / goi LLM, con lai None."""
    if not cfg.GUARD_ENABLED:
        return question, None
    enforce = cfg.GUARD_MODE == "enforce"
    verdict = guard.inspect_question(question)
    if verdict.level == "suspicious" and enforce:
        question = guard.strip_delimiters(question)
    if (verdict.level == "block" and enforce) or not question:
        intent = Intent.BLOCKED
    else:
        intent, question = small_talk.route(question)
    guard.log_decision(question, intent, verdict)
    if intent is Intent.BLOCKED:
        return question, (cfg.GUARD_REFUSAL_ANSWER, False, intent)
    if intent is not Intent.RAG:
        return question, (small_talk.reply_for(intent), True, intent)
    return question, None


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


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question, reply = _gate(_question(payload))
    if reply:
        answer, in_scope, intent = reply
        return AskResponse(answer=answer, in_scope=in_scope, sources=[], intent=intent.value)
    with _pipeline_errors():
        answer, hits, in_scope = qm.ask(
            question,
            top_k=payload.top_k,
            mode=payload.mode,
            kho=payload.kho,
            reference_date=payload.reference_date,
            khoa=payload.khoa,
            loai_van_ban=payload.loai_van_ban,
        )
    intent = Intent.BLOCKED if answer == cfg.GUARD_REFUSAL_ANSWER else Intent.RAG  # guard dau ra da chan cau tra loi
    return AskResponse(answer=answer, in_scope=in_scope, sources=_sources(hits), intent=intent.value)


def _sse(data):
    """Mot su kien SSE. json.dumps escape xuong dong nen moi su kien chi co mot dong `data:`."""
    return f"data: {data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream(payload: AskRequest):
    """Nhu /ask nhung tra cau tra loi dang SSE theo tung doan token.

    Truy van + rerank chay truoc khi gui header, nen loi o buoc nay van tra ma HTTP (400/500/502) nhu /ask.
    Loi khi dang sinh cau tra loi thi gui su kien `error` (status 200 da gui roi) va dong stream, khong co [DONE].
    """
    question, reply = _gate(_question(payload))
    if reply:  # chao hoi / bi chan: cung hop dong SSE, sources rong + dung mot delta
        answer, in_scope, intent = reply
        hits, chunks = [], iter([answer])
    else:
        with _pipeline_errors():
            hits = qm.find_hits(
                question,
                top_k=payload.top_k,
                mode=payload.mode,
                kho=payload.kho,
                reference_date=payload.reference_date,
                khoa=payload.khoa,
                loai_van_ban=payload.loai_van_ban,
            )
            chunks = qm.generate_answer_stream(question, hits) if hits else iter([qm.NO_RESULT_ANSWER])
        in_scope, intent = bool(hits), Intent.RAG
    sources = [s.model_dump() for s in _sources(hits)]
    # `intent` chi them vao su kien khi khac rag, de su kien cua cac cau hoi RAG giu nguyen tung byte
    extra = {} if intent is Intent.RAG else {"intent": intent.value}

    def events():
        yield _sse({"event": "sources", "in_scope": in_scope, "sources": sources, **extra})
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

