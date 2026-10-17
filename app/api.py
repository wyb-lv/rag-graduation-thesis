"""API HTTP cho FE goi RAG hybrid (dense + sparse + rerank), dung lai logic trong query_model.py.

Chay (trong thu muc Model):
    uvicorn api:app --reload --port 8000

FE goi:
    POST /ask  { "question": "...", "mode": "hybrid_rerank", "kho": "...", "reference_date": "2026-01-31" }
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import config as cfg
from . import encoders
from . import query_model as qm

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
    # Thread nen: server nhan request ngay (/health tra loi duoc), khong bat FE cho tren connection refused
    if cfg.WARMUP:
        threading.Thread(target=warm_up, name="model-warmup", daemon=True).start()
    yield


app = FastAPI(title="RAG QA API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=None, ge=1, le=10)  # so doan lam ngu canh; mac dinh top_k_rerank
    mode: Optional[Literal["dense", "hybrid", "hybrid_rerank"]] = None  # mac dinh RAG_MODE
    kho: Optional[str] = None
    reference_date: Optional[date] = None  # loc van ban con hieu luc tai ngay nay; mac dinh hom nay


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


@app.get("/health")
def health():
    return {"status": "ok", "models": _warmup["status"]}


def _question(payload):
    question = payload.question.strip()
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


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question = _question(payload)
    with _pipeline_errors():
        answer, hits, in_scope = qm.ask(
            question,
            top_k=payload.top_k,
            mode=payload.mode,
            kho=payload.kho,
            reference_date=payload.reference_date,
        )
    return AskResponse(answer=answer, in_scope=in_scope, sources=_sources(hits))


def _sse(data):
    """Mot su kien SSE. json.dumps escape xuong dong nen moi su kien chi co mot dong `data:`."""
    return f"data: {data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/stream")
def ask_stream(payload: AskRequest):
    """Nhu /ask nhung tra cau tra loi dang SSE theo tung doan token.

    Truy van + rerank chay truoc khi gui header, nen loi o buoc nay van tra ma HTTP (400/500/502) nhu /ask.
    Loi khi dang sinh cau tra loi thi gui su kien `error` (status 200 da gui roi) va dong stream, khong co [DONE].
    """
    question = _question(payload)
    with _pipeline_errors():
        hits = qm.find_hits(
            question,
            top_k=payload.top_k,
            mode=payload.mode,
            kho=payload.kho,
            reference_date=payload.reference_date,
        )
        chunks = qm.generate_answer_stream(question, hits) if hits else iter([qm.NO_RESULT_ANSWER])
    sources = [s.model_dump() for s in _sources(hits)]

    def events():
        yield _sse({"event": "sources", "in_scope": bool(hits), "sources": sources})
        try:
            for text in chunks:
                yield _sse({"event": "delta", "text": text})
        except Exception as e:
            yield _sse({"event": "error", "detail": f"Loi khi goi LLM: {e}"})
            return
        yield _sse("[DONE]")

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
