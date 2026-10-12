"""API HTTP cho FE goi RAG (phuong phap paragraph), dung lai logic trong query_model.py.

Chay (trong thu muc Model):
    uvicorn api:app --reload --port 8000

FE goi:
    POST /ask  { "question": "..." }  ->  { "answer": "...", "in_scope": true, "sources": [...] }
    Cau hoi khong co trong tai lieu -> in_scope=false, sources=[], answer la thong bao ngoai pham vi.
"""
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import query_model as qm

app = FastAPI(title="RAG QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=qm.TOP_K_PARAGRAPHS, ge=1, le=10)


class Source(BaseModel):
    source: str
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    in_scope: bool
    sources: List[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question khong duoc de trong")

    try:
        answer, hits, in_scope = qm.ask(question, top_k=payload.top_k)
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Thieu bien moi truong: {e}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Loi khi truy van Qdrant/goi LLM: {e}")

    return AskResponse(answer=answer, in_scope=in_scope, sources=[Source(**h) for h in hits])
