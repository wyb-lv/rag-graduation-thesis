"""Tra loi cau hoi bang RAG (phuong phap paragraph), dua tren du lieu do export_model.py tao ra.

Tim doan van gan nhat bang embedding BAAI/bge-m3 trong Qdrant (index HNSW, chay bang docker),
roi goi LLM (OpenAI-compatible, vi du Groq) de sinh cau tra loi.
Nap du lieu vao Qdrant bang index_qdrant.py (chay docker compose up -d truoc).
"""
import os
import re
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import scipy.linalg  # noqa: F401  (must load BLAS before torch to avoid a Windows DLL/heap crash)
import sklearn.metrics.pairwise  # noqa: F401

import httpx
import torch  # noqa: F401
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

EMBED_MODEL = "BAAI/bge-m3"
TOP_K_PARAGRAPHS = 1
MIN_INTERVAL = 2.2  # giay giua hai request LLM -> ~27 RPM, chua bien an toan duoi 30
# Nguong cosine toi thieu de coi cau hoi la co trong tai lieu (do tren bge-m3 + quy_che.pdf:
# cau trong pham vi >= 0.5, cau ngoai pham vi <= 0.56). Chinh qua bien moi truong SCORE_THRESHOLD.
SCORE_THRESHOLD = 0.5
OUT_OF_SCOPE_ANSWER = (
    "Xin lỗi, câu hỏi này nằm ngoài phạm vi văn bản tôi được cung cấp "
    "nên tôi không thể trả lời."
)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "paragraphs")
HNSW_EF = int(os.environ.get("HNSW_EF", "128"))  # efSearch: lon hon -> recall cao hon, cham hon

LLM_URL = os.environ["LLM_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_API_KEY = os.environ["API_KEY"]

_last_call = 0.0

_model = None
_qdrant = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
        _model.max_seq_length = 512
    return _model


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


def retrieve(question, top_k=TOP_K_PARAGRAPHS):
    q = get_embedding_model().encode([question], normalize_embeddings=True)[0]
    # Distance.COSINE -> score tra ve chinh la cosine, nen SCORE_THRESHOLD giu nguyen y nghia.
    res = get_qdrant().query_points(
        QDRANT_COLLECTION,
        query=q.tolist(),
        limit=top_k,
        with_payload=True,
        search_params=models.SearchParams(hnsw_ef=HNSW_EF),
    )
    return [{"source": p.payload["source"], "text": p.payload["text"], "score": float(p.score)}
            for p in res.points]


def generate_answer(question, hits):
    global _last_call
    context = "\n\n".join(f"[Tai lieu: {h['source']}]\n{h['text']}" for h in hits)
    prompt = (
        "Tra loi cau hoi bang tieng Viet, ngan gon, chi dua tren ngu canh duoi day. "
        "Neu ngu canh khong co thong tin, hay noi khong tim thay.\n\n"
        f"Ngu canh:\n{context}\n\nCau hoi: {question}"
    )
    for attempt in range(5):
        wait = MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        r = httpx.post(
            f"{LLM_URL}/chat/completions", timeout=180,
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={"model": LLM_MODEL, "temperature": 0, "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]},
        )
        if r.status_code == 429:  # van dinh limit -> cho theo Retry-After roi thu lai
            time.sleep(float(r.headers.get("retry-after", 5)) + 1)
            continue
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    raise RuntimeError("LLM tra ve 429 lien tuc, thu lai sau")


def ask(question, top_k=TOP_K_PARAGRAPHS, threshold=SCORE_THRESHOLD):
    """Tra ve (answer, hits, in_scope). Ngoai pham vi tai lieu -> khong goi LLM, hits rong."""
    hits = retrieve(question, top_k)
    relevant = [h for h in hits if h["score"] >= threshold]
    if not relevant:
        return OUT_OF_SCOPE_ANSWER, [], False
    return generate_answer(question, relevant), relevant, True


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    args = sys.argv[1:]
    if args:
        questions = [" ".join(args)]
    else:
        print("Nhap cau hoi (dong trong hoac Ctrl+C de thoat):")
        questions = iter(lambda: input("> ").strip(), "")

    for question in questions:
        if not question:
            continue
        answer, hits, in_scope = ask(question)
        if in_scope:
            print("Nguon:", hits[0]["source"], "| score:", round(hits[0]["score"], 4))
        print(answer)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        pass
