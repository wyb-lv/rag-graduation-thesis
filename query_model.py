"""Tra loi cau hoi bang RAG (phuong phap paragraph), dua tren du lieu do export_model.py tao ra.

Doc lai Model/paragraphs.txt va Model/paragraph_embeddings.txt, tim doan van gan nhat
bang embedding BAAI/bge-m3, roi goi LLM (OpenAI-compatible, vi du Groq) de sinh cau tra loi.
Day la phan LLM cua compare_sentence_paragraph.ipynb, tach ra thanh script dung lai duoc.
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
import numpy as np
import torch  # noqa: F401
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

EMBED_MODEL = "BAAI/bge-m3"
TOP_K_PARAGRAPHS = 1
MIN_INTERVAL = 2.2  # giay giua hai request LLM -> ~27 RPM, chua bien an toan duoi 30
# Nguong cosine toi thieu de coi cau hoi la co trong tai lieu (do tren bge-m3 + quy_che.pdf:
# cau trong pham vi >= 0.58, cau ngoai pham vi <= 0.56). Chinh qua bien moi truong SCORE_THRESHOLD.
SCORE_THRESHOLD = 0.5
OUT_OF_SCOPE_ANSWER = (
    "Xin lỗi, câu hỏi này nằm ngoài phạm vi văn bản tôi được cung cấp "
    "nên tôi không thể trả lời."
)

PARAGRAPHS_PATH = os.path.join(BASE_DIR, "paragraphs.txt")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "paragraph_embeddings.txt")

LLM_URL = os.environ["LLM_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]
LLM_API_KEY = os.environ["API_KEY"]

_LINE_RE = re.compile(r"^\[\d+\]\s\(([^)]*)\)\s(.*)$")
_last_call = 0.0

_model = None
_units = None
_embeddings = None


def load_paragraphs(path=PARAGRAPHS_PATH):
    units = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _LINE_RE.match(line.rstrip("\n"))
            if m:
                units.append({"source": m.group(1), "text": m.group(2)})
    return units


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
        _model.max_seq_length = 512
    return _model


def get_paragraph_index():
    global _units, _embeddings
    if _units is None:
        _units = load_paragraphs()
        _embeddings = np.loadtxt(EMBEDDINGS_PATH)
    return _units, _embeddings


def retrieve(question, top_k=TOP_K_PARAGRAPHS):
    model = get_embedding_model()
    units, embeddings = get_paragraph_index()
    q = model.encode([question], normalize_embeddings=True)
    scores = cosine_similarity(q, embeddings)[0]
    idx = np.argsort(scores)[::-1][:top_k]
    return [{**units[i], "score": float(scores[i])} for i in idx]


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
