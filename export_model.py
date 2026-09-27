"""Xuat ket qua cuoi cung (phuong phap paragraph) tu compare_sentence_paragraph.ipynb.

Tach PDF thanh cac doan van (paragraph), embedding bang BAAI/bge-m3, va luu:
  - Model/paragraphs.txt          : noi dung tung doan (kem nguon)
  - Model/paragraph_embeddings.txt: vector embedding tuong ung, moi dong la 1 doan
"""
import glob
import os
import re

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import scipy.linalg  # noqa: F401  (must load BLAS before torch to avoid a Windows DLL/heap crash)
import sklearn.metrics.pairwise  # noqa: F401

import numpy as np
import pymupdf
import torch  # noqa: F401
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR
DOCS_DIR = os.path.join(os.path.dirname(BASE_DIR), "sentence_vs_paragraph")
os.makedirs(MODEL_DIR, exist_ok=True)

EMBED_MODEL = "BAAI/bge-m3"
MAX_PARAGRAPH_CHARS = 700


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def load_blocks(path):
    blocks, seen = [], set()
    with pymupdf.open(path) as doc:
        for page in doc:
            for b in page.get_text("blocks"):
                t = clean(b[4])
                if t and t not in seen:
                    seen.add(t)
                    blocks.append(t)
    return blocks


def is_heading(block):
    return len(block) < 100 and not block.rstrip().endswith((".", ":", ";", ",", "!", "?"))


def build_paragraphs(blocks, max_chars=MAX_PARAGRAPH_CHARS):
    paragraphs, current = [], []
    for block in blocks:
        length = sum(len(x) + 1 for x in current)
        if current and ((is_heading(block) and not is_heading(current[-1])) or length + len(block) > max_chars):
            paragraphs.append(" ".join(current))
            current = []
        current.append(block)
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def main():
    doc_paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.pdf")))
    print("Documents:", doc_paths)

    paragraph_units = []
    for path in doc_paths:
        blocks = load_blocks(path)
        for p in build_paragraphs(blocks):
            paragraph_units.append({"text": p, "source": os.path.basename(path)})
    print(f"{len(paragraph_units)} paragraphs")

    model = SentenceTransformer(EMBED_MODEL)
    model.max_seq_length = 512
    embeddings = model.encode(
        [u["text"] for u in paragraph_units], normalize_embeddings=True
    )

    paragraphs_path = os.path.join(MODEL_DIR, "paragraphs.txt")
    with open(paragraphs_path, "w", encoding="utf-8") as f:
        for i, u in enumerate(paragraph_units):
            f.write(f"[{i}] ({u['source']}) {u['text']}\n")

    embeddings_path = os.path.join(MODEL_DIR, "paragraph_embeddings.txt")
    np.savetxt(embeddings_path, embeddings, fmt="%.8f")

    print("Saved:", paragraphs_path)
    print("Saved:", embeddings_path, "shape:", embeddings.shape)


if __name__ == "__main__":
    main()
