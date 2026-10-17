"""Xuat ket qua cuoi cung (phuong phap paragraph) tu compare_sentence_paragraph.ipynb.

Tach PDF thanh cac doan van (paragraph) va luu:
  - Model/paragraphs.txt: noi dung tung doan (kem nguon)
Embedding (dense + sparse) do index_qdrant.py tinh khi nap vao Qdrant, nen khong con paragraph_embeddings.txt.
"""
import glob
import os
import re
import sys

# Root project = thu muc cha cua scripts/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pymupdf

DOCS_DIR = os.path.join(ROOT, "data", "docs")
OUTPUT_PATH = os.path.join(ROOT, "data", "paragraphs.txt")

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
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, u in enumerate(paragraph_units):
            f.write(f"[{i}] ({u['source']}) {u['text']}\n")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
