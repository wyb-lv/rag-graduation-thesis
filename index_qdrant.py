import os
import re
import sys

import numpy as np
from qdrant_client import QdrantClient, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAGRAPHS_PATH = os.path.join(BASE_DIR, "paragraphs.txt")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "paragraph_embeddings.txt")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "paragraphs")

# Tham so HNSW: cung cau hinh da benchmark trong indexing_vector/indexing_vector.ipynb
HNSW_M = 32              # so lien ket moi nut
HNSW_EF_CONSTRUCT = 200  # do rong tim kiem luc dung do thi

_LINE_RE = re.compile(r"^\[\d+\]\s\(([^)]*)\)\s(.*)$")


def load_paragraphs(path=PARAGRAPHS_PATH):
    units = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _LINE_RE.match(line.rstrip("\n"))
            if m:
                units.append({"source": m.group(1), "text": m.group(2)})
    return units


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    units = load_paragraphs()
    embeddings = np.atleast_2d(np.loadtxt(EMBEDDINGS_PATH, dtype="float32"))
    if len(units) != len(embeddings):
        raise SystemExit(f"So doan ({len(units)}) khac so vector ({len(embeddings)}); chay lai export_model.py")

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=embeddings.shape[1], distance=models.Distance.COSINE),
        hnsw_config=models.HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
        # Mac dinh Qdrant chi dung do thi HNSW khi segment >= 20000 KB, nho hon thi quet tuan tu.
        # Ha nguong xuong de tap nho nay van duoc dung do thi HNSW that.
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
    )
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(id=i, vector=embeddings[i].tolist(), payload=units[i])
            for i in range(len(units))
        ],
        wait=True,
    )

    info = client.get_collection(COLLECTION)
    print(f"Da nap {info.points_count} doan vao collection '{COLLECTION}' ({QDRANT_URL})")
    print(f"HNSW: m={info.config.hnsw_config.m}, ef_construct={info.config.hnsw_config.ef_construct}")
    print(f"Trang thai: {info.status} | indexed_vectors_count={info.indexed_vectors_count}")


if __name__ == "__main__":
    main()
