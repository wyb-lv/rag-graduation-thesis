"""Nap paragraphs.txt vao collection Qdrant MOI (named vectors: dense + sparse), ma hoa bang BGE-M3 (FlagEmbedding).

Thu tu: tao collection -> tao payload index -> encode + upsert.
Collection cu (QDRANT_LEGACY_COLLECTION) khong bi ghi/xoa.
Chay: python index_qdrant.py [--recreate]   (docker compose up -d truoc)
"""
import argparse
import re
import sys

from qdrant_client import QdrantClient, models

import config as cfg

_LINE_RE = re.compile(r"^\[\d+\]\s\(([^)]*)\)\s(.*)$")


def load_paragraphs(path=cfg.PARAGRAPHS_PATH):
    units = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _LINE_RE.match(line.rstrip("\n"))
            if m:
                units.append({"source": m.group(1), "text": m.group(2)})
    return units


def legacy_hnsw_params(client):
    """(m, ef_construct) cua collection cu; khong doc duoc thi dung gia tri mac dinh trong config."""
    try:
        info = client.get_collection(cfg.QDRANT_LEGACY_COLLECTION)
    except Exception:  # collection cu khong ton tai / Qdrant tu choi
        return cfg.HNSW_M, cfg.HNSW_EF_CONSTRUCT
    vectors = info.config.params.vectors
    # Collection cu dung 1 vector khong ten: HNSW nam o config.hnsw_config (hoac VectorParams.hnsw_config neu co).
    hnsw = getattr(vectors, "hnsw_config", None) or info.config.hnsw_config
    return hnsw.m, hnsw.ef_construct


def create_collection(client, name, m, ef_construct):
    client.create_collection(
        name,
        vectors_config={
            cfg.DENSE_VECTOR_NAME: models.VectorParams(
                size=cfg.EMBED_DIM,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
            )
        },
        # Khong dat modifier=IDF: trong so lexical cua BGE-M3 da duoc hoc san.
        sparse_vectors_config={cfg.SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=cfg.INDEXING_THRESHOLD),
    )


def create_payload_indexes(client, name):
    """Tao index payload TRUOC khi nap du lieu."""
    client.create_payload_index(
        name,
        cfg.FIELD_KHO,
        field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
    )
    client.create_payload_index(name, cfg.FIELD_DOI_TUONG, field_schema=models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, cfg.FIELD_HIEU_LUC_TU, field_schema=models.PayloadSchemaType.DATETIME)
    client.create_payload_index(name, cfg.FIELD_HIEU_LUC_DEN, field_schema=models.PayloadSchemaType.DATETIME)


def ingest(client, name, units):
    # Import muon: cac buoc tao collection/index o tren khong can nap model.
    import encoders

    batch = cfg.UPSERT_BATCH_SIZE
    for start in range(0, len(units), batch):
        chunk = units[start:start + batch]
        dense, lexical = encoders.encode_texts([u["text"] for u in chunk])
        points = []
        for offset, unit in enumerate(chunk):
            vector = {cfg.DENSE_VECTOR_NAME: dense[offset].tolist()}
            sparse = encoders.to_sparse_vector(lexical[offset])
            if sparse.indices:  # doan khong co token nao (rong) thi chi luu dense
                vector[cfg.SPARSE_VECTOR_NAME] = sparse
            points.append(models.PointStruct(id=start + offset, vector=vector, payload=unit))
        client.upsert(name, points=points, wait=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recreate", action="store_true",
                        help="xoa va tao lai collection MOI neu da ton tai (khong bao gio dong vao collection cu)")
    args = parser.parse_args()

    name = cfg.QDRANT_COLLECTION
    if name == cfg.QDRANT_LEGACY_COLLECTION:
        raise SystemExit(f"QDRANT_COLLECTION='{name}' trung collection cu; hay dung ten khac (vd quy_dinh_hybrid)")

    units = load_paragraphs()
    if not units:
        raise SystemExit(f"Khong doc duoc doan nao tu {cfg.PARAGRAPHS_PATH}; chay lai export_model.py")

    client = QdrantClient(url=cfg.QDRANT_URL)
    if client.collection_exists(name):
        if not args.recreate:
            raise SystemExit(f"Collection '{name}' da ton tai; them --recreate de xoa va nap lai")
        client.delete_collection(name)

    m, ef_construct = legacy_hnsw_params(client)
    create_collection(client, name, m, ef_construct)
    create_payload_indexes(client, name)
    ingest(client, name, units)

    info = client.get_collection(name)
    print(f"Da nap {info.points_count} doan vao collection '{name}' ({cfg.QDRANT_URL})")
    print(f"HNSW (dense): m={m}, ef_construct={ef_construct}")
    print(f"Payload index: {sorted(info.payload_schema)}")
    print(f"Trang thai: {info.status} | indexed_vectors_count={info.indexed_vectors_count}")


if __name__ == "__main__":
    main()
