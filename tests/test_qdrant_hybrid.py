"""Chay voi Qdrant that o QDRANT_URL (docker compose up -d). Dung collection tam, khong dong vao collection that."""
import hashlib
import json
import sys
import uuid
from datetime import date

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from app import config as cfg
from app.models import encoders
import index_qdrant
from app.rag import query_model as qm

pytestmark = pytest.mark.integration


def unit(*components):
    v = np.zeros(cfg.EMBED_DIM, dtype="float32")
    for i, w in components:
        v[i] = w
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def client():
    c = QdrantClient(url=cfg.QDRANT_URL)
    try:
        c.get_collections()
    except Exception as e:
        pytest.skip(f"Khong ket noi duoc Qdrant {cfg.QDRANT_URL}: {e}")
    return c


def legacy_fingerprint(client):
    name = cfg.QDRANT_LEGACY_COLLECTION
    if not client.collection_exists(name):
        return None
    info = client.get_collection(name)
    points, _ = client.scroll(name, limit=10_000, with_payload=True, with_vectors=True)
    data = json.dumps([[p.id, p.payload, p.vector] for p in sorted(points, key=lambda p: p.id)], sort_keys=True)
    config = json.dumps(info.config.model_dump(mode="json"), sort_keys=True)
    return info.points_count, hashlib.sha256(data.encode()).hexdigest(), hashlib.sha256(config.encode()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def legacy_collection_is_untouched(client):
    before = legacy_fingerprint(client)
    yield
    assert legacy_fingerprint(client) == before


@pytest.fixture
def make_collection(client, monkeypatch):
    """Tao collection tam theo dung ham production, nap `points`, tro cau hinh ve do; tu xoa khi xong."""
    created = []

    def make(points=()):
        name = f"test_hybrid_{uuid.uuid4().hex[:8]}"
        created.append(name)
        m, ef = index_qdrant.legacy_hnsw_params(client)
        index_qdrant.create_collection(client, name, m, ef)
        index_qdrant.create_payload_indexes(client, name)
        if points:
            client.upsert(name, points=list(points), wait=True)
        monkeypatch.setattr(cfg, "QDRANT_COLLECTION", name)
        monkeypatch.setattr(qm, "_qdrant", client)
        return name

    yield make
    for name in created:
        assert name != cfg.QDRANT_LEGACY_COLLECTION
        client.delete_collection(name)


def pt(i, dense, sparse=None, **payload):
    vector = {"dense": dense.tolist()}
    if sparse:
        vector["sparse"] = models.SparseVector(indices=list(sparse), values=list(sparse.values()))
    return models.PointStruct(id=i, vector=vector, payload={"source": "quy_che.pdf", "text": f"doan {i}", **payload})


def stub_query(monkeypatch, dense, lexical):
    monkeypatch.setattr(
        encoders, "encode_texts", lambda texts, batch_size=None: (np.stack([dense]), [lexical])
    )


def test_collection_schema_matches_spec_and_indexes_exist_before_first_upsert(client, make_collection):
    name = make_collection()  # chua upsert diem nao
    info = client.get_collection(name)
    assert info.points_count == 0

    dense = info.config.params.vectors["dense"]
    assert dense.size == 1024 and dense.distance == models.Distance.COSINE
    if client.collection_exists(cfg.QDRANT_LEGACY_COLLECTION):
        legacy = client.get_collection(cfg.QDRANT_LEGACY_COLLECTION).config.hnsw_config
        expected = (legacy.m, legacy.ef_construct)
    else:
        expected = (cfg.HNSW_M, cfg.HNSW_EF_CONSTRUCT)
    assert (dense.hnsw_config.m, dense.hnsw_config.ef_construct) == expected

    sparse = info.config.params.sparse_vectors["sparse"]
    assert sparse.modifier is None  # khong IDF

    schema = info.payload_schema
    assert set(schema) == {"kho", "doi_tuong", "hieu_luc_tu", "hieu_luc_den"}
    assert schema["kho"].data_type == models.PayloadSchemaType.KEYWORD and schema["kho"].params.is_tenant is True
    assert schema["doi_tuong"].data_type == models.PayloadSchemaType.KEYWORD
    assert schema["hieu_luc_tu"].data_type == models.PayloadSchemaType.DATETIME
    assert schema["hieu_luc_den"].data_type == models.PayloadSchemaType.DATETIME


def test_hybrid_query_fuses_dense_and_sparse_with_rrf(client, make_collection, monkeypatch):
    # 10 diem dem (dense cosine ~0.8) day diem "chi khop sparse" xuong cuoi danh sach dense,
    # de thu tu RRF khong bi nhap nhang boi xep hang hoa nhau o duoi danh sach.
    fillers = [pt(10 + i, unit((0, 1), (20 + i, 0.75))) for i in range(10)]
    make_collection([
        pt(1, unit((0, 1)), None),                                  # dense hang 1, khong co sparse
        pt(2, unit((5, 1)), {10: 1.0}),                             # sparse hang 1, dense o cuoi
        pt(3, unit((0, 1), (1, 0.3)), {10: 0.8, 11: 0.5}),          # hang 2 o ca hai danh sach
        pt(4, unit((7, 1)), {99: 1.0}),                             # khong khop gi
        *fillers,
    ])
    stub_query(monkeypatch, unit((0, 1)), {"10": 1.0})

    hybrid = qm.retrieve("q", mode="hybrid")
    # RRF: 3 = 1/62 + 1/62 > 2 = 1/61 + ~1/73 > 1 = 1/61 > cac diem dem; 4 khong co o sparse nen cuoi cung
    assert [h["id"] for h in hybrid[:3]] == [3, 2, 1]
    assert hybrid[-1]["id"] == 4
    assert len(hybrid) == 14 and all(h["payload"]["source"] == "quy_che.pdf" for h in hybrid)
    scores = [h["retrieval_score"] for h in hybrid]
    assert scores == sorted(scores, reverse=True)

    dense_only = qm.retrieve("q", mode="dense")  # dense chay duoc tren collection moi qua vector `dense`
    assert [h["id"] for h in dense_only[:2]] == [1, 3]
    assert dense_only[0]["retrieval_score"] == pytest.approx(1.0, abs=1e-3)  # cosine


def test_empty_sparse_query_still_returns_dense_results(client, make_collection, monkeypatch):
    make_collection([pt(1, unit((0, 1)), {10: 1.0}), pt(2, unit((3, 1)), {11: 1.0})])
    stub_query(monkeypatch, unit((0, 1)), {})
    assert qm.retrieve("q", mode="hybrid")[0]["id"] == 1


def test_payload_filters_on_kho_and_effective_date(client, make_collection, monkeypatch):
    make_collection([
        pt(1, unit((0, 1)), {10: 1.0}, kho="KHOA_A", hieu_luc_tu="2020-01-01", hieu_luc_den="2099-12-31"),
        # het hieu luc nam 2021
        pt(2, unit((1, 1)), {10: 0.9}, kho="KHOA_B", hieu_luc_tu="2020-01-01", hieu_luc_den="2021-12-31"),
        # chua co hieu luc cho den 2030
        pt(3, unit((2, 1)), {10: 0.8}, kho="KHOA_A", hieu_luc_tu="2030-01-01"),
        pt(4, unit((3, 1)), {10: 0.7}),                                  # payload cu: khong co kho/ngay
    ])
    stub_query(monkeypatch, unit((0, 1)), {"10": 1.0})

    def ids(mode, **kw):
        return {h["id"] for h in qm.retrieve("q", mode=mode, **kw)}

    for mode in ("dense", "hybrid"):
        assert ids(mode, reference_date=date(2025, 6, 1)) == {1, 4}
        assert ids(mode, reference_date=date(2021, 6, 1)) == {1, 2, 4}
        assert ids(mode, reference_date=date(2035, 1, 1), kho="KHOA_A") == {1, 3}
        assert ids(mode, reference_date=date(2025, 6, 1), kho="KHOA_B") == set()


def test_no_candidates_after_filtering_answers_without_calling_groq(client, make_collection, monkeypatch):
    make_collection([pt(1, unit((0, 1)), {10: 1.0}, kho="KHOA_A")])
    stub_query(monkeypatch, unit((0, 1)), {"10": 1.0})

    def boom(*a, **k):
        raise AssertionError("khong duoc goi Groq khi khong co ung vien")

    monkeypatch.setattr(qm.groq_client, "chat", boom)
    answer, hits, in_scope = qm.ask("q", mode="hybrid_rerank", kho="KHOA_KHONG_TON_TAI")
    assert (answer, hits, in_scope) == (qm.NO_RESULT_ANSWER, [], False)


def test_ingestion_creates_indexes_before_upsert_and_keeps_payload_and_ids(client, monkeypatch):
    name = f"test_hybrid_{uuid.uuid4().hex[:8]}"
    units = [{"source": "quy_che.pdf", "text": f"doan {i}"} for i in range(3)]
    order = []

    class Recording(QdrantClient):
        def create_collection(self, *a, **k):
            order.append("create_collection")
            return super().create_collection(*a, **k)

        def create_payload_index(self, collection_name, field_name, *a, **k):
            order.append(f"index:{field_name}")
            return super().create_payload_index(collection_name, field_name, *a, **k)

        def upsert(self, *a, **k):
            order.append("upsert")
            return super().upsert(*a, **k)

    monkeypatch.setattr(index_qdrant, "QdrantClient", Recording)
    monkeypatch.setattr(index_qdrant, "load_paragraphs", lambda: units)
    monkeypatch.setattr(cfg, "QDRANT_COLLECTION", name)
    monkeypatch.setattr(sys, "argv", ["index_qdrant.py"])
    monkeypatch.setattr(
        encoders, "encode_texts",
        lambda texts, batch_size=None: (np.stack([unit((i, 1)) for i, _ in enumerate(texts)]),
                                        [{str(100 + i): 0.5} for i, _ in enumerate(texts)]),
    )
    try:
        index_qdrant.main()
        assert order[0] == "create_collection"
        assert sorted(order[1:5]) == ["index:doi_tuong", "index:hieu_luc_den", "index:hieu_luc_tu", "index:kho"]
        assert order[5:] == ["upsert"]

        points, _ = client.scroll(name, limit=10, with_payload=True, with_vectors=True)
        assert sorted(p.id for p in points) == [0, 1, 2]  # id = vi tri doan, nhu schema cu
        for p in points:
            assert p.payload == units[p.id]
            assert len(p.vector["dense"]) == 1024 and p.vector["sparse"].indices == [100 + p.id]
    finally:
        if client.collection_exists(name):
            client.delete_collection(name)


def test_ingestion_refuses_to_target_the_legacy_collection(client, monkeypatch):
    monkeypatch.setattr(cfg, "QDRANT_COLLECTION", cfg.QDRANT_LEGACY_COLLECTION)
    monkeypatch.setattr(sys, "argv", ["index_qdrant.py", "--recreate"])
    with pytest.raises(SystemExit):
        index_qdrant.main()
