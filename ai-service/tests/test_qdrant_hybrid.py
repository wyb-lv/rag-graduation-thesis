"""Chay voi Qdrant that o QDRANT_URL (docker compose up -d). Dung collection tam, khong dong vao collection that."""
import hashlib
import json
import uuid
from datetime import date

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from app import bm25
from app import config as cfg
from app import qdrant_setup
from app.models import encoders
from app.rag import query_model as qm

pytestmark = pytest.mark.integration


def unit(*components):
    v = np.zeros(cfg.EMBED_DIM, dtype="float32")
    for i, w in components:
        v[i] = w
    return v / np.linalg.norm(v)


@pytest.fixture(scope="module")
def client():
    c = QdrantClient(url=cfg.QDRANT_URL, timeout=60)  # mac dinh 5s: de timeout khi may dang tai nang
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
        qdrant_setup.create_collection(client, name)
        if points:
            client.upsert(name, points=list(points), wait=True)
        monkeypatch.setattr(cfg, "QDRANT_COLLECTION", name)
        monkeypatch.setattr(qm, "_qdrant", client)
        return name

    yield make
    for name in created:
        assert name.startswith("test_")  # tuyet doi khong dong vao collection that
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


def test_new_collection_defines_all_three_legs_and_indexes_exist_before_first_upsert(client, make_collection):
    name = make_collection()  # chua upsert diem nao
    info = client.get_collection(name)
    assert info.points_count == 0

    dense = info.config.params.vectors["dense"]
    assert dense.size == 1024 and dense.distance == models.Distance.COSINE
    assert (dense.hnsw_config.m, dense.hnsw_config.ef_construct) == (cfg.HNSW_M, cfg.HNSW_EF_CONSTRUCT)

    sparse = info.config.params.sparse_vectors
    assert set(sparse) == {"sparse", "bm25"}
    assert sparse["sparse"].modifier is None  # trong so hoc duoc cua BGE-M3: khong IDF
    assert sparse["bm25"].modifier == models.Modifier.IDF  # IDF do Qdrant tinh

    schema = info.payload_schema
    assert set(schema) == {"kho", "doi_tuong", "loai_van_ban", "hieu_luc_tu", "hieu_luc_den", "khoa_tu", "khoa_den"}
    for field in ("kho", "doi_tuong", "loai_van_ban"):
        assert schema[field].data_type == models.PayloadSchemaType.KEYWORD
    for field in ("hieu_luc_tu", "hieu_luc_den"):
        assert schema[field].data_type == models.PayloadSchemaType.DATETIME
    for field in ("khoa_tu", "khoa_den"):
        assert schema[field].data_type == models.PayloadSchemaType.INTEGER


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


# ── nhanh BM25 ───────────────────────────────────────────────────────────────

CHUNKS = {
    1: "[Quy chế học bổng – Chương II – Điều 15. Học bổng khuyến khích]\n1. Sinh viên có điểm trung bình học kỳ từ 3,2 được xét học bổng khuyến khích học tập.",
    2: "[Quy chế học bổng – Chương II – Điều 16. Học bổng tài năng]\n1. Sinh viên đạt giải olympic quốc tế được xét học bổng tài năng.",
    3: "[Quy chế đào tạo – Chương III – Điều 20. Xét tốt nghiệp]\n1. Sinh viên được xét tốt nghiệp khi tích lũy đủ 120 tín chỉ.",
    4: "[Quy chế đào tạo – Chương III – Điều 21. Điều kiện dự thi]\n1. Sinh viên vắng quá 20% số buổi học thì không được dự thi.",
    5: "[Quy chế đào tạo – Chương IV – Điều 30. Đăng ký học phần]\n1. Sinh viên đăng ký học phần theo kế hoạch của khoa.",
    6: "[Quy định học phí – Điều 3. Mức thu]\n1. Học phí được thu theo số tín chỉ đăng ký mỗi học kỳ.",
}


def bpt(i, text, **payload):
    """Điểm có đủ ba vector; dense của mỗi điểm trực giao với nhau nên nhánh dense không giúp tìm đúng chunk."""
    vector = {
        "dense": unit((i, 1)).tolist(),
        "sparse": models.SparseVector(indices=[1000 + i], values=[1.0]),
        "bm25": bm25.bm25_doc_vector(text),
    }
    return models.PointStruct(id=i, vector=vector, payload={"source": "quy_che.pdf", "text": text, **payload})


@pytest.fixture
def chunk_collection(make_collection):
    return make_collection([bpt(i, text, loai_van_ban="quy_che") for i, text in CHUNKS.items()])


def no_embedder(monkeypatch):
    def boom(texts, batch_size=None):
        raise AssertionError("truy van chi-bm25 khong duoc nap BGE-M3")

    monkeypatch.setattr(encoders, "encode_texts", boom)


@pytest.mark.parametrize(
    "question,expected",
    [("olympic", 2), ("Điều 15", 1), ("Điều 20", 3), ("điều 21", 4), ("học phí", 6)],
)
def test_bm25_leg_ranks_the_rare_token_or_article_first(chunk_collection, monkeypatch, question, expected):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("bm25",))
    no_embedder(monkeypatch)
    hits = qm.retrieve(question, mode="hybrid")
    assert hits[0]["id"] == expected
    assert hits[0]["retrieval_score"] > 0


def test_article_bigram_picks_the_right_chunk_among_partial_matches(chunk_collection, monkeypatch):
    # "Điều 20": chunk 4 khớp "điều" và "20" (20%) nhưng chỉ chunk 3 có bigram điều_20
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("bm25",))
    no_embedder(monkeypatch)
    hits = qm.retrieve("Điều 20", mode="hybrid")
    assert hits[0]["id"] == 3 and {h["id"] for h in hits} >= {3, 4}


def test_three_leg_hybrid_still_surfaces_the_lexical_match(chunk_collection, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("dense", "sparse", "bm25"))
    stub_query(monkeypatch, unit((200, 1)), {"99999": 1.0})  # dense trực giao với mọi chunk, sparse không khớp gì
    hits = qm.retrieve("olympic", mode="hybrid")
    assert hits[0]["id"] == 2
    scores = [h["retrieval_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)  # điểm RRF


def test_default_legs_ignore_bm25_even_when_the_collection_has_it(chunk_collection, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("dense", "sparse"))
    stub_query(monkeypatch, unit((6, 1)), {"1006": 1.0})
    assert qm.retrieve("olympic", mode="hybrid")[0]["id"] == 6  # khớp dense + sparse của chunk 6, không phải "olympic"


META = [
    (1, dict(loai_van_ban="quy_che", kho="A", hieu_luc_tu="2020-01-01", hieu_luc_den="2099-12-31", khoa_tu=2020)),
    (2, dict(loai_van_ban="thong_bao", kho="A", hieu_luc_tu="2020-01-01", hieu_luc_den="2021-12-31")),  # hết hiệu lực 2021
    (3, dict(loai_van_ban="thong_bao", kho="B", hieu_luc_tu="2030-01-01")),  # chưa có hiệu lực
    (4, dict(loai_van_ban="thong_bao", kho="A", khoa_den=2019)),  # chỉ áp dụng khóa <= 2019
    (5, dict(loai_van_ban="bieu_mau")),  # không có kho / ngày / khóa
]
LEG_SETS = [("bm25",), ("dense", "bm25"), ("sparse", "bm25"), ("dense", "sparse", "bm25")]


@pytest.mark.parametrize("legs", LEG_SETS, ids=lambda legs: "+".join(legs))
def test_every_leg_respects_kho_validity_khoa_and_loai_van_ban(make_collection, monkeypatch, legs):
    text = "[Quy chế học bổng – Điều 16]\nSinh viên đạt giải olympic quốc tế."
    make_collection([
        models.PointStruct(
            id=i,
            vector={
                "dense": unit((0, 1)).tolist(),
                "sparse": models.SparseVector(indices=[10], values=[1.0]),
                "bm25": bm25.bm25_doc_vector(text),
            },
            payload={"source": "quy_che.pdf", "text": text, **meta},
        )
        for i, meta in META
    ])
    monkeypatch.setattr(cfg, "HYBRID_LEGS", legs)
    stub_query(monkeypatch, unit((0, 1)), {"10": 1.0})  # mọi nhánh đều khớp CẢ 5 điểm: filter rò sẽ lộ ra ngay
    if legs == ("bm25",):
        no_embedder(monkeypatch)

    def ids(**kw):
        return {h["id"] for h in qm.retrieve("olympic", mode="hybrid", **kw)}

    ref = date(2025, 6, 1)
    assert ids(reference_date=ref) == {1, 4, 5}
    assert ids(reference_date=date(2021, 6, 1)) == {1, 2, 4, 5}
    assert ids(reference_date=ref, kho="A") == {1, 4}
    assert ids(reference_date=ref, khoa=2024) == {1, 5}
    assert ids(reference_date=ref, khoa=2018) == {4, 5}
    assert ids(reference_date=ref, loai_van_ban=["thong_bao"]) == {4}
    assert ids(reference_date=ref, loai_van_ban=["quy_che", "bieu_mau"]) == {1, 5}
    assert ids(reference_date=ref, loai_van_ban=["thong_bao"], khoa=2024) == set()
    assert ids(reference_date=ref, loai_van_ban=["khong_ton_tai"]) == set()
    assert ids(reference_date=ref, kho="A", khoa=2024, loai_van_ban=["quy_che"]) == {1}


def test_no_candidate_after_filtering_with_bm25_answers_without_calling_groq(chunk_collection, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("dense", "sparse", "bm25"))
    stub_query(monkeypatch, unit((0, 1)), {"1001": 1.0})

    def boom(*a, **k):
        raise AssertionError("khong duoc goi Groq khi khong co ung vien")

    monkeypatch.setattr(qm.groq_client, "chat", boom)
    answer, hits, in_scope = qm.ask("olympic", mode="hybrid", loai_van_ban=["khong_ton_tai"])
    assert (answer, hits, in_scope) == (qm.NO_RESULT_ANSWER, [], False)


def test_qdrant_accepts_empty_and_missing_bm25_vectors(client, make_collection, monkeypatch):
    name = make_collection([bpt(1, CHUNKS[2])])
    empty = models.PointStruct(
        id=2,
        vector={"dense": unit((2, 1)).tolist(), "bm25": models.SparseVector(indices=[], values=[])},
        payload={"source": "s", "text": "?!"},
    )
    missing = models.PointStruct(id=3, vector={"dense": unit((3, 1)).tolist()}, payload={"source": "s", "text": "?!"})
    client.upsert(name, points=[empty, missing], wait=True)
    assert client.count(name, exact=True).count == 3
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("bm25",))
    no_embedder(monkeypatch)
    assert [h["id"] for h in qm.retrieve("olympic", mode="hybrid")] == [1]  # chỉ điểm có token bm25 khớp


def test_validate_legs_against_real_collections(client, make_collection):
    full = make_collection()
    qdrant_setup.validate_legs(client, full, ("dense", "sparse", "bm25"))  # không ném lỗi

    legacy = f"test_hybrid_{uuid.uuid4().hex[:8]}"  # dạng collection cũ: chưa có bm25
    client.create_collection(
        legacy,
        vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    try:
        qdrant_setup.validate_legs(client, legacy, ("dense", "sparse"))
        with pytest.raises(RuntimeError, match="bm25") as err:
            qdrant_setup.validate_legs(client, legacy, ("dense", "sparse", "bm25"))
        assert "migrate_add_bm25" in str(err.value) and legacy in str(err.value)
        assert qdrant_setup.has_bm25(client, full) and not qdrant_setup.has_bm25(client, legacy)
    finally:
        client.delete_collection(legacy)
