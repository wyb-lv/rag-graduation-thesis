"""Chay voi Qdrant that o QDRANT_URL (docker compose up -d). Dung collection tam, khong dong vao collection that."""
import hashlib
import json
import uuid
from datetime import date

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

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


def test_new_collection_defines_both_legs_and_indexes_exist_before_first_upsert(client, make_collection):
    name = make_collection()  # chua upsert diem nao
    info = client.get_collection(name)
    assert info.points_count == 0

    dense = info.config.params.vectors["dense"]
    assert dense.size == 1024 and dense.distance == models.Distance.COSINE
    assert (dense.hnsw_config.m, dense.hnsw_config.ef_construct) == (cfg.HNSW_M, cfg.HNSW_EF_CONSTRUCT)

    sparse = info.config.params.sparse_vectors
    assert set(sparse) == {"sparse"}
    assert sparse["sparse"].modifier is None  # trong so hoc duoc cua BGE-M3: khong IDF

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


# ── filter metadata (kho, hieu luc, khoa, loai_van_ban) tren tung nhanh, voi mot hoac hai cau ──

META = [
    (1, dict(loai_van_ban="quy_che", kho="A", hieu_luc_tu="2020-01-01", hieu_luc_den="2099-12-31", khoa_tu=2020)),
    (2, dict(loai_van_ban="thong_bao", kho="A", hieu_luc_tu="2020-01-01", hieu_luc_den="2021-12-31")),  # hết hiệu lực 2021
    (3, dict(loai_van_ban="thong_bao", kho="B", hieu_luc_tu="2030-01-01")),  # chưa có hiệu lực
    (4, dict(loai_van_ban="thong_bao", kho="A", khoa_den=2019)),  # chỉ áp dụng khóa <= 2019
    (5, dict(loai_van_ban="bieu_mau")),  # không có kho / ngày / khóa
]
LEG_SETS = [("dense",), ("sparse",), ("dense", "sparse")]


@pytest.mark.parametrize("alt", [None, "cau viet lai"], ids=["mot_cau", "hai_cau"])
@pytest.mark.parametrize("legs", LEG_SETS, ids=lambda legs: "+".join(legs))
def test_every_leg_respects_kho_validity_khoa_and_loai_van_ban(make_collection, monkeypatch, legs, alt):
    text = "[Quy chế học bổng – Điều 16]\nSinh viên đạt giải olympic quốc tế."
    make_collection([
        models.PointStruct(
            id=i,
            vector={
                "dense": unit((0, 1)).tolist(),
                "sparse": models.SparseVector(indices=[10], values=[1.0]),
            },
            payload={"source": "quy_che.pdf", "text": text, **meta},
        )
        for i, meta in META
    ])
    monkeypatch.setattr(cfg, "HYBRID_LEGS", legs)
    stub_query(monkeypatch, unit((0, 1)), {"10": 1.0})  # mọi nhánh đều khớp CẢ 5 điểm: filter rò sẽ lộ ra ngay

    def ids(**kw):
        return {h["id"] for h in qm.retrieve("olympic", mode="hybrid", alt_question=alt, **kw)}

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


def test_validate_legs_against_real_collections(client, make_collection):
    full = make_collection()
    qdrant_setup.validate_legs(client, full, ("dense", "sparse"))  # không ném lỗi

    dense_only = f"test_hybrid_{uuid.uuid4().hex[:8]}"  # dạng collection cũ: chưa có sparse
    client.create_collection(dense_only, vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)})
    try:
        qdrant_setup.validate_legs(client, dense_only, ("dense",))
        with pytest.raises(RuntimeError, match="sparse") as err:
            qdrant_setup.validate_legs(client, dense_only, ("dense", "sparse"))
        assert dense_only in str(err.value)
    finally:
        client.delete_collection(dense_only)


# ---- hai cau truy van (cau goc + cau viet lai): dense + sparse cua ca hai, mot lan RRF, gop trung theo id ----

def stub_by_sentence(monkeypatch, table):
    """encode_texts tra vector theo noi dung cau: table = {cau: (dense, lexical)}."""
    monkeypatch.setattr(
        encoders, "encode_texts",
        lambda texts, batch_size=None: (np.stack([table[t][0] for t in texts]), [table[t][1] for t in texts]),
    )


def two_sentence_points():
    fillers = [pt(10 + i, unit((0, 1), (20 + i, 0.75)), kho="A") for i in range(4)]  # gan "q" ve dense, day diem chi khop "alt" ra khoi top 3
    return [
        pt(1, unit((0, 1)), {10: 1.0}, kho="A"),                          # chi khop "q"
        pt(2, unit((5, 1)), {50: 1.0}, kho="B"),                          # chi khop "alt"
        pt(3, unit((0, 1), (5, 1)), {10: 0.5, 50: 0.5}, kho="A"),         # khop vua phai ca hai
        *fillers,
    ]


TWO_SENTENCES = {"q": (unit((0, 1)), {"10": 1.0}), "alt": (unit((5, 1)), {"50": 1.0})}


def test_second_sentence_reaches_a_chunk_the_first_cannot_and_a_chunk_in_several_lists_is_returned_once(client, make_collection, monkeypatch):
    make_collection(two_sentence_points())
    stub_by_sentence(monkeypatch, TWO_SENTENCES)

    alone = [h["id"] for h in qm.retrieve("q", mode="hybrid", top_k=3)]
    assert 2 not in alone  # diem 2 khong nam trong top 3 nhanh nao cua "q"

    both = qm.retrieve("q", mode="hybrid", top_k=3, alt_question="alt")
    ids = [h["id"] for h in both]
    # 1 chi den tu "q", 2 chi den tu "alt"; 3 nam trong 3 trong 4 danh sach nhung chi tra ve MOT lan (RRF gop trung theo id)
    assert sorted(ids) == [1, 2, 3]
    scores = [h["retrieval_score"] for h in both]
    assert scores == sorted(scores, reverse=True) and all(h["payload"]["source"] == "quy_che.pdf" for h in both)


def test_filter_applies_to_both_sentences_so_the_rewrite_cannot_bring_back_an_excluded_chunk(client, make_collection, monkeypatch):
    make_collection(two_sentence_points())
    stub_by_sentence(monkeypatch, TWO_SENTENCES)
    ids = [h["id"] for h in qm.retrieve("q", mode="hybrid", top_k=3, alt_question="alt", kho="A")]
    assert 2 not in ids and 3 in ids  # diem 2 (kho B) la ket qua tot nhat cua "alt" nhung bi filter kho chan
    assert len(ids) == 3  # van du top_k du diem tot nhat bi loai: loc TRUOC khi lay top-k, khong phai cat bot ket qua sau do
