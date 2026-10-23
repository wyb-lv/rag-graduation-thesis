import itertools
import logging
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from qdrant_client import models

from app import api
from app import config as cfg
from app import qdrant_setup
from app.models import encoders
from app.rag import query_model as qm

ALL_COMBOS = [c for n in (1, 2, 3) for c in itertools.combinations(cfg.HYBRID_LEG_NAMES, n)]


class FakeQdrant:
    def __init__(self):
        self.calls = []

    def query_points(self, collection, **kwargs):
        self.calls.append((collection, kwargs))
        return SimpleNamespace(points=[])


@pytest.fixture
def stub(monkeypatch):
    """Thay encode và Qdrant bằng bản giả; `encoded` đếm số lần gọi BGE-M3."""
    state = SimpleNamespace(lexical={"10": 0.5, "20": 0.25}, qdrant=FakeQdrant(), encoded=0)

    def encode(texts, batch_size=None):
        state.encoded += 1
        return np.ones((len(texts), 1024), "float32"), [state.lexical]

    monkeypatch.setattr(encoders, "encode_texts", encode)
    monkeypatch.setattr(qm, "get_qdrant", lambda: state.qdrant)
    return state


def test_default_legs_are_dense_and_sparse():
    assert cfg.HYBRID_LEGS == ("dense", "sparse")


def test_default_config_sends_the_same_hybrid_request_as_before_bm25(stub, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("dense", "sparse"))
    qm.retrieve("cau hoi", mode="hybrid", kho="K", reference_date=date(2025, 6, 1), khoa=2024)

    flt = qm.build_filter("K", date(2025, 6, 1), 2024)
    ef = models.SearchParams(hnsw_ef=cfg.HNSW_EF)
    [(collection, kwargs)] = stub.qdrant.calls
    assert collection == cfg.QDRANT_COLLECTION
    assert kwargs == dict(
        prefetch=[
            models.Prefetch(query=[1.0] * 1024, using="dense", filter=flt, limit=20, params=ef),
            models.Prefetch(
                query=models.SparseVector(indices=[10, 20], values=[0.5, 0.25]), using="sparse", filter=flt, limit=20
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=20,
        with_payload=True,
    )


def test_dense_mode_and_empty_sparse_send_the_same_dense_request_as_before(stub):
    stub.lexical = {}
    qm.retrieve("cau hoi", mode="hybrid")
    qm.retrieve("cau hoi", mode="dense")
    flt = qm.build_filter()
    expected = dict(
        query=[1.0] * 1024,
        using="dense",
        query_filter=flt,
        limit=20,
        with_payload=True,
        search_params=models.SearchParams(hnsw_ef=cfg.HNSW_EF),
    )
    assert [kw for _, kw in stub.qdrant.calls] == [expected, expected]


@pytest.mark.parametrize("legs", ALL_COMBOS, ids=lambda legs: "+".join(legs))
def test_the_same_filter_reaches_every_leg(stub, monkeypatch, legs):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", legs)
    qm.retrieve(
        "Điều 15 học bổng", mode="hybrid", kho="K", reference_date=date(2025, 6, 1), khoa=2024, loai_van_ban=["thong_bao"]
    )
    flt = qm.build_filter("K", date(2025, 6, 1), 2024, ["thong_bao"])
    [(_, kwargs)] = stub.qdrant.calls
    if len(legs) == 1:  # một nhánh: truy vấn thẳng, không có prefetch
        assert "prefetch" not in kwargs and kwargs["query_filter"] == flt
        assert kwargs["using"] == cfg.HYBRID_LEG_VECTORS[legs[0]]
        return
    prefetch = kwargs["prefetch"]
    assert [p.using for p in prefetch] == [cfg.HYBRID_LEG_VECTORS[leg] for leg in legs]
    assert all(p.filter is prefetch[0].filter for p in prefetch)  # cùng một đối tượng filter
    assert all(p.filter == flt for p in prefetch)
    assert all(p.limit == 20 for p in prefetch) and kwargs["limit"] == 20  # prefetch >= limit ngoài
    assert kwargs["query"] == models.FusionQuery(fusion=models.Fusion.RRF)


def test_bm25_leg_uses_bm25_query_vector(stub, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("dense", "sparse", "bm25"))
    qm.retrieve("Điều 15", mode="hybrid")
    bm25_prefetch = stub.qdrant.calls[0][1]["prefetch"][2]
    assert bm25_prefetch.using == "bm25" and bm25_prefetch.params is None
    assert len(bm25_prefetch.query.indices) == 3 and set(bm25_prefetch.query.values) == {1.0}


def test_leg_order_follows_config(stub, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("bm25", "dense"))
    qm.retrieve("Điều 15", mode="hybrid")
    assert [p.using for p in stub.qdrant.calls[0][1]["prefetch"]] == ["bm25", "dense"]


def test_bm25_leg_is_skipped_when_question_has_no_tokens(stub, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("dense", "sparse", "bm25"))
    qm.retrieve("?!...", mode="hybrid")
    assert [p.using for p in stub.qdrant.calls[0][1]["prefetch"]] == ["dense", "sparse"]


def test_every_leg_empty_falls_back_to_dense(stub, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("bm25",))
    qm.retrieve("?!...", mode="hybrid")
    kwargs = stub.qdrant.calls[0][1]
    assert kwargs["using"] == "dense" and "prefetch" not in kwargs and stub.encoded == 1


def test_bm25_only_does_not_run_the_embedder(stub, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("bm25",))
    qm.retrieve("học bổng", mode="hybrid_rerank")
    kwargs = stub.qdrant.calls[0][1]
    assert kwargs["using"] == "bm25" and "search_params" not in kwargs
    assert stub.encoded == 0  # không dùng GPU cho truy vấn chỉ-bm25


def test_dense_mode_ignores_hybrid_legs(stub, monkeypatch):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("bm25",))
    qm.retrieve("học bổng", mode="dense")
    assert stub.qdrant.calls[0][1]["using"] == "dense"


def test_filter_by_loai_van_ban():
    flt = qm.build_filter(loai_van_ban=["thong_bao", "quy_che"])
    cond = flt.must[-1]
    assert cond.key == "loai_van_ban" and cond.match == models.MatchAny(any=["thong_bao", "quy_che"])
    assert len(qm.build_filter().must) == 2  # mặc định không đổi
    assert len(qm.build_filter(loai_van_ban=None).must) == 2
    assert len(qm.build_filter(loai_van_ban=[]).must) == 2


def test_find_hits_and_ask_pass_loai_van_ban_down(stub, monkeypatch):
    seen = []
    monkeypatch.setattr(qm, "retrieve", lambda *a, **kw: seen.append(kw) or [])
    qm.find_hits("cau hoi", loai_van_ban=["thong_bao"])
    qm.ask("cau hoi", loai_van_ban=["bieu_mau"])
    assert [kw["loai_van_ban"] for kw in seen] == [["thong_bao"], ["bieu_mau"]]


# ── kiểm tra nhánh lúc khởi động ─────────────────────────────────────────────

def fake_client(vectors=("dense",), sparse=("sparse",), exists=True):
    params = SimpleNamespace(
        vectors={name: object() for name in vectors}, sparse_vectors={name: object() for name in sparse}
    )
    return SimpleNamespace(
        collection_exists=lambda name: exists,
        get_collection=lambda name: SimpleNamespace(config=SimpleNamespace(params=params)),
    )


def test_missing_leg_raises_a_clear_error():
    with pytest.raises(RuntimeError) as err:
        qdrant_setup.validate_legs(fake_client(), "quy_dinh_hybrid", ("dense", "sparse", "bm25"))
    message = str(err.value)
    assert "bm25" in message and "quy_dinh_hybrid" in message and "migrate_add_bm25" in message
    assert "dense" not in message.split("thiếu vector cho nhánh")[1].split("(")[0]  # chỉ nêu nhánh thật sự thiếu


def test_every_missing_leg_is_named():
    with pytest.raises(RuntimeError) as err:
        qdrant_setup.validate_legs(fake_client(vectors=("dense",), sparse=()), "c", ("dense", "sparse", "bm25"))
    assert "sparse, bm25" in str(err.value)


def test_present_legs_pass_and_are_logged(caplog):
    client = fake_client(sparse=("sparse", "bm25"))
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        qdrant_setup.validate_legs(client, "c", ("dense", "sparse", "bm25"))
    assert "dense, sparse, bm25" in caplog.text


def test_unnamed_dense_vector_does_not_crash_validation():
    client = fake_client()
    client.get_collection = lambda name: SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=object(), sparse_vectors=None))
    )
    with pytest.raises(RuntimeError):
        qdrant_setup.validate_legs(client, "c", ("dense",))


def test_missing_collection_or_qdrant_down_does_not_block_startup(caplog):
    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        qdrant_setup.validate_legs(fake_client(exists=False), "chua_co", ("dense", "bm25"))
    assert "chưa tồn tại" in caplog.text

    def down(name):
        raise ConnectionError("qdrant down")

    client = fake_client()
    client.collection_exists = down
    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        qdrant_setup.validate_legs(client, "c", ("dense", "bm25"))
    assert "qdrant down" in caplog.text


def test_server_does_not_start_when_a_configured_leg_is_missing(monkeypatch):
    monkeypatch.setattr(cfg, "WARMUP", False)
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("dense", "sparse", "bm25"))
    monkeypatch.setattr(qm, "get_qdrant", lambda: fake_client())
    with pytest.raises(RuntimeError, match="bm25"):
        with TestClient(api.app):
            pass
