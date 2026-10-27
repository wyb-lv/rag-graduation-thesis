from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import models

from app import config as cfg
from app.models import encoders
from app.rag import query_model as qm


class FakeQdrant:
    def __init__(self, points):
        self.points, self.calls = points, []

    def query_points(self, collection, **kwargs):
        self.calls.append((collection, kwargs))
        return SimpleNamespace(points=self.points)


def point(i, score=0.9, text=None):
    return SimpleNamespace(id=i, score=score, payload={"source": "quy_che.pdf", "text": text or f"doan {i}"})


@pytest.fixture
def stub(monkeypatch):
    """Thay encode, Qdrant va Groq bang ban gia; tra ve doi tuong de test chinh gia tri."""
    state = SimpleNamespace(
        lexical={"10": 0.5, "20": 0.25},
        qdrant=FakeQdrant([point(i, 1 - i / 100) for i in range(8)]),
        groq_calls=[],
    )
    monkeypatch.setattr(
        encoders,
        "encode_texts",
        lambda texts, batch_size=None: (np.ones((len(texts), 1024), "float32"), [state.lexical]),
    )
    monkeypatch.setattr(qm, "get_qdrant", lambda: state.qdrant)
    monkeypatch.setattr(
        qm.groq_client, "chat", lambda messages, **kw: state.groq_calls.append(messages) or "tra loi [quy_che.pdf]"
    )
    return state


def test_dense_mode_queries_named_dense_vector_only(stub):
    qm.retrieve("cau hoi", mode="dense")
    collection, kw = stub.qdrant.calls[0]
    assert collection == cfg.QDRANT_COLLECTION
    assert kw["using"] == "dense" and "prefetch" not in kw
    assert kw["limit"] == 20 and len(kw["query"]) == 1024


def test_hybrid_mode_prefetches_dense_and_sparse_with_rrf(stub):
    qm.retrieve("cau hoi", mode="hybrid")
    _, kw = stub.qdrant.calls[0]
    dense, sparse = kw["prefetch"]
    assert (dense.using, dense.limit) == ("dense", 20)
    assert (sparse.using, sparse.limit) == ("sparse", 20)
    assert sparse.query.indices == [10, 20] and sparse.query.values == [0.5, 0.25]
    assert kw["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    assert kw["limit"] == 20
    assert dense.filter is not None and sparse.filter is not None  # filter nam trong tung prefetch


def test_empty_sparse_query_falls_back_to_dense_prefetch_only(stub):
    stub.lexical = {}
    qm.retrieve("cau hoi", mode="hybrid")
    _, kw = stub.qdrant.calls[0]
    assert kw["using"] == "dense" and "prefetch" not in kw


def test_unknown_mode_is_rejected(stub):
    with pytest.raises(ValueError):
        qm.retrieve("cau hoi", mode="khong_co")
    assert stub.qdrant.calls == []


def test_mode_defaults_to_config(stub, monkeypatch):
    monkeypatch.setattr(cfg, "RAG_MODE", "dense")
    qm.retrieve("cau hoi")
    assert stub.qdrant.calls[0][1]["using"] == "dense"


def test_filter_has_kho_and_effective_date_range():
    flt = qm.build_filter(kho="Khoa CNTT", reference_date=date(2025, 6, 1))
    kho, tu, den = flt.must
    assert kho.key == "kho" and kho.match.value == "Khoa CNTT"
    ref = datetime(2025, 6, 1)
    assert tu.should[1].key == "hieu_luc_tu" and tu.should[1].range.lte == ref
    assert den.should[1].key == "hieu_luc_den" and den.should[1].range.gte == ref
    # moc de trong (thieu field) van hop le
    assert all(isinstance(c.should[0], models.IsEmptyCondition) for c in (tu, den))


def test_filter_without_kho_defaults_reference_date_to_today():
    flt = qm.build_filter()
    assert len(flt.must) == 2
    assert flt.must[0].should[1].range.lte.date() == date.today()


def test_filter_accepts_iso_string_date():
    assert qm.build_filter(reference_date="2025-06-01").must[0].should[1].range.lte == datetime(2025, 6, 1)


def test_rerank_returns_exactly_top_5_ordered_by_score(monkeypatch):
    hits = [{"id": i, "source": "s", "text": f"t{i}", "payload": {}, "retrieval_score": 0.0, "rerank_score": None}
            for i in range(20)]
    scores = [(i * 7) % 20 / 20 for i in range(20)]  # xao tron, khong trung
    monkeypatch.setattr(encoders, "rerank_scores", lambda q, passages: scores)
    top = qm.rerank("q", hits)
    assert len(top) == cfg.TOP_K_RERANK == 5
    got = [h["rerank_score"] for h in top]
    assert got == sorted(scores, reverse=True)[:5]
    assert [h["id"] for h in top] == sorted(range(20), key=lambda i: -scores[i])[:5]


def test_no_candidates_returns_no_result_without_calling_groq(stub):
    stub.qdrant.points = []
    answer, hits, in_scope = qm.ask("cau hoi", mode="hybrid_rerank")
    assert (answer, hits, in_scope) == (qm.NO_RESULT_ANSWER, [], False)
    assert stub.groq_calls == []


def test_ask_hybrid_rerank_cites_top5_chunks(stub, monkeypatch):
    monkeypatch.setattr(encoders, "rerank_scores", lambda q, passages: [0.9 - i / 10 for i in range(len(passages))])
    answer, hits, in_scope = qm.ask("cau hoi", mode="hybrid_rerank")
    assert in_scope and answer.startswith("tra loi")
    assert len(hits) == 5
    assert [h["rerank_score"] for h in hits] == sorted((h["rerank_score"] for h in hits), reverse=True)
    assert hits[0]["payload"] == {"source": "quy_che.pdf", "text": hits[0]["text"]}
    user_prompt = stub.groq_calls[0][1]["content"]
    assert hits[0]["text"] in user_prompt and "quy_che.pdf" in user_prompt


def test_rerank_score_below_floor_counts_as_no_result(stub, monkeypatch):
    monkeypatch.setattr(encoders, "rerank_scores", lambda q, passages: [cfg.RERANK_MIN_SCORE - 0.01] * len(passages))
    answer, hits, in_scope = qm.ask("cau hoi", mode="hybrid_rerank")
    assert (answer, hits, in_scope) == (qm.NO_RESULT_ANSWER, [], False)
    assert stub.groq_calls == []


def test_rerank_floor_keeps_only_chunks_at_or_above_min_score(stub, monkeypatch):
    # Thang cosine cua jina: doan lien quan ~0.3-0.7, doan khong lien quan quanh 0 hoac am.
    scores = [0.35, 0.12, 0.099, -0.04, -0.1, 0.02, 0.0, -0.2]
    monkeypatch.setattr(cfg, "RERANK_MIN_SCORE", 0.1)
    monkeypatch.setattr(encoders, "rerank_scores", lambda q, passages: scores[:len(passages)])
    _, hits, in_scope = qm.ask("cau hoi", mode="hybrid_rerank")
    assert in_scope and [h["rerank_score"] for h in hits] == [0.35, 0.12]


def test_dense_mode_applies_cosine_threshold(stub):
    stub.qdrant.points = [point(0, 0.3), point(1, 0.2)]
    assert qm.ask("cau hoi", mode="dense")[2] is False
    assert stub.groq_calls == []
    stub.qdrant.points = [point(0, 0.8), point(1, 0.2)]
    _, hits, in_scope = qm.ask("cau hoi", mode="dense")
    assert in_scope and [h["id"] for h in hits] == [0]


def test_hybrid_mode_skips_rerank_and_keeps_top_k(stub, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("hybrid khong duoc goi reranker")

    monkeypatch.setattr(encoders, "rerank_scores", boom)
    _, hits, _ = qm.ask("cau hoi", mode="hybrid", top_k=3)
    assert [h["id"] for h in hits] == [0, 1, 2]
    assert all(h["rerank_score"] is None for h in hits)


def test_prompt_demands_citations_and_admits_missing_answer():
    system = qm.build_messages("q", [{"source": "qd.pdf", "text": "noi dung"}])[0]["content"]
    assert "Điều" in system and "Khoản" in system
    assert "không chứa thông tin" in system
