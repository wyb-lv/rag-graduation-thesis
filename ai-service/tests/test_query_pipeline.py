from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from qdrant_client import models

from app import bm25
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
        qdrant=FakeQdrant([point(i, 1 - i / 100) for i in range(8)]),
        groq_calls=[],
    )
    monkeypatch.setattr(
        encoders,
        "encode_texts",
        lambda texts, batch_size=None: np.ones((len(texts), 1024), "float32"),
    )
    monkeypatch.setattr(qm, "get_qdrant", lambda: state.qdrant)
    monkeypatch.setattr(
        qm.groq_client, "chat", lambda messages, **kw: state.groq_calls.append(messages) or "tra loi [quy_che.pdf]"
    )
    return state


def test_retrieve_prefetches_dense_and_bm25_with_rrf(stub):
    qm.retrieve("cau hoi")
    collection, kw = stub.qdrant.calls[0]
    assert collection == cfg.QDRANT_COLLECTION
    dense, lexical = kw["prefetch"]
    assert (dense.using, dense.limit) == ("dense", 20)
    assert (lexical.using, lexical.limit) == ("bm25", 20)
    assert lexical.query == bm25.bm25_query_vector("cau hoi")  # token am tiet + bigram, IDF do Qdrant nhan
    assert kw["query"] == models.FusionQuery(fusion=models.Fusion.RRF)
    assert kw["limit"] == 20
    assert dense.filter is not None and lexical.filter is not None  # filter nam trong tung prefetch


def test_empty_bm25_query_falls_back_to_dense_prefetch_only(stub):
    qm.retrieve("?!...")  # cau hoi toan dau cau: khong co token bm25
    _, kw = stub.qdrant.calls[0]
    assert kw["using"] == "dense" and "prefetch" not in kw


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
    answer, hits, in_scope = qm.ask("cau hoi")
    assert (answer, hits, in_scope) == (qm.NO_RESULT_ANSWER, [], False)
    assert stub.groq_calls == []


def test_ask_cites_the_top5_reranked_chunks(stub, monkeypatch):
    monkeypatch.setattr(encoders, "rerank_scores", lambda q, passages: [0.9 - i / 10 for i in range(len(passages))])
    answer, hits, in_scope = qm.ask("cau hoi")
    assert in_scope and answer.startswith("tra loi")
    assert len(hits) == 5
    assert [h["rerank_score"] for h in hits] == sorted((h["rerank_score"] for h in hits), reverse=True)
    assert hits[0]["payload"] == {"source": "quy_che.pdf", "text": hits[0]["text"]}
    user_prompt = stub.groq_calls[0][1]["content"]
    assert hits[0]["text"] in user_prompt and "quy_che.pdf" in user_prompt


def test_rerank_score_below_floor_counts_as_no_result(stub, monkeypatch):
    monkeypatch.setattr(encoders, "rerank_scores", lambda q, passages: [cfg.RERANK_MIN_SCORE - 0.01] * len(passages))
    answer, hits, in_scope = qm.ask("cau hoi")
    assert (answer, hits, in_scope) == (qm.NO_RESULT_ANSWER, [], False)
    assert stub.groq_calls == []


def test_rerank_floor_keeps_only_chunks_at_or_above_min_score(stub, monkeypatch):
    # Thang cosine cua jina: doan lien quan ~0.3-0.7, doan khong lien quan quanh 0 hoac am.
    scores = [0.35, 0.12, 0.099, -0.04, -0.1, 0.02, 0.0, -0.2]
    monkeypatch.setattr(cfg, "RERANK_MIN_SCORE", 0.1)
    monkeypatch.setattr(encoders, "rerank_scores", lambda q, passages: scores[:len(passages)])
    _, hits, in_scope = qm.ask("cau hoi")
    assert in_scope and [h["rerank_score"] for h in hits] == [0.35, 0.12]


def test_prompt_demands_citations_and_admits_missing_answer():
    system = qm.build_messages("q", [{"source": "qd.pdf", "text": "noi dung"}])[0]["content"]
    assert "Điều" in system and "Khoản" in system
    assert "không chứa thông tin" in system
