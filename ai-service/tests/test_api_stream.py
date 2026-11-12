import json
import time

import pytest
from fastapi.testclient import TestClient

from app import api
from app.llm import groq_client
from app.rag import query_model as qm

HIT = {
    "source": "quy_che.pdf",
    "text": "Thi sinh duoc dang ky toi da 15 nguyen vong",
    "payload": {"source": "quy_che.pdf"},
    "retrieval_score": 0.03,
    "rerank_score": 0.6,
}


def events(resp):
    """Tach body SSE thanh danh sach payload `data:` (da parse JSON, tru [DONE])."""
    out = []
    for block in resp.text.split("\n\n"):
        if block.startswith("data: "):
            raw = block[len("data: "):]
            out.append(raw if raw == "[DONE]" else json.loads(raw))
    return out


@pytest.fixture
def client():
    return TestClient(api.app)


def test_stream_sends_sources_then_deltas_then_done(client, monkeypatch):
    monkeypatch.setattr(qm, "find_hits", lambda question, **kw: [HIT])
    monkeypatch.setattr(qm, "generate_answer_stream", lambda question, hits: iter(["Toi da ", "15\nnguyen vong"]))
    resp = client.post("/ask/stream", json={"question": "Bao nhieu nguyen vong?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    sources, *deltas, done = events(resp)
    assert sources["event"] == "sources" and sources["in_scope"] is True
    assert sources["sources"][0]["score"] == 0.6 and sources["sources"][0]["source"] == "quy_che.pdf"
    assert [d["text"] for d in deltas] == ["Toi da ", "15\nnguyen vong"]
    assert done == "[DONE]"


def test_stream_out_of_scope_skips_llm(client, monkeypatch):
    monkeypatch.setattr(qm, "find_hits", lambda question, **kw: [])
    monkeypatch.setattr(qm, "generate_answer_stream", lambda *a: pytest.fail("khong duoc goi LLM"))
    sources, delta, done = events(client.post("/ask/stream", json={"question": "Thoi tiet?"}))
    assert sources == {"event": "sources", "in_scope": False, "sources": [], "session_id": sources["session_id"]}
    assert delta == {"event": "delta", "text": qm.NO_RESULT_ANSWER}
    assert done == "[DONE]"


def test_stream_error_mid_answer_sends_error_event_without_done(client, monkeypatch):
    def broken(question, hits):
        yield "Toi da "
        raise groq_client.GroqError("mat ket noi")

    monkeypatch.setattr(qm, "find_hits", lambda question, **kw: [HIT])
    monkeypatch.setattr(qm, "generate_answer_stream", broken)
    *_, last = events(client.post("/ask/stream", json={"question": "Bao nhieu nguyen vong?"}))
    assert last["event"] == "error" and "mat ket noi" in last["detail"]


def test_stream_empty_question_is_400(client):
    assert client.post("/ask/stream", json={"question": "  "}).status_code == 400


def test_stream_retrieval_failure_is_502_before_streaming(client, monkeypatch):
    def down(question, **kw):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(qm, "find_hits", down)
    resp = client.post("/ask/stream", json={"question": "cau hoi"})
    assert resp.status_code == 502 and "qdrant down" in resp.json()["detail"]


def test_startup_warms_up_embedder_and_reranker(monkeypatch):
    calls = []
    monkeypatch.setattr(api.qdrant_setup, "validate_legs", lambda client: None)  # lifespan không chạm Qdrant thật
    monkeypatch.setattr(api.encoders, "encode_texts", lambda texts: calls.append("encode"))
    monkeypatch.setattr(api.encoders, "rerank_scores", lambda q, passages: calls.append("rerank"))
    monkeypatch.setitem(api._warmup, "status", "pending")
    with TestClient(api.app) as client:  # `with` -> chay lifespan
        for _ in range(100):
            if client.get("/health").json()["models"] != "pending":
                break
            time.sleep(0.01)
        assert client.get("/health").json() == {"status": "ok", "models": "ready"}
    assert calls == ["encode", "rerank"]


def test_warmup_failure_is_reported_not_raised(monkeypatch):
    def oom(texts):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(api.encoders, "encode_texts", oom)
    monkeypatch.setitem(api._warmup, "status", "pending")
    api.warm_up()
    assert api._warmup["status"] == "error"


def test_stream_missing_env_is_500(client, monkeypatch):
    def missing(question, hits):
        raise KeyError("GROQ_QWEN_MODEL")

    monkeypatch.setattr(qm, "find_hits", lambda question, **kw: [HIT])
    monkeypatch.setattr(qm, "generate_answer_stream", missing)
    assert client.post("/ask/stream", json={"question": "cau hoi"}).status_code == 500


def test_loai_van_ban_is_passed_to_the_pipeline(client, monkeypatch):
    seen = []
    monkeypatch.setattr(qm, "find_hits", lambda question, **kw: seen.append(kw) or [])
    monkeypatch.setattr(qm, "ask", lambda question, **kw: seen.append(kw) or ("tra loi", [], False))
    client.post("/ask/stream", json={"question": "cau hoi", "loai_van_ban": ["thong_bao", "quy_che"]})
    client.post("/ask", json={"question": "cau hoi", "loai_van_ban": ["bieu_mau"]})
    assert [kw["loai_van_ban"] for kw in seen] == [["thong_bao", "quy_che"], ["bieu_mau"]]


def test_loai_van_ban_defaults_to_none(client, monkeypatch):
    seen = []
    monkeypatch.setattr(qm, "find_hits", lambda question, **kw: seen.append(kw) or [])
    client.post("/ask/stream", json={"question": "cau hoi"})
    assert seen[0]["loai_van_ban"] is None


def test_loai_van_ban_must_be_a_list_of_strings(client):
    assert client.post("/ask", json={"question": "cau hoi", "loai_van_ban": "thong_bao"}).status_code == 422
