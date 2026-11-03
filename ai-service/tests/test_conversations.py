"""Luu tru hoi thoai trong MongoDB. Nhom dau: api.py voi kho gia trong RAM (khong cham mang). Nhom cuoi: conversations.py voi MongoDB that
(marker `integration`; TEST_MONGODB_URI, mac dinh mongodb://localhost:27017; khong ket noi duoc thi bo qua)."""
import asyncio
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import api, conversations, session
from app import config as cfg
from app.rag import query_model as qm
from app.router.router_llm import Action
from app.router.small_talk import Intent

SID, SID2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
HIT = {"source": "qd1.pdf", "text": "t", "rerank_score": 0.9, "retrieval_score": 0.9, "payload": {}}
ENDPOINTS = pytest.mark.parametrize("endpoint", ["/ask", "/ask/stream"])
U1, U2 = {"X-User-Id": "u1"}, {"X-User-Id": "u2"}


@pytest.fixture
def archive(monkeypatch):
    """Kho gia co cung hinh dang tra ve voi conversations.py; `db.fail` mo phong MongoDB chet. Pipeline RAG gia: tra loi "ANS-1"."""
    monkeypatch.setattr(cfg, "CONVERSATIONS_ENABLED", True)
    monkeypatch.setattr(cfg, "SESSION_ENABLED", True)
    monkeypatch.setattr(session, "_store", None)
    db = SimpleNamespace(docs={}, fail=False)

    def check():
        if db.fail:
            raise ConnectionError("mongodb://user:SECRET@host:27017 unreachable")

    async def add_turn(sid, user_id, question, answer, intent):
        check()
        now = datetime.now(timezone.utc)
        doc = db.docs.setdefault(session.key_for(sid, user_id), {
            "session_id": sid, "user_id": user_id, "title": question[:cfg.CONVERSATION_TITLE_CHARS], "created_at": now, "messages": []})
        doc["updated_at"] = now
        doc["messages"] += [{"role": "user", "content": question, "ts": now},
                            {"role": "assistant", "content": answer, "intent": intent, "ts": now}]

    async def list_for(user_id, limit):
        check()
        mine = sorted((d for d in db.docs.values() if d["user_id"] == user_id), key=lambda d: d["updated_at"], reverse=True)
        return [{k: v for k, v in d.items() if k not in ("messages", "user_id")} for d in mine[:limit]]

    async def get(sid, user_id):
        check()
        return db.docs.get(session.key_for(sid, user_id))

    async def delete(sid, user_id):
        check()
        return db.docs.pop(session.key_for(sid, user_id), None) is not None

    for fn in (add_turn, list_for, get, delete):
        monkeypatch.setattr(conversations, fn.__name__, fn)
    monkeypatch.setattr(qm, "ask", lambda q, **kw: ("ANS-1", [], True))
    monkeypatch.setattr(qm, "find_hits", lambda q, **kw: [HIT])
    monkeypatch.setattr(qm, "generate_answer_stream", lambda q, hits, **kw: iter(["ANS", "-1"]))
    return db


@pytest.fixture
def client():
    return TestClient(api.app)


def ask(client, endpoint, question, sid=SID, headers=U1):
    resp = client.post(endpoint, json={"question": question, "session_id": sid}, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp


# ---- Ghi ----

@ENDPOINTS
def test_a_completed_turn_is_archived_in_order_under_one_conversation(client, archive, endpoint):
    ask(client, endpoint, "cau hoi mot")
    ask(client, endpoint, "cau hoi hai")
    (doc,) = archive.docs.values()
    assert doc["session_id"] == SID and doc["user_id"] == "u1" and doc["title"] == "cau hoi mot"
    assert [(m["role"], m["content"]) for m in doc["messages"]] == [
        ("user", "cau hoi mot"), ("assistant", "ANS-1"), ("user", "cau hoi hai"), ("assistant", "ANS-1")]
    assert [m.get("intent") for m in doc["messages"]] == [None, "rag", None, "rag"]


@ENDPOINTS
def test_a_turn_kept_out_of_the_prompt_history_is_still_archived(client, archive, monkeypatch, endpoint):
    async def plan(t):
        return Action(Intent.OFF_TOPIC, answer=cfg.OFF_TOPIC_ANSWER, in_scope=False, store=False)

    monkeypatch.setattr(api, "_plan", plan)
    ask(client, endpoint, "viet giup bai tho")
    (doc,) = archive.docs.values()
    assert [(m["content"], m.get("intent")) for m in doc["messages"]] == [("viet giup bai tho", None), (cfg.OFF_TOPIC_ANSWER, "off_topic")]
    assert asyncio.run(_history(SID, "u1")) == []  # nhung khong vao lich su prompt


def test_a_stream_that_ends_in_an_error_archives_nothing(client, archive, monkeypatch):
    def broken(q, hits, **kw):
        yield "ANS"
        raise RuntimeError("mat ket noi")

    monkeypatch.setattr(qm, "generate_answer_stream", broken)
    resp = client.post("/ask/stream", json={"question": "q", "session_id": SID}, headers=U1)
    assert json.loads(resp.text.split("\n\n")[-2][len("data: "):])["event"] == "error"
    assert archive.docs == {}


@ENDPOINTS
def test_a_mongodb_outage_does_not_break_the_answer_and_only_the_error_class_is_logged(client, archive, caplog, endpoint):
    archive.fail = True
    resp = ask(client, endpoint, "CAU-HOI-RIENG")
    assert "ANS" in resp.text
    assert "ConnectionError" in caplog.text and "SECRET" not in caplog.text and "CAU-HOI-RIENG" not in caplog.text


@ENDPOINTS
def test_nothing_is_archived_without_mongodb_uri_or_without_a_session(client, archive, monkeypatch, endpoint):
    monkeypatch.setattr(cfg, "CONVERSATIONS_ENABLED", False)
    ask(client, endpoint, "q")
    monkeypatch.setattr(cfg, "CONVERSATIONS_ENABLED", True)
    monkeypatch.setattr(cfg, "SESSION_ENABLED", False)  # khong co session_id thi khong co cuoc hoi thoai
    assert client.post(endpoint, json={"question": "q"}).status_code == 200
    assert archive.docs == {}


# ---- Doc / xoa ----

def test_list_needs_the_gateway_user_header_and_returns_only_that_users_conversations_newest_first(client, archive):
    ask(client, "/ask", "cu", sid=SID)
    ask(client, "/ask", "moi", sid=SID2)
    ask(client, "/ask", "cua nguoi khac", sid=SID, headers=U2)
    assert client.get("/conversations").status_code == 401
    body = client.get("/conversations", headers=U1).json()
    assert [c["title"] for c in body] == ["moi", "cu"] and [c["session_id"] for c in body] == [SID2, SID]
    assert all(set(c) == {"session_id", "title", "created_at", "updated_at"} for c in body)  # khong kem tin nhan
    assert len(client.get("/conversations?limit=1", headers=U1).json()) == 1
    for bad in (0, cfg.CONVERSATIONS_LIST_MAX + 1):
        assert client.get(f"/conversations?limit={bad}", headers=U1).status_code == 422


def test_get_returns_the_messages_and_never_another_users_conversation(client, archive):
    ask(client, "/ask", "BI-MAT-CUA-U1")
    body = client.get(f"/conversations/{SID}", headers=U1).json()
    assert [(m["role"], m["content"]) for m in body["messages"]] == [("user", "BI-MAT-CUA-U1"), ("assistant", "ANS-1")]
    assert body["messages"][1]["intent"] == "rag" and "user_id" not in body
    assert client.get(f"/conversations/{SID}", headers=U2).status_code == 404
    assert client.get(f"/conversations/{SID}").status_code == 404  # khong header = khong phai cua u1
    assert client.get(f"/conversations/{SID2}", headers=U1).status_code == 404


@pytest.mark.parametrize("method", ["get", "delete"])
def test_a_non_uuid_session_id_is_a_400(client, archive, method):
    assert getattr(client, method)("/conversations/abc", headers=U1).status_code == 400


def test_delete_removes_the_archive_and_the_short_term_history_and_is_per_user(client, archive):
    ask(client, "/ask", "cau hoi")
    assert asyncio.run(_history(SID, "u1")) != []
    assert client.delete(f"/conversations/{SID}", headers=U2).status_code == 404
    assert client.delete(f"/conversations/{SID}", headers=U1).status_code == 200
    assert asyncio.run(_history(SID, "u1")) == []
    assert client.get(f"/conversations/{SID}", headers=U1).status_code == 404
    assert client.delete(f"/conversations/{SID}", headers=U1).status_code == 404


def test_endpoints_are_503_when_mongodb_is_not_configured(client, archive, monkeypatch):
    monkeypatch.setattr(cfg, "CONVERSATIONS_ENABLED", False)
    for call in (client.get("/conversations", headers=U1), client.get(f"/conversations/{SID}", headers=U1),
                 client.delete(f"/conversations/{SID}", headers=U1)):
        assert call.status_code == 503


@pytest.mark.parametrize("call", ["list", "get", "delete"])
def test_a_mongodb_outage_is_a_502_without_leaking_the_error(client, archive, caplog, call):
    archive.fail = True
    resp = {"list": lambda: client.get("/conversations", headers=U1),
            "get": lambda: client.get(f"/conversations/{SID}", headers=U1),
            "delete": lambda: client.delete(f"/conversations/{SID}", headers=U1)}[call]()
    assert resp.status_code == 502 and "SECRET" not in resp.text + caplog.text


async def _history(sid, user_id):
    s = await session.get_store().open(session.key_for(sid, user_id))
    try:
        return s.history
    finally:
        s.release()


# ---- MongoDB that ----

@pytest.fixture
def mongo(monkeypatch):
    monkeypatch.setattr(cfg, "MONGODB_URI", os.environ.get("TEST_MONGODB_URI", "mongodb://localhost:27017"))
    monkeypatch.setattr(cfg, "MONGODB_DB", "hsu_chatbot_test")
    monkeypatch.setattr(cfg, "MONGODB_TIMEOUT_MS", 1000)
    monkeypatch.setattr(conversations, "_collection", None)  # AsyncMongoClient gan voi event loop: moi test mot client
    return conversations


def mongo_test(fn):
    """Chay `fn(conversations)` trong mot event loop, don collection truoc / sau; MongoDB khong len thi bo qua."""
    async def run(mod):
        from pymongo.errors import ServerSelectionTimeoutError
        try:
            await mod.collection().delete_many({})
        except ServerSelectionTimeoutError:
            pytest.skip("khong ket noi duoc MongoDB (dat TEST_MONGODB_URI)")
        try:
            await fn(mod)
        finally:
            await mod.collection().drop()

    def wrapper(mongo):
        asyncio.run(run(mongo))
    wrapper.__name__ = fn.__name__
    return pytest.mark.integration(wrapper)


@mongo_test
async def test_mongo_turns_append_to_one_document_with_a_stable_title(c):
    await c.ensure_indexes()
    await c.ensure_indexes()  # idempotent
    await c.add_turn(SID, "u1", "cau hoi mot", "tra loi mot", "rag")
    await c.add_turn(SID, "u1", "cau hoi hai", "tra loi hai", "clarify")
    assert await c.collection().count_documents({}) == 1
    doc = await c.get(SID, "u1")
    assert doc["_id"] == f"u1:{SID}" and doc["title"] == "cau hoi mot" and doc["session_id"] == SID
    assert [m["content"] for m in doc["messages"]] == ["cau hoi mot", "tra loi mot", "cau hoi hai", "tra loi hai"]
    assert doc["messages"][3]["intent"] == "clarify" and "intent" not in doc["messages"][0]
    assert doc["created_at"] < doc["updated_at"] and doc["created_at"].tzinfo is not None
    assert "user_id" not in doc


@mongo_test
async def test_mongo_a_long_first_question_is_cut_for_the_title_but_stored_whole(c):
    await c.add_turn(SID, "u1", "x" * 500, "a", "rag")
    doc = await c.get(SID, "u1")
    assert len(doc["title"]) == cfg.CONVERSATION_TITLE_CHARS and len(doc["messages"][0]["content"]) == 500


@mongo_test
async def test_mongo_list_is_per_user_newest_first_without_messages_and_respects_the_limit(c):
    await c.add_turn(SID, "u1", "cu", "a", "rag")
    await c.add_turn(SID2, "u1", "moi", "a", "rag")
    await c.add_turn(SID, "u2", "cua u2", "a", "rag")
    await c.add_turn(SID, "u1", "cu tiep", "a", "rag")  # SID vua duoc cap nhat nen len dau
    rows = await c.list_for("u1", 10)
    assert [r["session_id"] for r in rows] == [SID, SID2] and all("messages" not in r and "user_id" not in r for r in rows)
    assert len(await c.list_for("u1", 1)) == 1
    assert [r["title"] for r in await c.list_for("u2", 10)] == ["cua u2"]
    assert await c.list_for("u3", 10) == []


@mongo_test
async def test_mongo_a_conversation_is_only_reachable_with_the_same_user_id(c):
    await c.add_turn(SID, "u1", "q", "a", "rag")
    assert await c.get(SID, "u2") is None and await c.get(SID, None) is None
    assert await c.delete(SID, "u2") is False and await c.get(SID, "u1") is not None
    assert await c.delete(SID, "u1") is True and await c.delete(SID, "u1") is False


@mongo_test
async def test_mongo_anonymous_conversations_are_keyed_by_session_id_and_not_listed(c):
    await c.add_turn(SID, None, "q", "a", "rag")
    assert (await c.get(SID, None))["_id"] == SID
    assert await c.list_for("u1", 10) == []


@mongo_test
async def test_mongo_concurrent_turns_on_one_conversation_lose_no_message(c):
    await asyncio.gather(*(c.add_turn(SID, "u1", f"q{i}", f"a{i}", "rag") for i in range(20)))
    doc = await c.get(SID, "u1")
    assert len(doc["messages"]) == 40
    assert {m["content"] for m in doc["messages"] if m["role"] == "user"} == {f"q{i}" for i in range(20)}
