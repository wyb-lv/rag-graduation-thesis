import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from app import config as cfg
from app import session
from app.session import SessionStore

SID_A, SID_B = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"


def run(coro):
    return asyncio.run(coro)


async def turn(store, key, question, answer, intent="rag", facts=None):
    s = await store.open(key)
    await s.add_turn(question, answer, intent, facts or {})
    s.release()


# ---- id va khoa ----

@pytest.mark.parametrize("raw", [
    SID_A, SID_A.upper(), "6f9619ff-8b86-4011-b42d-00c04fc964ff",
])
def test_parse_id_accepts_canonical_uuids_in_lower_case(raw):
    assert session.parse_id(raw) == raw.lower()


@pytest.mark.parametrize("raw", [
    "", "abc", "../../etc/passwd", SID_A.replace("-", ""), "{" + SID_A + "}", "urn:uuid:" + SID_A, " " + SID_A,
    SID_A + "\n", SID_A + "x", "1' OR '1'='1", None, 123,
])
def test_parse_id_rejects_everything_else(raw):
    assert session.parse_id(raw) is None


def test_key_is_namespaced_by_the_trusted_user_id_when_the_gateway_sends_one():
    assert session.key_for(SID_A) == SID_A
    assert session.key_for(SID_A, "42") == f"42:{SID_A}"
    assert session.key_for(SID_A, "42") != session.key_for(SID_A, "43")


def test_digest_is_short_and_does_not_contain_the_key():
    assert len(session.digest(SID_A)) == 8 and session.digest(SID_A) not in SID_A


# ---- luu va doc ----

def test_two_sessions_never_see_each_other():
    async def go():
        store = SessionStore()
        await turn(store, "a", "hoi A", "tra loi A")
        await turn(store, "b", "hoi B", "tra loi B")
        a, b = await store.open("a"), await store.open("b")
        return a.history, b.history

    a, b = run(go())
    assert a == [("user", "hoi A"), ("assistant", "tra loi A")]
    assert b == [("user", "hoi B"), ("assistant", "tra loi B")]


def test_unknown_session_starts_empty():
    s = run(SessionStore().open("chua-tung-thay"))
    assert s.history == [] and s.facts == {} and s.previous_was_clarify is False


def test_reset_clears_only_that_session():
    async def go():
        store = SessionStore()
        await turn(store, "a", "q", "a")
        await turn(store, "b", "q", "a")
        s = await store.open("a", reset=True)
        h_reset = s.history
        s.release()
        return h_reset, (await store.open("a")).history, (await store.open("b")).history

    after_reset, a, b = run(go())
    assert after_reset == [] and a == [] and len(b) == 2


def test_history_keeps_only_role_and_text_not_llamaindex_metadata():
    async def go():
        store = SessionStore()
        s = await store.open("k")
        Message = store.message_class
        # LlamaIndex / ban cu co the nhet session_id vao metadata cua tin nhan; no khong duoc vao prompt
        await s._entry.memory.aput_messages([
            Message(role="user", content="cau hoi", additional_kwargs={"session_id": SID_A, "intent": "rag"}),
            Message(role="assistant", content="tra loi", additional_kwargs={"session_id": SID_A}),
        ])
        s.release()
        return (await store.open("k")).history

    history = run(go())
    assert history == [("user", "cau hoi"), ("assistant", "tra loi")]
    assert SID_A not in repr(history)


def test_window_is_capped_by_turns(monkeypatch):
    monkeypatch.setattr(cfg, "SESSION_MAX_TURNS", 2)
    monkeypatch.setattr(cfg, "SESSION_MAX_HISTORY_TOKENS", 100_000)

    async def go():
        store = SessionStore()
        for i in range(5):
            await turn(store, "k", f"q{i}", f"a{i}")
        return (await store.open("k")).history

    assert [t for _, t in run(go())] == ["q3", "a3", "q4", "a4"]


def test_window_is_capped_by_tokens_whichever_is_smaller(monkeypatch):
    monkeypatch.setattr(cfg, "SESSION_MAX_TURNS", 6)
    monkeypatch.setattr(cfg, "SESSION_MAX_HISTORY_TOKENS", 120)  # ~240 ky tu (2 ky tu / token)

    async def go():
        store = SessionStore()
        for i in range(4):
            await turn(store, "k", f"q{i} " + "x" * 80, f"a{i} " + "y" * 80)
        return (await store.open("k")).history

    history = run(go())
    assert 2 <= len(history) < 8
    assert sum(len(t) for _, t in history) / cfg.SESSION_CHARS_PER_TOKEN <= 120
    assert history[-1][1].startswith("a3")  # luot moi nhat con, luot cu bi bo


def test_each_history_message_is_truncated_so_one_long_answer_cannot_fill_the_budget(monkeypatch):
    monkeypatch.setattr(cfg, "SESSION_MESSAGE_MAX_CHARS", 50)

    async def go():
        store = SessionStore()
        await turn(store, "k", "q", "A" * 5000)
        return (await store.open("k")).history

    (_, q), (_, a) = run(go())
    assert q == "q" and len(a) <= 51 and a.endswith("…")


# ---- khoa / kho va intent di qua mot vong luu - doc ----

def test_facts_and_intent_survive_a_round_trip_in_memory_mode():
    async def go():
        store = SessionStore()
        await turn(store, "k", "q1", "Ban muon hoi gi?", intent="clarify", facts={"khoa": 2024, "kho": "tuyen_sinh"})
        return await store.open("k")

    s = run(go())
    assert s.facts == {"khoa": 2024, "kho": "tuyen_sinh"}
    assert s.previous_was_clarify is True


def test_previous_was_clarify_is_true_only_when_the_last_stored_answer_was_a_clarify():
    async def go():
        store = SessionStore()
        out = []
        await turn(store, "k", "dieu kien la gi?", "hoi lai", intent="clarify")
        s = await store.open("k"); out.append(s.previous_was_clarify); s.release()
        await turn(store, "k", "hoc bong", "tra loi", intent="rag")
        s = await store.open("k"); out.append(s.previous_was_clarify); s.release()
        await turn(store, "k", "cam on", "khong co gi", intent="thanks")
        s = await store.open("k"); out.append(s.previous_was_clarify)
        return out

    assert run(go()) == [True, False, False]


def test_intent_and_facts_survive_a_new_store_on_the_same_sqlite_file(tmp_path):
    uri = f"sqlite+aiosqlite:///{(tmp_path / 's.db').as_posix()}"

    async def write():
        await turn(SessionStore(uri), "k", "q", "Ban hoi khoa nao?", intent="clarify", facts={"khoa": 2023})

    async def read():
        return await SessionStore(uri).open("k")  # store moi = nhu sau khi restart

    run(write())
    s = run(read())
    assert s.history == [("user", "q"), ("assistant", "Ban hoi khoa nao?")]
    assert s.facts == {"khoa": 2023} and s.previous_was_clarify is True


def test_a_new_llamaindex_memory_with_the_same_id_and_uri_sees_earlier_messages(tmp_path):
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.memory import Memory

    uri = f"sqlite+aiosqlite:///{(tmp_path / 'm.db').as_posix()}"
    tok = lambda text: range(len(text))

    async def go():
        first = Memory.from_defaults(session_id="S1", token_limit=800, tokenizer_fn=tok, async_database_uri=uri)
        await first.aput(ChatMessage(role="user", content="nho giup toi"))
        second = Memory.from_defaults(session_id="S1", token_limit=800, tokenizer_fn=tok, async_database_uri=uri)
        other = Memory.from_defaults(session_id="S2", token_limit=800, tokenizer_fn=tok, async_database_uri=uri)
        return [m.content for m in await second.aget()], await other.aget()

    seen, other = run(go())
    assert seen == ["nho giup toi"] and other == []


def test_sqlite_row_older_than_the_ttl_is_forgotten_when_its_session_is_opened_again(tmp_path, monkeypatch):
    uri = f"sqlite+aiosqlite:///{(tmp_path / 's.db').as_posix()}"
    run(turn(SessionStore(uri), "k", "q", "a"))
    monkeypatch.setattr(cfg, "SESSION_TTL_S", -1)  # moi dong deu "cu hon TTL"
    s = run(SessionStore(uri).open("k"))
    assert s.history == []


def test_sqlite_sweep_deletes_expired_sessions_nobody_comes_back_for(tmp_path, monkeypatch):
    uri = f"sqlite+aiosqlite:///{(tmp_path / 's.db').as_posix()}"
    monkeypatch.setattr(cfg, "SESSION_STORE", "sqlite")

    async def go():
        old = SessionStore(uri)
        await turn(old, "abandoned", "q", "a")
        monkeypatch.setattr(cfg, "SESSION_TTL_S", -1)
        fresh = SessionStore(uri)
        await fresh.open("someone-else")  # lan open dau tien kich hoat sweep
        return await fresh._chat_store.get_messages("abandoned", status=None)

    assert run(go()) == []


def test_two_first_requests_get_the_same_store(monkeypatch):
    monkeypatch.setattr(session, "_store", None)

    async def go():
        return await asyncio.gather(session.store(), session.store(), session.store())

    a, b, c = run(go())
    assert a is b is c


@pytest.mark.parametrize("env, ok, needle", [
    ({"SESSION_STORE": "sqlite", "SESSION_DB_URI": "sqlite+aiosqlite:////data/s.db"}, True, ""),
    ({"SESSION_STORE": "sqlite"}, False, "SESSION_DB_URI"),
    ({"SESSION_STORE": "mongodb"}, False, "sqlite"),
])
def test_config_validates_the_session_store(env, ok, needle):
    root = Path(__file__).resolve().parent.parent
    base = {k: v for k, v in __import__("os").environ.items() if not k.startswith(("SESSION_", "ROUTER_"))}
    # load_dotenv(override=True) cua config.py se de `.env` cua may dev len env cua test, nen bo no di trong tien trinh con
    done = subprocess.run([sys.executable, "-I", "-c", "import sys, dotenv; sys.path.insert(0, '.'); dotenv.load_dotenv = lambda *a, **k: False; import app.config"],
                          cwd=root, env={**base, **env}, capture_output=True, text=True)
    assert (done.returncode == 0) is ok, done.stderr
    assert needle in done.stderr


# ---- TTL, LRU, lock ----

def test_idle_session_is_evicted_after_the_ttl_and_its_history_is_gone(monkeypatch):
    async def go():
        store = SessionStore()
        await turn(store, "old", "q", "a")
        store._entries["old"].used -= cfg.SESSION_TTL_S + 1  # coi nhu da nam im qua TTL
        s = await store.open("other")
        s.release()
        return "old" in store._entries, (await store.open("old")).history

    still_there, history = run(go())
    assert still_there is False and history == []


def test_lru_evicts_the_least_recently_used_session_over_the_cap(monkeypatch):
    monkeypatch.setattr(cfg, "SESSION_MAX_SESSIONS", 2)

    async def go():
        store = SessionStore()
        await turn(store, "a", "q", "a")
        await turn(store, "b", "q", "a")
        await turn(store, "a", "q2", "a2")  # a dung lai -> b cu nhat
        await turn(store, "c", "q", "a")
        keys = list(store._entries)
        sa = await store.open("a")  # a truoc: mo b (de kiem tra no da mat) lai la mot lan vuot tran nua
        a = sa.history
        sa.release()
        return keys, (await store.open("b")).history, a

    keys, b, a = run(go())
    assert "b" not in keys  # b cu nhat -> bi don ngay khi c vuot tran
    assert b == []  # va lich su cua b mat
    assert ("user", "q2") in a  # a duoc dung gan day nen con


def test_a_session_in_use_is_never_evicted(monkeypatch):
    monkeypatch.setattr(cfg, "SESSION_MAX_SESSIONS", 1)

    async def go():
        store = SessionStore()
        busy = await store.open("busy")  # giu lock
        await busy.add_turn("q", "a", "rag", {})
        other = await store.open("other")  # vuot tran nhung "busy" dang dung
        other.release()
        in_use = "busy" in store._entries
        busy.release()
        return in_use

    assert run(go()) is True


def test_requests_on_one_session_are_serialized_and_stored_in_order():
    events = []

    async def request(store, name, hold):
        s = await store.open("k")
        events.append(("enter", name))
        await asyncio.sleep(hold)
        await s.add_turn(name, f"re:{name}", "rag", {})
        events.append(("exit", name))
        s.release()

    async def go():
        store = SessionStore()
        await asyncio.gather(request(store, "first", 0.15), request(store, "second", 0.0), request(store, "third", 0.0))
        return (await store.open("k")).history

    history = run(go())
    kinds = [k for k, _ in events]
    assert kinds == ["enter", "exit"] * 3  # khong bao gio hai request cung o trong
    assert [t for r, t in history if r == "user"] == [n for k, n in events if k == "enter"]
    assert len(history) == 6


def test_release_is_idempotent():
    async def go():
        s = await SessionStore().open("k")
        s.release()
        s.release()
        return s._entry.users, s._entry.lock.locked()

    assert run(go()) == (0, False)


def test_a_waiting_request_does_not_get_a_session_evicted_under_it(monkeypatch):
    monkeypatch.setattr(cfg, "SESSION_MAX_SESSIONS", 1)

    async def go():
        store = SessionStore()
        first = await store.open("k")
        waiter = asyncio.create_task(store.open("k"))
        await asyncio.sleep(0.01)  # waiter da vao hang cho lock (users == 2)
        await (await store.open("zzz")).add_turn("q", "a", "rag", {})  # keo theo mot dot don
        await first.add_turn("q1", "a1", "rag", {})
        first.release()
        second = await waiter
        return second.history

    assert ("user", "q1") in run(go())


def test_default_window_keeps_the_later_numbered_items_of_a_long_answer():
    answer = "1. " + "a" * 900 + " 2. MUC_HAI_NOI_DUNG " + "b" * 300

    async def go():
        store = SessionStore()
        await turn(store, "k", "q", answer)
        return (await store.open("k")).history

    (_, _), (_, kept) = run(go())
    assert "MUC_HAI_NOI_DUNG" in kept  # truoc day cat o 600 ky tu nen model tra loi khong thay muc 2
