"""Session: lich su hoi thoai ngan han theo session_id, de cau follow up ("con hoc ky dau tien thi sao?") duoc hieu.

Moi session la mot LlamaIndex `Memory` (chi lich su chat ngan han, khong co memory block nao goi LLM) tren MOT engine SQLite
dung chung: trong RAM (SESSION_STORE=memory, mat khi restart) hoac file (sqlite). Server khong bao gio nhan lich su tu client:
chi luot da hoan tat (cau hoi chuan hoa + cau tra loi cuoi sau guard dau ra) do chinh server ghi, kem `intent`, khoa / kho va thoi diem
trong `additional_kwargs` cua tin nhan (nho vay "luot truoc la clarify" va khoa / kho con nguyen sau khi doc lai tu SQLite).
Moi session co mot asyncio.Lock: request thu hai cung session cho request dau xong (ca khi stream) roi moi doc lich su.
"""
import asyncio
import hashlib
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from . import config as cfg

log = logging.getLogger("uvicorn.error")

_FACTS = ("khoa", "kho")


def parse_id(raw):
    """session_id dang UUID chuan (co gach noi) -> chuoi chu thuong; khong phai UUID -> None."""
    try:
        text = str(uuid.UUID(raw))
    except (ValueError, TypeError, AttributeError):
        return None
    return text if text == raw.lower() else None


def key_for(session_id, user_id=None):
    """Khoa luu tru. Co danh tinh tin cay tu gateway (X-User-Id) thi gan vao khoa: biet UUID cua nguoi khac cung khong doc duoc."""
    return f"{user_id[:64]}:{session_id}" if user_id else session_id


def digest(key):
    """Dau van tay 8 ky tu de log (khong log session_id / user_id that)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def _tokens(text):
    return math.ceil(len(text) / cfg.SESSION_CHARS_PER_TOKEN)


def _cut(text, limit):
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _window(messages):
    """[(role, text)] cho prompt: toi da SESSION_MAX_TURNS luot VA SESSION_MAX_HISTORY_TOKENS token (cai nao nho hon), moi tin bi cat.
    Chi lay role + noi dung nen metadata LlamaIndex (session_id, intent, ...) khong bao gio vao prompt."""
    cap = min(cfg.SESSION_MESSAGE_MAX_CHARS, int(cfg.SESSION_MAX_HISTORY_TOKENS * cfg.SESSION_CHARS_PER_TOKEN / 2))
    items = [(m.role.value, _cut(m.content or "", cap)) for m in messages if m.role.value in ("user", "assistant")]
    items = items[-2 * cfg.SESSION_MAX_TURNS:]
    if items and items[0][0] == "assistant":
        items.pop(0)
    while items and sum(_tokens(text) + 4 for _, text in items) > cfg.SESSION_MAX_HISTORY_TOKENS:
        items = items[2:]  # bo ca mot luot (hoi + tra loi) cu nhat
    return items


def _expired(message):
    ts = message.additional_kwargs.get("ts")
    return ts is not None and time.time() - ts > cfg.SESSION_TTL_S


def _last(messages, role):
    return next((m for m in reversed(messages) if m.role.value == role), None)


@dataclass
class _Entry:
    memory: object
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    used: float = 0.0
    users: int = 0  # request dang giu hoac cho lock: entry nay khong bi don


class Session:
    """Mot session dang mo cua mot request; giu lock cho den khi release()."""

    def __init__(self, store, entry, messages):
        self._store, self._entry, self._released = store, entry, False
        self.history = _window(messages)
        last_user = _last(messages, "user")
        self.facts = {k: v for k, v in (last_user.additional_kwargs if last_user else {}).items() if k in _FACTS}
        last_reply = _last(messages, "assistant")
        # luot truoc la cau hoi lam ro: chi doc tu metadata server ghi, khong bao gio tu request
        self.previous_was_clarify = bool(last_reply and last_reply.additional_kwargs.get("intent") == "clarify")

    async def add_turn(self, question, answer, intent, facts):
        """Luu mot luot da hoan tat. `facts`: khoa / kho hieu luc cua luot nay (request de len gia tri da luu)."""
        now = time.time()
        Message = self._store.message_class
        await self._entry.memory.aput_messages([
            Message(role="user", content=question,
                    additional_kwargs={"ts": now, **{k: v for k, v in facts.items() if k in _FACTS and v is not None}}),
            Message(role="assistant", content=answer, additional_kwargs={"ts": now, "intent": str(intent)}),
        ])

    def release(self):
        if not self._released:
            self._released = True
            self._entry.users -= 1
            self._entry.used = time.monotonic()
            self._entry.lock.release()


class SessionStore:
    def __init__(self, db_uri=None):
        # Nap LlamaIndex luc dung (khong luc import module): khi SESSION_ENABLED=false service khong ton gi
        from llama_index.core.base.llms.types import ChatMessage
        from llama_index.core.memory import Memory
        from llama_index.core.storage.chat_store.sql import SQLAlchemyChatStore

        self.message_class, self._memory_class = ChatMessage, Memory
        # Mot engine chung (khong phai mot engine / thread aiosqlite cho moi session); moi session la mot khoa trong cung bang
        self._chat_store = SQLAlchemyChatStore(table_name="chat_turns", async_database_uri=db_uri)
        self._entries = OrderedDict()  # theo thu tu dung gan nhat o cuoi (LRU)
        self._swept = 0.0

    def _new_memory(self, key):
        limit = cfg.SESSION_MAX_HISTORY_TOKENS
        return self._memory_class.from_defaults(
            session_id=key,
            token_limit=limit,
            chat_history_token_ratio=1.0,  # khong co memory block nen toan bo ngan sach la lich su
            token_flush_size=max(1, limit // 10),
            tokenizer_fn=lambda text: range(_tokens(text)),  # chi can len(); khong tai tu vung tiktoken, khong can mang
            chat_store=self._chat_store,
        )

    async def open(self, key, reset=False):
        """Mo (hoac tao) session `key`, cho lock cua no. `reset`: xoa het lich su truoc. Nho goi Session.release()."""
        entry = self._entries.get(key)
        if entry is None:
            entry = self._entries[key] = _Entry(self._new_memory(key))
        entry.users += 1  # ghim truoc khi don: _evict() khong bao gio don entry dang duoc mo
        self._entries.move_to_end(key)
        entry.used = time.monotonic()
        session = None
        try:
            await self._evict()
            await entry.lock.acquire()
            try:
                if reset:
                    await entry.memory.areset()
                messages = await entry.memory.aget()
                if messages and _expired(messages[-1]):
                    await entry.memory.areset()  # luot cuoi cu hon TTL (vd dong con lai trong file SQLite tu lan chay truoc)
                    messages = []
                session = Session(self, entry, messages)
            finally:
                if session is None:
                    entry.lock.release()
        finally:
            if session is None:
                entry.users -= 1
        return session

    async def _evict(self):
        """Bo session het han (SESSION_TTL_S) va session cu nhat khi vuot SESSION_MAX_SESSIONS; bo la xoa luon lich su.
        Session dang co request giu hoac cho (users > 0) thi giu lai."""
        now = time.monotonic()
        for key in list(self._entries):
            entry = self._entries[key]
            if entry.users or not (now - entry.used > cfg.SESSION_TTL_S or len(self._entries) > cfg.SESSION_MAX_SESSIONS):
                continue
            async with entry.lock:  # users == 0 nen lock ranh; chan request moi doc lich su giua luc dang xoa
                await entry.memory.areset()
            if not entry.users:
                self._entries.pop(key, None)
        await self._sweep()

    async def _sweep(self):
        """SESSION_STORE=sqlite: xoa dong cua session het han ma khong ai goi lai (khong nam trong _entries); moi SESSION_TTL_S mot lan."""
        now = time.time()
        if cfg.SESSION_STORE != "sqlite" or now - self._swept < cfg.SESSION_TTL_S:
            return
        self._swept = now
        for key in await self._chat_store.get_keys():
            if key in self._entries:
                continue
            messages = await self._chat_store.get_messages(key, status=None)
            if not messages or _expired(messages[-1]):
                await self._chat_store.delete_messages(key)


_store = None
_store_lock = threading.Lock()


def get_store():
    """SessionStore dung chung. Nap LlamaIndex mat ~1s: trong luc xu ly request hay goi qua `store()`."""
    global _store
    with _store_lock:  # hai request dau tien khong duoc tao hai store (hai bo lock rieng)
        if _store is None:
            _store = SessionStore(cfg.SESSION_DB_URI if cfg.SESSION_STORE == "sqlite" else None)
        return _store


async def store():
    return _store or await asyncio.to_thread(get_store)
