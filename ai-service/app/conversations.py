"""Conversations: luu tru lau dai cac cuoc hoi thoai trong MongoDB, de FE liet ke / mo lai / xoa.

Khac session.py (lich su ngan han dua vao prompt, het han sau SESSION_TTL_S): day la ban ghi cho nguoi dung xem, khong bao gio vao prompt.
Mot cuoc hoi thoai = mot document `_id = session.key_for(session_id, user_id)` (cung khoa voi session: biet UUID cua nguoi khac
cung khong doc duoc), `messages` chi them vao cuoi. Chi ghi luot da hoan tat do chinh server tao ra, giong session.
shortcut: messages la mang trong mot document (16MB ~ vai nghin luot), tach ra collection rieng neu mot cuoc hoi thoai dai hon.
"""
from datetime import datetime, timezone

from . import config as cfg
from .session import key_for

_collection = None


def collection():
    """Nap pymongo luc dung (khong luc import module): khi MONGODB_URI rong service khong ton gi."""
    global _collection
    if _collection is None:
        from pymongo import AsyncMongoClient

        client = AsyncMongoClient(cfg.MONGODB_URI, serverSelectionTimeoutMS=cfg.MONGODB_TIMEOUT_MS, tz_aware=True)
        _collection = client[cfg.MONGODB_DB]["conversations"]
    return _collection


def _owner(user_id):
    return user_id[:64] if user_id else None  # cung do dai voi session.key_for


async def ensure_indexes():
    await collection().create_index([("user_id", 1), ("updated_at", -1)])  # danh sach cua mot nguoi dung, moi nhat truoc


async def add_turn(session_id, user_id, question, answer, intent):
    now = datetime.now(timezone.utc)
    await collection().update_one(
        {"_id": key_for(session_id, user_id)},
        {
            "$push": {"messages": {"$each": [
                {"role": "user", "content": question, "ts": now},
                {"role": "assistant", "content": answer, "intent": intent, "ts": now},
            ]}},
            "$set": {"updated_at": now},
            "$setOnInsert": {"user_id": _owner(user_id), "session_id": session_id, "created_at": now,
                             "title": question[:cfg.CONVERSATION_TITLE_CHARS]},
        },
        upsert=True,
    )


async def list_for(user_id, limit):
    """Tom tat cac cuoc hoi thoai cua `user_id` (khong kem tin nhan), moi nhat truoc."""
    cursor = collection().find({"user_id": _owner(user_id)}, {"messages": 0, "user_id": 0}).sort("updated_at", -1).limit(limit)
    return await cursor.to_list(limit)


async def get(session_id, user_id):
    return await collection().find_one({"_id": key_for(session_id, user_id)}, {"user_id": 0})


async def delete(session_id, user_id):
    return (await collection().delete_one({"_id": key_for(session_id, user_id)})).deleted_count == 1
