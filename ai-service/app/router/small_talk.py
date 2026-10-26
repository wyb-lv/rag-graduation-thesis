"""Cac y dinh (intent) cua mot cau hoi va cau tra loi co dinh cua chung. Viec PHAN LOAI do router LLM (router_llm.py) lam;
khong con luat tu khoa nao quyet dinh chao hoi / tan cong / ngoai pham vi."""
from enum import StrEnum

from .. import config as cfg


class Intent(StrEnum):
    RAG = "rag"
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    CAPABILITY = "capability"
    META = "meta"  # hoi ve model / prompt / cau hinh noi bo
    BLOCKED = "blocked"  # router danh dau injection, hoac guard dau ra chan cau tra loi
    OFF_TOPIC = "off_topic"  # ngoai pham vi quy dinh cua truong
    CLARIFY = "clarify"  # cau hoi thieu chu de / doi tuong / thoi diem, hoi lai nguoi dung


def reply_for(intent):
    return getattr(cfg, f"GUARD_REPLY_{intent.name}")
