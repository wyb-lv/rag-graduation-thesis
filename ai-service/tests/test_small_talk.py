"""Cau tra loi co dinh cho cac y dinh. Viec chon y dinh la cua router LLM (xem test_router_llm.py), khong con luat tu khoa."""
import pytest

from app import config as cfg
from app.router import small_talk
from app.router.small_talk import Intent


@pytest.mark.parametrize("intent", [Intent.GREETING, Intent.CAPABILITY, Intent.META, Intent.THANKS, Intent.GOODBYE])
def test_every_fixed_intent_has_a_vietnamese_reply_that_names_neither_model_nor_vendor(intent):
    reply = small_talk.reply_for(intent)
    assert reply
    if intent not in (Intent.THANKS, Intent.GOODBYE):
        assert "Hoa Sen" in reply
    for banned in ("Qwen", "Groq", "Alibaba", "OpenAI", "GPT", "Claude", "LLM"):
        assert banned.lower() not in reply.lower()


def test_greeting_and_capability_replies_cover_what_the_service_does():
    for intent in (Intent.GREETING, Intent.CAPABILITY):
        reply = small_talk.reply_for(intent)
        for part in ("quy định", "thông báo", "biểu mẫu", "hướng dẫn", "trích dẫn", "Điều", "Khoản", "liên hệ"):
            assert part in reply, (intent, part)
    assert "không chia sẻ" in small_talk.reply_for(Intent.META)
    assert len(small_talk.reply_for(Intent.THANKS)) < 100 and len(small_talk.reply_for(Intent.GOODBYE)) < 100


def test_reply_texts_come_from_config(monkeypatch):
    monkeypatch.setattr(cfg, "GUARD_REPLY_GREETING", "Xin chào, đây là thử nghiệm")
    assert small_talk.reply_for(Intent.GREETING) == "Xin chào, đây là thử nghiệm"
