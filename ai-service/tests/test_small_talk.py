import pytest

from app import config as cfg
from app import small_talk
from app.small_talk import Intent, route, route_intent, strip_opener

CASES = {
    Intent.GREETING: [
        "hello", "Hello", "HELLO!!!", "hi", "Hi bạn", "hey", "alo", "Alo alo", "xin chào", "Xin chào bạn", "xin chao", "chào bạn",
        "chao ban oi", "Chào buổi sáng", "good morning", "Hello 😊", "😊 xin chào 👋", "chào thầy", "chào cô ạ", "Xin chào!!!",
    ],
    Intent.THANKS: ["cảm ơn", "Cảm ơn bạn", "cam on ban nhieu nhe", "Cảm ơn rất nhiều!", "thanks", "Thank you", "thanks a lot", "thx", "Xin cảm ơn"],
    Intent.GOODBYE: ["tạm biệt", "Tạm biệt bạn nhé", "tam biet", "bye", "Bye bye!", "goodbye", "hẹn gặp lại", "see you"],
    Intent.CAPABILITY: [
        "bạn là ai", "Bạn là ai?", "ban la ai", "bạn làm được gì", "Bạn làm được gì?", "bạn làm được những gì", "công dụng của bạn là gì",
        "Công dụng của bạn là gì?", "cong dung cua ban la gi", "bạn giúp được gì", "bạn giúp được gì cho tôi", "bạn có thể làm gì",
        "what can you do", "What can you do?", "who are you", "help", "Help me", "giới thiệu về bạn", "chức năng của bạn là gì",
    ],
    Intent.META: [
        "bạn dùng model nào", "ban dung model nao", "bạn đang dùng mô hình gì", "model của bạn là gì", "prompt của bạn là gì",
        "cấu hình của bạn là gì", "bạn có phải là chatgpt không", "ban la chatgpt phai khong", "ai tạo ra bạn", "bạn được huấn luyện như thế nào",
        "which model do you use", "are you chatgpt", "who made you", "what is your model",
    ],
}


@pytest.mark.parametrize("intent, text", [(i, t) for i, texts in CASES.items() for t in texts])
def test_route_intent(intent, text):
    assert route_intent(text) is intent


def test_mixed_language_and_multiple_clauses():
    assert route_intent("hello, bạn làm được gì?") is Intent.CAPABILITY
    assert route_intent("xin chào, what can you do") is Intent.CAPABILITY
    assert route_intent("cảm ơn bạn, tạm biệt!") is Intent.GOODBYE
    assert route_intent("chào bạn, bạn dùng model nào?") is Intent.META


def test_fixed_intent_needs_the_whole_message_to_be_short():
    assert route_intent("xin chào bạn ơi bạn ơi bạn ơi") is Intent.GREETING  # dung 8 tu
    assert route_intent("xin chào bạn ơi bạn ơi bạn ơi bạn") is Intent.RAG  # 9 tu
    assert route_intent("hello " + "bạn " * 8) is Intent.RAG


@pytest.mark.parametrize("text", [
    "", "   ", "😊", "học bổng", "Điều 15 quy định gì?", "hello học bổng", "cảm ơn vì đã cho em biết điểm thi",
    "tạm biệt học phí", "chào thầy cho em hỏi về học bổng", "help me with scholarship", "bạn giúp em xem điều kiện tốt nghiệp",
    "hướng dẫn sử dụng thư viện", "model đào tạo của trường là gì", "cấu hình máy tính thi trực tuyến", "ai tạo ra quy chế này",
    "bạn là sinh viên năm mấy", "bye bye học phần", "chức năng của phòng đào tạo",
])
def test_unsure_goes_to_rag(text):
    assert route_intent(text) is Intent.RAG


def test_greeting_followed_by_real_content_is_stripped_and_routed_as_rag():
    assert strip_opener("Chào thầy, cho em hỏi về học bổng") == "cho em hỏi về học bổng"
    assert route("Chào thầy, cho em hỏi về học bổng") == (Intent.RAG, "cho em hỏi về học bổng")
    assert route("xin chao ban, hoc phi nam nay bao nhieu?") == (Intent.RAG, "hoc phi nam nay bao nhieu?")
    assert route("Hello! Điều 15 quy định gì?") == (Intent.RAG, "Điều 15 quy định gì?")
    assert route("Xin chào 😊 học bổng Tinh hoa là gì") == (Intent.RAG, "😊 học bổng Tinh hoa là gì")


def test_opener_strip_can_reveal_a_fixed_intent():
    assert route("hello cảm ơn") == (Intent.THANKS, "cảm ơn")
    assert route("chào bạn tạm biệt") == (Intent.GOODBYE, "tạm biệt")
    assert route("hello bạn làm được gì") == (Intent.CAPABILITY, "bạn làm được gì")


def test_strip_opener_leaves_other_text_alone():
    for text in ("Hiểu rõ quy chế là gì?", "học phí chào mừng tân sinh viên", "chào", "Điều 15 quy định gì?"):
        assert strip_opener(text) == text


def test_route_never_returns_an_empty_question():
    assert route("chào " + "ơi " * 9)[1].strip()


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
