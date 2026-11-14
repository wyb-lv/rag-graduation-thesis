import groq
import httpx
import pytest

from app import config as cfg
from app.llm import groq_client

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "Câu hỏi?"}]


def _request():
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def rate_limit_error(retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after else {}
    return groq.RateLimitError("429", response=httpx.Response(429, request=_request(), headers=headers), body=None)


class Resp:
    def __init__(self, text, total_tokens=100):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]
        self.usage = type("U", (), {"total_tokens": total_tokens})()


class FakeClient:
    """Tra lan luot cac phan tu cua `script`: exception thi nem, con lai thi tra ve."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **params):
        self.calls.append(params)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def settings(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "GROQ_QWEN_MODEL", "qwen/test-model")
    monkeypatch.setattr(cfg, "GROQ_API_KEY", "key")
    monkeypatch.setattr(cfg, "GROQ_REASONING_EFFORT", "none")
    monkeypatch.setattr(cfg, "GROQ_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(cfg, "GROQ_MAX_RETRIES", 3)
    monkeypatch.setattr(cfg, "GROQ_BACKOFF_BASE", 2)
    monkeypatch.setattr(cfg, "GROQ_BACKOFF_MAX", 100)
    monkeypatch.setattr(groq_client.random, "uniform", lambda a, b: 0)


def make_limiter(rpm=30, tpm=8000):
    clock = Clock()
    return groq_client.RateLimiter(rpm, tpm, clock=clock, sleep=clock.sleep), clock


def test_chat_sends_model_and_reasoning_effort():
    client = FakeClient([Resp("Trả lời")])
    limiter, clock = make_limiter()
    assert groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=clock.sleep) == "Trả lời"
    params = client.calls[0]
    assert params["model"] == "qwen/test-model"
    assert params["reasoning_effort"] == "none"
    assert params["messages"] == MESSAGES


def test_reasoning_effort_omitted_when_blank(monkeypatch):
    monkeypatch.setattr(cfg, "GROQ_REASONING_EFFORT", "")
    client = FakeClient([Resp("ok")])
    limiter, clock = make_limiter()
    groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=clock.sleep)
    assert "reasoning_effort" not in client.calls[0]


def test_missing_model_env_raises_keyerror(monkeypatch):
    monkeypatch.setattr(cfg, "GROQ_QWEN_MODEL", None)
    with pytest.raises(KeyError, match="GROQ_QWEN_MODEL"):
        groq_client.chat(MESSAGES, client=FakeClient([]))


def test_think_block_is_stripped():
    client = FakeClient([Resp("<think>nghi</think>Đáp án")])
    limiter, clock = make_limiter()
    assert groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=clock.sleep) == "Đáp án"


def test_429_retries_with_exponential_backoff_then_succeeds():
    client = FakeClient([rate_limit_error(), rate_limit_error(), Resp("xong")])
    limiter, clock = make_limiter()
    sleeps = []
    assert groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=sleeps.append) == "xong"
    assert len(client.calls) == 3
    assert sleeps == [2, 4]  # base * 2**attempt


def test_429_honours_retry_after_when_longer_than_backoff():
    client = FakeClient([rate_limit_error(retry_after=30), Resp("xong")])
    limiter, _ = make_limiter()
    sleeps = []
    groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=sleeps.append)
    assert sleeps == [30]


def test_timeout_is_retried():
    client = FakeClient([groq.APITimeoutError(request=_request()), Resp("xong")])
    limiter, _ = make_limiter()
    assert groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=lambda s: None) == "xong"
    assert len(client.calls) == 2


def test_error_after_retry_limit_is_clear():
    client = FakeClient([rate_limit_error() for _ in range(10)])
    limiter, _ = make_limiter()
    sleeps = []
    with pytest.raises(groq_client.GroqError, match="429"):
        groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=sleeps.append)
    assert len(client.calls) == cfg.GROQ_MAX_RETRIES + 1
    assert len(sleeps) == cfg.GROQ_MAX_RETRIES  # khong ngu them sau lan thu cuoi


def test_retries_zero_makes_one_attempt_without_sleeping_and_says_so():
    client = FakeClient([rate_limit_error(), Resp("khong duoc toi")])
    limiter, _ = make_limiter()
    sleeps = []
    with pytest.raises(groq_client.GroqError, match=r"sau 1 lan thu"):
        groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=sleeps.append, retries=0)
    assert len(client.calls) == 1 and sleeps == []


def test_retries_override_does_not_change_the_default():
    client = FakeClient([rate_limit_error(), Resp("xong")])
    limiter, _ = make_limiter()
    assert groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=lambda s: None, retries=1) == "xong"
    client = FakeClient([rate_limit_error() for _ in range(10)])
    with pytest.raises(groq_client.GroqError, match=rf"sau {cfg.GROQ_MAX_RETRIES + 1} lan thu"):
        groq_client.chat([{"role": "user", "content": "khac"}], client=client, limiter=limiter, sleep=lambda s: None)  # khac MESSAGES: tranh cache
    assert len(client.calls) == cfg.GROQ_MAX_RETRIES + 1


def test_non_retryable_error_propagates_immediately():
    bad = groq.BadRequestError("bad", response=httpx.Response(400, request=_request()), body=None)
    client = FakeClient([bad, Resp("khong duoc toi")])
    limiter, _ = make_limiter()
    with pytest.raises(groq.BadRequestError):
        groq_client.chat(MESSAGES, client=client, limiter=limiter, sleep=lambda s: None)
    assert len(client.calls) == 1


def test_disk_cache_hit_skips_the_api():
    limiter, clock = make_limiter()
    first = FakeClient([Resp("lần một")])
    assert groq_client.chat(MESSAGES, client=first, limiter=limiter, sleep=clock.sleep) == "lần một"
    second = FakeClient([])  # goi API se IndexError
    assert groq_client.chat(MESSAGES, client=second, limiter=limiter, sleep=clock.sleep) == "lần một"
    assert second.calls == []


def test_cache_key_depends_on_model_and_prompt(monkeypatch):
    base = groq_client.cache_key("m1", MESSAGES)
    assert groq_client.cache_key("m1", MESSAGES) == base
    assert groq_client.cache_key("m2", MESSAGES) != base
    assert groq_client.cache_key("m1", MESSAGES[:1]) != base
    monkeypatch.setattr(cfg, "GROQ_REASONING_EFFORT", "high")
    assert groq_client.cache_key("m1", MESSAGES) != base


def test_rate_limiter_enforces_rpm():
    limiter, clock = make_limiter(rpm=30, tpm=10**9)
    for _ in range(30):
        limiter.acquire(1)
    assert clock.sleeps == []
    limiter.acquire(1)  # request thu 31 phai cho den khi request dau het cua so 60s
    assert clock.now >= 60
    assert sum(clock.sleeps) >= 60


def test_rate_limiter_enforces_tpm_and_settle_frees_budget():
    limiter, clock = make_limiter(rpm=30, tpm=8000)
    entry = limiter.acquire(5000)
    limiter.settle(entry, 1000)  # so token that nho hon uoc luong
    limiter.acquire(5000)  # 1000 + 5000 <= 8000 -> khong phai cho
    assert clock.sleeps == []
    limiter.acquire(5000)  # 1000 + 5000 + 5000 > 8000 -> phai cho
    assert clock.now >= 60


def test_rate_limiter_rejects_request_larger_than_tpm():
    limiter, _ = make_limiter(tpm=8000)
    with pytest.raises(groq_client.GroqError):
        limiter.acquire(9000)


# ---- chat_stream ----


class Chunk:
    def __init__(self, text=None, total_tokens=None):
        delta = type("D", (), {"content": text})()
        self.choices = [type("C", (), {"delta": delta})()] if text is not None else []
        usage = type("U", (), {"total_tokens": total_tokens})() if total_tokens else None
        self.x_groq = type("X", (), {"usage": usage})()
        self.usage = None


class FakeStream:
    """Lap qua `items`: exception thi nem (mo phong mat ket noi giua chung), con lai la Chunk."""

    def __init__(self, items):
        self.items, self.closed = items, False

    def __iter__(self):
        for item in self.items:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self.closed = True


def stream_of(*texts, total_tokens=50):
    return FakeStream([Chunk(t) for t in texts] + [Chunk(total_tokens=total_tokens)])


def test_chat_stream_yields_tokens_and_requests_streaming():
    client = FakeClient([stream_of("Thí ", "sinh ", "được")])
    limiter, clock = make_limiter()
    parts = list(groq_client.chat_stream(MESSAGES, client=client, limiter=limiter, sleep=clock.sleep))
    assert parts == ["Thí ", "sinh ", "được"]
    assert client.calls[0]["stream"] is True
    assert client.calls[0]["model"] == "qwen/test-model"


def test_chat_stream_settles_actual_usage_and_closes_stream():
    stream = stream_of("a", total_tokens=123)
    limiter, clock = make_limiter()
    list(groq_client.chat_stream(MESSAGES, client=FakeClient([stream]), limiter=limiter, sleep=clock.sleep))
    assert limiter._events[0][1] == 123
    assert stream.closed


def test_chat_stream_caches_full_answer_for_chat_and_stream():
    limiter, _ = make_limiter()
    list(groq_client.chat_stream(MESSAGES, client=FakeClient([stream_of("Đáp ", "án ")]), limiter=limiter))
    assert groq_client.chat(MESSAGES, client=FakeClient([]), limiter=limiter) == "Đáp án"
    assert list(groq_client.chat_stream(MESSAGES, client=FakeClient([]), limiter=limiter)) == ["Đáp án"]


def test_chat_stream_strips_think_split_across_chunks():
    client = FakeClient([stream_of("<thi", "nk>nghĩ</th", "ink>\n\nĐáp", " án")])
    limiter, _ = make_limiter()
    assert "".join(groq_client.chat_stream(MESSAGES, client=client, limiter=limiter)) == "Đáp án"


def test_think_filter_keeps_text_that_only_looks_like_a_tag():
    f = groq_client.ThinkFilter()
    assert f.feed("a <th") + f.feed("e> b <") + f.flush() == "a <the> b <"


def test_chat_stream_missing_model_raises_before_iterating(monkeypatch):
    monkeypatch.setattr(cfg, "GROQ_QWEN_MODEL", None)
    with pytest.raises(KeyError, match="GROQ_QWEN_MODEL"):
        groq_client.chat_stream(MESSAGES, client=FakeClient([]))


def test_chat_stream_retries_429_before_first_token():
    client = FakeClient([rate_limit_error(), stream_of("xong")])
    limiter, _ = make_limiter()
    sleeps = []
    assert list(groq_client.chat_stream(MESSAGES, client=client, limiter=limiter, sleep=sleeps.append)) == ["xong"]
    assert sleeps == [2]


def test_chat_stream_retries_connection_drop_before_first_token():
    dropped = FakeStream([groq.APIConnectionError(request=_request())])
    client = FakeClient([dropped, stream_of("xong")])
    limiter, _ = make_limiter()
    assert list(groq_client.chat_stream(MESSAGES, client=client, limiter=limiter, sleep=lambda s: None)) == ["xong"]
    assert dropped.closed


def test_chat_stream_does_not_retry_after_tokens_were_sent():
    dropped = FakeStream([Chunk("Thí "), groq.APIConnectionError(request=_request())])
    client = FakeClient([dropped, stream_of("khong duoc toi")])
    limiter, _ = make_limiter()
    received = []
    with pytest.raises(groq_client.GroqError, match="giua chung"):
        for text in groq_client.chat_stream(MESSAGES, client=client, limiter=limiter, sleep=lambda s: None):
            received.append(text)
    assert received == ["Thí "]
    assert len(client.calls) == 1


def test_chat_stream_error_after_retry_limit_is_clear():
    client = FakeClient([rate_limit_error() for _ in range(10)])
    limiter, _ = make_limiter()
    with pytest.raises(groq_client.GroqError, match="429"):
        list(groq_client.chat_stream(MESSAGES, client=client, limiter=limiter, sleep=lambda s: None))
    assert len(client.calls) == cfg.GROQ_MAX_RETRIES + 1


def test_closing_chat_stream_early_closes_groq_stream():
    stream = stream_of("a", "b", "c")
    limiter, _ = make_limiter()
    gen = groq_client.chat_stream(MESSAGES, client=FakeClient([stream]), limiter=limiter)
    assert next(gen) == "a"
    gen.close()  # client SSE ngat ket noi
    assert stream.closed
