"""Goi Groq (Qwen): gioi han toc do client (RPM/TPM), retry backoff khi 429/timeout, cache dia theo (model, prompt)."""
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import deque

import groq

from .. import config as cfg


class GroqError(RuntimeError):
    """Goi Groq that bai sau khi het so lan thu lai (hoac yeu cau khong the thoa gioi han)."""


class RateLimiter:
    """Cua so truot 60 giay: toi da `rpm` request va `tpm` token. acquire() cho den khi con ngan sach."""

    def __init__(self, rpm, tpm, clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.tpm = rpm, tpm
        self._clock, self._sleep = clock, sleep
        self._events = deque()  # [thoi diem, so token] theo thu tu thoi gian
        self._lock = threading.Lock()

    def acquire(self, tokens):
        if tokens > self.tpm:
            raise GroqError(f"Prompt uoc luong {tokens} token vuot gioi han {self.tpm} token/phut")
        while True:
            with self._lock:
                now = self._clock()
                while self._events and now - self._events[0][0] >= 60:
                    self._events.popleft()
                used = sum(e[1] for e in self._events)
                if len(self._events) < self.rpm and used + tokens <= self.tpm:
                    entry = [now, tokens]
                    self._events.append(entry)
                    return entry
                wait = 60 - (now - self._events[0][0])
            self._sleep(max(wait, 0.01))

    def settle(self, entry, actual_tokens):
        """Thay so token uoc luong bang so token that Groq bao ve."""
        with self._lock:
            entry[1] = actual_tokens


def estimate_tokens(messages, max_tokens):
    chars = sum(len(m["content"]) for m in messages)
    return int(chars / cfg.GROQ_CHARS_PER_TOKEN) + max_tokens


def cache_key(model, messages):
    """Hash cua (model, prompt) + tham so sinh, de doi reasoning/temperature khong tra ve cau cu."""
    blob = json.dumps(
        {
            "model": model,
            "messages": messages,
            "reasoning_effort": cfg.GROQ_REASONING_EFFORT,
            "max_tokens": cfg.GROQ_MAX_TOKENS,
            "temperature": cfg.GROQ_TEMPERATURE,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_path(key):
    return os.path.join(cfg.GROQ_CACHE_DIR, f"{key}.json")


def _cache_get(key):
    try:
        with open(_cache_path(key), "r", encoding="utf-8") as f:
            return json.load(f)["answer"]
    except (OSError, ValueError, KeyError):
        return None


def _cache_put(key, model, answer):
    os.makedirs(cfg.GROQ_CACHE_DIR, exist_ok=True)
    tmp = _cache_path(key) + f".{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"model": model, "answer": answer}, f, ensure_ascii=False)
    os.replace(tmp, _cache_path(key))  # ghi nguyen tu: request song song khong doc phai file do


def _retry_after(err):
    try:
        return float(err.response.headers.get("retry-after", 0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


_client = None
_limiter = None
_singleton_lock = threading.Lock()


def get_client():
    global _client
    with _singleton_lock:
        if _client is None:
            # max_retries=0: viec thu lai do chat() quan ly (backoff + rate limiter), SDK khong tu thu lai.
            _client = groq.Groq(api_key=cfg.GROQ_API_KEY, max_retries=0, timeout=cfg.GROQ_TIMEOUT)
        return _client


def get_limiter():
    global _limiter
    with _singleton_lock:
        if _limiter is None:
            _limiter = RateLimiter(cfg.GROQ_RPM, cfg.GROQ_TPM)
        return _limiter


def _check_config(client):
    model = cfg.GROQ_QWEN_MODEL
    if not model:
        raise KeyError("GROQ_QWEN_MODEL")
    if not (cfg.GROQ_API_KEY or client):
        raise KeyError("GROQ_API_KEY")
    return model


def _params(model, messages):
    params = {
        "model": model,
        "messages": messages,
        "temperature": cfg.GROQ_TEMPERATURE,
        "max_tokens": cfg.GROQ_MAX_TOKENS,
    }
    if cfg.GROQ_REASONING_EFFORT:
        params["reasoning_effort"] = cfg.GROQ_REASONING_EFFORT
    return params


def _retry_error(last_error, retries=None):
    retries = cfg.GROQ_MAX_RETRIES if retries is None else retries
    kind = "429 (vuot gioi han toc do)" if isinstance(last_error, groq.RateLimitError) else "timeout/mat ket noi"
    return GroqError(f"Groq loi {kind} sau {retries + 1} lan thu, hay thu lai sau: {last_error}")


def chat(messages, client=None, limiter=None, sleep=time.sleep, retries=None):
    """Gui `messages` toi model Qwen (GROQ_QWEN_MODEL), tra ve noi dung tra loi. Co cache dia + retry.
    `retries` (mac dinh GROQ_MAX_RETRIES): so lan thu lai; buoc phu nhu viet lai cau hoi dat 0 de loi thi bo qua ngay, khong cho backoff."""
    model = _check_config(client)
    retries = cfg.GROQ_MAX_RETRIES if retries is None else retries

    key = cache_key(model, messages)
    if cfg.GROQ_CACHE_ENABLED:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    client = client or get_client()
    limiter = limiter or get_limiter()
    params = _params(model, messages)

    estimate = estimate_tokens(messages, cfg.GROQ_MAX_TOKENS)
    last_error = None
    for attempt in range(retries + 1):
        entry = limiter.acquire(estimate)
        backoff = min(cfg.GROQ_BACKOFF_MAX, cfg.GROQ_BACKOFF_BASE * 2 ** attempt) + random.uniform(0, 1)
        try:
            resp = client.chat.completions.create(**params)
        except groq.RateLimitError as e:  # HTTP 429: ton 1 request, cho theo Retry-After neu server bao
            last_error, delay = e, max(backoff, _retry_after(e))
        except groq.APIConnectionError as e:  # gom ca APITimeoutError
            last_error, delay = e, backoff
        else:
            usage = getattr(resp, "usage", None)
            limiter.settle(entry, getattr(usage, "total_tokens", None) or estimate)
            text = resp.choices[0].message.content or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
            if cfg.GROQ_CACHE_ENABLED and text:
                _cache_put(key, model, text)
            return text
        if attempt < retries:
            sleep(delay)

    raise _retry_error(last_error, retries) from last_error


class ThinkFilter:
    """Bo khoi <think>...</think> khoi luong token; the co the bi cat ngang giua hai chunk."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self._buf, self._inside = "", False

    def feed(self, text):
        self._buf += text
        out = []
        while True:
            tag = self.CLOSE if self._inside else self.OPEN
            i = self._buf.find(tag)
            if i < 0:
                break
            if not self._inside:
                out.append(self._buf[:i])
            self._buf, self._inside = self._buf[i + len(tag):], not self._inside
        # Giu lai phan duoi co the la dau cua the, cho chunk sau
        keep = next((k for k in range(len(tag) - 1, 0, -1) if self._buf.endswith(tag[:k])), 0)
        if not self._inside:
            out.append(self._buf[:len(self._buf) - keep])
        self._buf = self._buf[len(self._buf) - keep:]
        return "".join(out)

    def flush(self):
        rest = "" if self._inside else self._buf
        self._buf = ""
        return rest


def _chunk_usage(chunk):
    usage = getattr(chunk, "usage", None) or getattr(getattr(chunk, "x_groq", None), "usage", None)
    return getattr(usage, "total_tokens", None)


def chat_stream(messages, client=None, limiter=None, sleep=time.sleep):
    """Nhu chat() nhung tra ve generator cac doan text ngay khi Groq sinh ra (cache hit -> mot doan duy nhat).

    Kiem tra cau hinh ngay khi goi (KeyError), con request Groq chi bat dau khi lap generator.
    Chi thu lai khi loi xay ra truoc token dau tien; loi giua chung thi nem GroqError vi nguoi nhan da co mot phan.
    """
    model = _check_config(client)
    return _stream(model, messages, client, limiter, sleep)


def _stream(model, messages, client, limiter, sleep):
    key = cache_key(model, messages)
    if cfg.GROQ_CACHE_ENABLED:
        cached = _cache_get(key)
        if cached is not None:
            yield cached
            return

    client = client or get_client()
    limiter = limiter or get_limiter()
    params = _params(model, messages)

    estimate = estimate_tokens(messages, cfg.GROQ_MAX_TOKENS)
    last_error = None
    for attempt in range(cfg.GROQ_MAX_RETRIES + 1):
        entry = limiter.acquire(estimate)
        backoff = min(cfg.GROQ_BACKOFF_MAX, cfg.GROQ_BACKOFF_BASE * 2 ** attempt) + random.uniform(0, 1)
        stream, parts, total_tokens = None, [], None
        try:
            stream = client.chat.completions.create(**params, stream=True)
            think = ThinkFilter()
            for chunk in stream:
                total_tokens = _chunk_usage(chunk) or total_tokens
                piece = chunk.choices[0].delta.content if chunk.choices else None
                text = think.feed(piece) if piece else ""
                if not parts:
                    text = text.lstrip()
                if text:
                    parts.append(text)
                    yield text
            tail = think.flush()
            if not parts:
                tail = tail.lstrip()
            if tail:
                parts.append(tail)
                yield tail
        except (groq.RateLimitError, groq.APIConnectionError) as e:
            if parts:
                raise GroqError(f"Groq ngat ket noi giua chung cau tra loi: {e}") from e
            last_error = e
            delay = max(backoff, _retry_after(e)) if isinstance(e, groq.RateLimitError) else backoff
        else:
            limiter.settle(entry, total_tokens or estimate)
            text = "".join(parts).strip()
            if cfg.GROQ_CACHE_ENABLED and text:
                _cache_put(key, model, text)
            return
        finally:
            if stream is not None:
                stream.close()
        if attempt < cfg.GROQ_MAX_RETRIES:
            sleep(delay)

    raise _retry_error(last_error) from last_error
