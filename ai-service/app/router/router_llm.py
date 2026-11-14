"""Router LLM: mot lan goi Gemini (REST generateContent qua httpx) tu suy luan cau hoi thuoc loai nao (chao hoi, tan cong, ngoai pham vi, thieu thong
tin, hay tra cuu) va viet lai cau follow up thanh cau tim kiem doc lap. Khong co danh sach tu khoa nao.

`route()` tra ve RouteDecision (hoac nem RouterError); `resolve()` bat loi thanh Routed(decision=None, failure=...) de
api cho cau hoi di thang vao RAG; `apply_policy()` doi RouteDecision thanh hanh dong (tra loi co dinh / di tiep RAG).
Gemini chi duoc gui: cau hoi da chuan hoa, vai luot lich su cuoi (da cat), khoa/kho va co previous_was_clarify. Khong gui doan trich,
payload, system prompt, bi mat hay dinh danh nguoi dung. Moi cau tra loi hien ra cho nguoi dung deu lay tu config, khong bao gio
tu chuoi do router sinh ra. Chi `_complete` cham vao mang, nen test thay no la du (khong goi mang).
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field

from .. import config as cfg
from .. import texts
from . import guard, small_talk
from .small_talk import Intent

log = logging.getLogger("uvicorn.error")

ROUTER_PROMPT_VERSION = "2"  # doi prompt / schema thi tang so nay: cache quyet dinh khoa theo no


class RouteDecision(BaseModel):
    intent: Literal["rag", "off_topic", "clarify", "greeting", "thanks", "goodbye", "capability", "meta", "injection"]
    confidence: float = Field(ge=0, le=1)
    # Khong co gia tri mac dinh: schema gui Gemini danh dau ca bon truong la bat buoc. Co mac dinh thi Gemini bo qua khoa `missing`
    # (do 2026-10: clarify luon thieu `missing`), va clarify khong co `missing` thi bi coi la rag.
    standalone_query: str | None  # chi cho rag, con lai null
    missing: Literal["topic", "audience", "time"] | None  # chi cho clarify, con lai null


class RouterError(Exception):
    """Router khong cho ra quyet dinh dung duoc. `kind`: timeout | quota | rate_limited | invalid_json | error."""

    def __init__(self, kind):
        super().__init__(kind)
        self.kind = kind


# ---- Prompt ----

ROUTER_PROMPT = texts.read_text("router_prompt.txt")

_EXAMPLES_DIR = Path(__file__).parent / "examples"
_CLARIFY_TEXT_REF = re.compile(r"\{(CLARIFY_ANSWER_\w+)\}")  # cau hoi lai co dinh lay tu config, khong chep lai vao file vi du


def _history_text(text):
    ref = _CLARIFY_TEXT_REF.fullmatch(text)
    return getattr(cfg, ref.group(1)) if ref else text


def _load_examples(directory=_EXAMPLES_DIR):
    """Vi du few-shot cho prompt: moi dong cua examples/*.jsonl (theo thu tu ten file) la mot vi du JSON, doi thanh
    (khoa, previous_was_clarify, history, question, decision). Dong trong va dong bat dau bang # bi bo qua. Sai o dau la nem ValueError
    ngay luc import, kem ten file va so dong."""
    examples = []
    for path in sorted(directory.glob("*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                e = json.loads(line)
                history = [(role, _history_text(text)) for role, text in e.get("history", [])]
                decision = {"intent": e["intent"], "confidence": e["confidence"], "standalone_query": e.get("query"), "missing": e.get("missing")}
                examples.append((e.get("khoa"), bool(e.get("previous_was_clarify")), history, e["question"], decision))
            except (ValueError, KeyError, TypeError, AttributeError) as err:
                raise ValueError(f"{path.name}:{number}: vi du khong hop le ({type(err).__name__}: {err})") from err
    if not examples:
        raise ValueError(f"khong co vi du router nao trong {directory}")
    return examples


# (khoa, previous_was_clarify, history, question, decision)
_EXAMPLES = _load_examples()

def _clip(text, limit):
    text = " ".join(guard.neutralize(text).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def render_input(question, history, facts, previous_was_clarify):
    """Noi dung user message gui cho router. `history`: [(role, text)] cu -> moi; chi lay ROUTER_HISTORY_TURNS luot cuoi, moi tin bi cat."""
    kho = _clip(facts["kho"], 64) if facts.get("kho") else None
    lines = "".join(
        f"{role}: {_clip(text, cfg.ROUTER_HISTORY_MSG_CHARS)}\n" for role, text in history[-2 * cfg.ROUTER_HISTORY_TURNS:]
    )
    block = f"<history>\n{lines}</history>\n" if lines else ""
    return (
        f"session_facts: {json.dumps({'khoa': facts.get('khoa'), 'kho': kho}, ensure_ascii=False)}\n"
        f"previous_was_clarify: {'true' if previous_was_clarify else 'false'}\n"
        f"{block}<question>\n{guard.neutralize(question)}\n</question>"
    )


def _examples_block():
    return "\n\n".join(
        f"{render_input(q, hist, {'khoa': khoa, 'kho': None}, prev)}\noutput: {json.dumps(out, ensure_ascii=False)}"
        for khoa, prev, hist, q, out in _EXAMPLES
    )


SYSTEM_PROMPT = ROUTER_PROMPT + "\n" + _examples_block()


def build_messages(question, history, facts, previous_was_clarify):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_input(guard.normalize_question(question), history, facts, previous_was_clarify)},
    ]


def _body(messages):
    system, user = messages[0]["content"], messages[1]["content"]
    return {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": cfg.ROUTER_MAX_TOKENS,
            "responseMimeType": "application/json",
            "responseJsonSchema": RouteDecision.model_json_schema(),  # ca bon truong bat buoc; ta van tu validate sau do
            # xem config: "none" (= minimal) lam Gemini 3.5 flash-lite lap vo han
            "thinkingConfig": {"thinkingLevel": "minimal" if cfg.ROUTER_REASONING_EFFORT == "none" else cfg.ROUTER_REASONING_EFFORT},
        },
    }


async def _complete(messages):
    """(noi dung tra ve, ten model da tra loi). Thu lai ROUTER_MAX_RETRIES lan khi timeout / mat ket noi / 5xx; 429 va 4xx khac khong thu lai."""
    # shortcut: moi lan route mo ket noi TLS moi (~100-200 ms), dung chung mot AsyncClient neu do thay dang ke
    async with httpx.AsyncClient(timeout=None) as client:  # timeout tong do asyncio.timeout lo, khong phai timeout theo pha cua httpx
        for _ in range(cfg.ROUTER_MAX_RETRIES + 1):
            try:
                async with asyncio.timeout(cfg.ROUTER_TIMEOUT_S):
                    resp = await client.post(
                        cfg.GEMINI_URL.format(cfg.ROUTER_MODEL), json=_body(messages), headers={"x-goog-api-key": cfg.GEMINI_API_KEY or ""}
                    )
                resp.raise_for_status()  # khoa nam o header, khong o URL, nen thong bao loi khong lo khoa
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code < 500:
                    raise
                err = e
            except (TimeoutError, httpx.TransportError) as e:
                err = e
        else:
            raise err
    data = resp.json()
    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought")), data.get("modelVersion") or cfg.ROUTER_MODEL


# ---- Gioi han toc do + cache ----

_calls = deque()  # thoi diem cac lan goi trong 60s gan nhat


def _acquire():
    now = time.monotonic()
    while _calls and now - _calls[0] >= 60:
        _calls.popleft()
    if len(_calls) >= cfg.ROUTER_RPM:
        return False
    _calls.append(now)
    return True


_cache = OrderedDict()  # key -> (het han, RouteDecision)


def _cache_key(question, facts):
    blob = json.dumps([ROUTER_PROMPT_VERSION, cfg.ROUTER_MODEL, question, facts.get("khoa"), facts.get("kho")])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_get(key):
    item = _cache.get(key)
    if item is None:
        return None
    if item[0] < time.monotonic():
        del _cache[key]
        return None
    _cache.move_to_end(key)
    return item[1]


def _cache_put(key, decision):
    _cache[key] = (time.monotonic() + cfg.ROUTER_CACHE_TTL_S, decision)
    _cache.move_to_end(key)
    while len(_cache) > cfg.ROUTER_CACHE_SIZE:
        _cache.popitem(last=False)


def _failure_class(exc):
    text = f"{type(exc).__name__} {exc}".lower()  # chi de khop tu khoa; noi dung loi khong bao gio duoc log
    if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
        return "timeout"
    if getattr(exc, "status_code", None) == 429 or any(w in text for w in ("ratelimit", "rate limit", "quota", "resource_exhausted", "429")):
        return "quota"
    return "error"


def _parse(raw):
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return RouteDecision.model_validate_json(raw)


# ---- Quyet dinh ----

async def _route(question, history, facts, previous_was_clarify):
    question = guard.normalize_question(question)
    # Cache chi cho request khong co lich su; khoa/kho nam trong khoa cache vi chung doi ket qua (audience)
    key = _cache_key(question, facts) if not history else None
    if key and (hit := _cache_get(key)) is not None:
        return hit, cfg.ROUTER_MODEL, True
    if not _acquire():
        raise RouterError("rate_limited")
    messages = build_messages(question, history, facts, previous_was_clarify)
    for _ in range(2):  # lan hai chi khi JSON khong hop le
        try:
            raw, model = await _complete(messages)
        except Exception as e:
            raise RouterError(_failure_class(e)) from None
        try:
            decision = _parse(raw)
        except ValueError:
            continue
        if key:
            _cache_put(key, decision)
        return decision, model, False
    raise RouterError("invalid_json")


async def route(question, history, session_facts, previous_was_clarify=False):
    """RouteDecision cho mot cau hoi. `history`: [(role, text)]; `session_facts`: {"khoa", "kho"} sau khi gop request len gia tri da luu."""
    return (await _route(question, history, session_facts, previous_was_clarify))[0]


@dataclass
class Routed:
    decision: RouteDecision | None  # None = router loi: cau hoi di thang vao RAG
    failure: str | None = None
    model: str = ""
    cached: bool = False
    latency_ms: int = 0


async def resolve(question, history, session_facts, previous_was_clarify=False):
    """Nhu route() nhung khong bao gio nem: moi loi cua router thanh Routed(decision=None, failure=<loai loi>)."""
    start = time.perf_counter()
    try:
        decision, model, cached = await _route(question, history, session_facts, previous_was_clarify)
        failure = None
    except RouterError as e:
        decision, model, cached, failure = None, cfg.ROUTER_MODEL, False, e.kind
    except Exception:  # loi bat ngo trong chinh router cung khong duoc phep lam hong request
        decision, model, cached, failure = None, cfg.ROUTER_MODEL, False, "error"
    return Routed(decision, failure, model, cached, int((time.perf_counter() - start) * 1000))


@dataclass(frozen=True)
class Action:
    intent: Intent
    answer: str | None = None  # cau tra loi co dinh; None = di tiep pipeline RAG
    in_scope: bool = True
    query: str | None = None  # cau tim kiem (chi khi answer is None)
    store: bool = True  # co luu luot nay vao lich su khong
    note: str = ""  # ly do dac biet de log


_CLARIFY_TEXT = {
    "topic": lambda: cfg.CLARIFY_ANSWER_TOPIC,
    "audience": lambda: cfg.CLARIFY_ANSWER_AUDIENCE,
    "time": lambda: cfg.CLARIFY_ANSWER_TIME,
}


def _query(decision, question):
    query = guard.normalize_question(" ".join((decision.standalone_query or "").split()))[:cfg.ROUTER_QUERY_MAX_CHARS].rstrip()
    return query or question


def apply_policy(decision, question, facts, previous_was_clarify):
    """RouteDecision -> Action, theo thu tu: injection, off_topic, clarify, nguong chung, small talk. Duoi nguong -> rag."""
    intent, conf = decision.intent, decision.confidence
    rag = Action(Intent.RAG, query=_query(decision, question))
    if intent == "injection":
        if conf >= cfg.ROUTER_BLOCK_CONFIDENCE:
            return Action(Intent.BLOCKED, cfg.GUARD_REFUSAL_ANSWER, in_scope=False, store=False)
        return replace(rag, note="injection_below_threshold")
    if intent == "off_topic":
        if conf >= cfg.ROUTER_OFFTOPIC_CONFIDENCE:
            return Action(Intent.OFF_TOPIC, cfg.OFF_TOPIC_ANSWER, in_scope=False, store=False)
        return rag
    if intent == "clarify":
        asks = decision.missing  # clarify khong co `missing` thi khong co cau hoi co dinh de dung -> rag
        if (asks and conf >= cfg.ROUTER_CLARIFY_CONFIDENCE and not previous_was_clarify
                and not (asks == "audience" and facts.get("khoa"))):
            return Action(Intent.CLARIFY, _CLARIFY_TEXT[asks](), in_scope=True)
        return rag
    if intent != "rag" and conf >= cfg.ROUTER_MIN_CONFIDENCE:
        return Action(Intent(intent), small_talk.reply_for(Intent(intent)))
    return rag
