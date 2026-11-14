"""Viet lai cau tim kiem bang Qwen (Groq) de tim them mot nhanh: `find_hits` tim dense + bm25 cho ca cau goc va cau viet lai, RRF
mot lan tren moi danh sach roi moi rerank. Cau viet lai chi dung de tim, khong bao gio vao prompt tra loi hay lam moc rerank.
Loi nao cung thanh None (tim mot cau nhu cu): buoc nay khong duoc lam hong request. Cau hoi chi cho buoc nay toi da QUERY_REWRITE_TIMEOUT_S
(tinh ca ket noi, cho gioi han toc do, thu lai); loi 429 / mat ket noi thoang qua duoc thu lai mot lan sau 1s."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import groq

from .. import config as cfg
from .. import texts
from ..llm import groq_client
from ..router import guard

log = logging.getLogger("uvicorn.error")

# Chay trong luong rieng de bo cuoc dung han: groq_client.chat co the cho rat lau (ket noi roi goi, RateLimiter.acquire cho toi 60 s),
# ma day chi la buoc phu. Luong bi bo lai tu ket thuc khi het timeout cua request.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query-rewrite")

SYSTEM_PROMPT = texts.read_text("query_rewrite_prompt.txt")


def _kind(e):
    """Loai loi + loai nguyen nhan, khong bao gio ghi noi dung loi: GroqError<-RateLimitError = 429, <-APITimeoutError = timeout,
    <-APIConnectionError = mat ket noi."""
    return type(e).__name__ + (f"<-{type(e.__cause__).__name__}" if e.__cause__ else "")


def _chat(messages):
    """Goi Groq, thu lai MOT lan sau 1s khi 429 / mat ket noi (loi nhanh nen re). Khong thu lai timeout (lan sau kha nang cung treo them mot
    timeout nua) va loi khac (thieu cau hinh...). Khong dung `retries` cua chat(): backoff cua no theo Retry-After, co the cho rat lau."""
    start = time.perf_counter()
    try:
        return groq_client.chat(messages, retries=0, timeout=cfg.QUERY_REWRITE_TIMEOUT_S)
    except groq_client.GroqError as e:
        if isinstance(e.__cause__, groq.APITimeoutError):
            raise
        log.warning("query rewrite loi (%s) sau %.1fs: thu lai", _kind(e), time.perf_counter() - start)
        time.sleep(1)
        return groq_client.chat(messages, retries=0, timeout=cfg.QUERY_REWRITE_TIMEOUT_S)


def _kind(e):
    """Loai loi + loai nguyen nhan, khong bao gio ghi noi dung loi: GroqError<-RateLimitError = 429, <-APITimeoutError = timeout,
    <-APIConnectionError = mat ket noi."""
    return type(e).__name__ + (f"<-{type(e.__cause__).__name__}" if e.__cause__ else "")


def _chat(messages):
    """Goi Groq, thu lai MOT lan sau 1s khi 429 / mat ket noi (loi nhanh nen re). Khong thu lai timeout (lan sau kha nang cung treo them mot
    timeout nua) va loi khac (thieu cau hinh...). Khong dung `retries` cua chat(): backoff cua no theo Retry-After, co the cho rat lau."""
    start = time.perf_counter()
    try:
        return groq_client.chat(messages, retries=0, timeout=cfg.QUERY_REWRITE_TIMEOUT_S)
    except groq_client.GroqError as e:
        if isinstance(e.__cause__, groq.APITimeoutError):
            raise
        log.warning("query rewrite loi (%s) sau %.1fs: thu lai", _kind(e), time.perf_counter() - start)
        time.sleep(1)
        return groq_client.chat(messages, retries=0, timeout=cfg.QUERY_REWRITE_TIMEOUT_S)


def rewrite(question):
    """Cau viet lai (mot dong, toi da QUERY_REWRITE_MAX_CHARS), hoac None khi Qwen loi / tra rong / giong cau goc."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<question>\n{guard.neutralize(question)}\n</question>"},
    ]
    start = time.perf_counter()
    try:
        raw = _executor.submit(_chat, messages).result(timeout=cfg.QUERY_REWRITE_TIMEOUT_S)
    except Exception as e:  # GroqError, thieu GROQ_QWEN_MODEL / key, loi mang, het gio: chi ghi loai loi + nguyen nhan, khong ghi noi dung
        log.warning("query rewrite loi (%s) sau %.1fs: tim bang cau goc", _kind(e), time.perf_counter() - start)
        return None
    line = next((ln for ln in raw.splitlines() if ln.strip()), "")
    query = guard.normalize_question(line.strip().strip("\"'`"))[:cfg.QUERY_REWRITE_MAX_CHARS].rstrip()
    return query if query and query != question else None
