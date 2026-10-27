"""Module TTS (Text-to-Speech): chuẩn hóa văn bản học vụ HSU, phân đoạn câu thông minh,
Circuit Breaker chống nghẽn Edge-TTS, prefetch song song và cache LRU trên server
để giảm độ trễ phát âm thanh đầu tiên (TTFB ~0.2–0.4s).
"""
import asyncio
from collections import OrderedDict
import logging
import re
import time
from typing import AsyncIterator, List, Optional, Tuple

import edge_tts
import httpx

from .. import config as cfg

log = logging.getLogger("uvicorn.error")
_EDGE_TTS_SEM = asyncio.Semaphore(1)

# Circuit Breaker: Khi máy chủ Microsoft Edge-TTS chặn/đóng kết nối (NoAudioReceived),
# tự động ngắt mạch sang Google TTS trực tiếp trong 300s để không tốn thời gian chờ timeout ở các câu tiếp theo.
_EDGE_TTS_COOLDOWN_SECONDS = 300.0
_edge_tts_cooldown_until: float = 0.0

# Shared HTTP client giữ kết nối Keep-Alive tới Google TTS giúp giảm ~300ms bắt tay TLS mỗi câu
_http_client: Optional[httpx.AsyncClient] = None

# Cache LRU trên RAM cho từng phân đoạn câu đã tổng hợp: (segment, voice, rate) -> mp3_bytes
_TTS_CACHE: "OrderedDict[Tuple[str, str, str], bytes]" = OrderedDict()
_TTS_CACHE_MAX = 128

# ==============================================================================
# HẰNG SỐ CHUẨN HÓA VĂN BẢN HỌC VỤ (DÀNH CHO TTS PHÁP QUY HSU)
# ==============================================================================
ACADEMIC_TTS_RULES: List[Tuple[re.Pattern, str]] = [
    # Ký hiệu pháp quy / văn bản hành chính HSU
    (re.compile(r"\bQĐ-ĐHHS\b"), "Quyết định Đại học Hoa Sen"),
    (re.compile(r"\bTB-ĐHHS\b"), "Thông báo Đại học Hoa Sen"),
    (re.compile(r"\bĐHHS\b"), "Đại học Hoa Sen"),
    (re.compile(r"\bHSU\b"), "Đại học Hoa Sen"),
    (re.compile(r"\bTP\.\s*HCM\b", re.IGNORECASE), "Thành phố Hồ Chí Minh"),
    (re.compile(r"\bTPHCM\b", re.IGNORECASE), "Thành phố Hồ Chí Minh"),
    (re.compile(r"\bSV\b"), "sinh viên"),
    (re.compile(r"\bGV\b"), "giảng viên"),
    (re.compile(r"\bCNTT\b"), "Công nghệ thông tin"),
    (re.compile(r"\bKLTN\b"), "khóa luận tốt nghiệp"),
    (re.compile(r"\bĐRL\b"), "điểm rèn luyện"),
    (re.compile(r"\bĐTBCTL\b"), "điểm trung bình chung tích lũy"),
    (re.compile(r"\bĐTBHK\b"), "điểm trung bình học kỳ"),
    (re.compile(r"\bTC\b"), "tín chỉ"),
    (re.compile(r"\bHP\b"), "học phần"),
    # Xử lý dấu gạch chéo trong năm học và mã quyết định
    (re.compile(r"(\d{4})\s*-\s*(\d{4})"), r"\1 đến \2"),
    (re.compile(r"\b(\d+)\s*/\s*(QĐ|TB|KH|HD)\b"), r"\1 \2"),
    # Ký hiệu so sánh & phần trăm
    (re.compile(r">=\s*(\d+(?:[.,]\d+)?)"), r"lớn hơn hoặc bằng \1"),
    (re.compile(r"<=\s*(\d+(?:[.,]\d+)?)"), r"nhỏ hơn hoặc bằng \1"),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*%"), r"\1 phần trăm"),
    # Đọc điểm số thập phân kiểu 2.5 / 4.0
    (re.compile(r"(\d+)\.(\d+)\s*/\s*(\d+)\.(\d+)"), r"\1 chấm \2 trên \3 chấm \4"),
    # Loại bỏ trích dẫn nguồn file RAG trong ngoặc vuông (vd: [QuyDinhHocBong.pdf, Điều 1, Khoản 2], [quy_che.pdf])
    (re.compile(r"\[\s*[^\]]*?\.(?:pdf|docx?|txt|xlsx?)[^\]]*?\]", re.IGNORECASE), ""),
    # Loại bỏ ký tự trang trí Markdown làm TTS đọc ngắc ngứ
    (re.compile(r"[*_`#>-]{2,}"), " "),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
]


def _get_http_client() -> httpx.AsyncClient:
    """Lấy hoặc khởi tạo httpx.AsyncClient dùng chung (Keep-Alive connection pool)."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=8.0,
            headers={"User-Agent": "Mozilla/5.0"},
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )
    return _http_client


def normalize_academic_text(text: str) -> str:
    """Chuẩn hóa ký hiệu viết tắt & mã văn bản học vụ HSU trước khi đưa vào bộ đọc TTS."""
    if not text:
        return ""
    cleaned = text
    for pattern, replacement in ACADEMIC_TTS_RULES:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r"[#*`_~]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_speakable_text(text: str) -> bool:
    """Kiểm tra văn bản có chứa chữ cái hoặc số để phát âm hay không.
    Tránh gửi các chuỗi chỉ có dấu câu như '.', '...', '-', ':' sang Edge-TTS gây lỗi NoAudioReceived.
    """
    if not text:
        return False
    return bool(re.search(r"[a-zA-Z0-9à-ỹÀ-Ỹ]", text))


def _split_tts_sentences(text: str, max_chars: int = 180) -> List[str]:
    """Chia văn bản dài thành các câu/cụm ngắn <= max_chars."""
    parts: List[str] = []
    for raw_sentence in re.split(r"(?<=[.!?;:])\s+|\n+", text):
        sentence = raw_sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            split_idx = sentence.rfind(" ", 0, max_chars)
            if split_idx <= 0:
                split_idx = max_chars
            parts.append(sentence[:split_idx].strip())
            sentence = sentence[split_idx:].strip()
        if sentence:
            parts.append(sentence)
    return parts


def _split_stream_segments(text: str, first_max_chars: int = 120, next_max_chars: int = 280) -> List[str]:
    """Phân đoạn thông minh trên server:
    - Đoạn đầu tiên ngắn (<= 120 ký tự) để trả byte âm thanh đầu tiên trong ~0.25–0.4s.
    - Các đoạn tiếp theo gộp câu vừa phải (<= 280 ký tự) để prefetch mượt mà ở nền.
    """
    raw_sentences = [s for s in _split_tts_sentences(text, max_chars=next_max_chars) if is_speakable_text(s)]
    if not raw_sentences:
        return []

    segments: List[str] = []
    first = raw_sentences[0]
    if len(first) > first_max_chars:
        comma_idx = first.rfind(",", 0, first_max_chars)
        split_idx = comma_idx + 1 if comma_idx > 35 else first.rfind(" ", 0, first_max_chars)
        if split_idx <= 0:
            split_idx = first_max_chars
        segments.append(first[:split_idx].strip())
        remainder = first[split_idx:].strip()
        rest_sentences = ([remainder] if is_speakable_text(remainder) else []) + raw_sentences[1:]
    else:
        segments.append(first)
        rest_sentences = raw_sentences[1:]

    buf = ""
    for sent in rest_sentences:
        if not buf:
            buf = sent
        elif len(buf) + 1 + len(sent) <= next_max_chars:
            buf = f"{buf} {sent}"
        else:
            segments.append(buf)
            buf = sent
    if buf:
        segments.append(buf)

    return segments


def _cache_get(key: Tuple[str, str, str]) -> Optional[bytes]:
    if key in _TTS_CACHE:
        _TTS_CACHE.move_to_end(key)
        return _TTS_CACHE[key]
    return None


def _cache_put(key: Tuple[str, str, str], data: bytes) -> None:
    if not data:
        return
    _TTS_CACHE[key] = data
    _TTS_CACHE.move_to_end(key)
    while len(_TTS_CACHE) > _TTS_CACHE_MAX:
        _TTS_CACHE.popitem(last=False)


async def _fallback_google_tts_chunks(text: str) -> AsyncIterator[bytes]:
    """Fallback sang Google Translate TTS (tiếng Việt) qua persistent httpx client.
    Tải song song các cụm câu nhưng yield ngay khi cụm đầu tiên hoàn tất để giảm tối đa TTFB.
    """
    segments = [s for s in _split_tts_sentences(text, max_chars=180) if is_speakable_text(s)]
    if not segments:
        return
    client = _get_http_client()
    sem = asyncio.Semaphore(4)

    async def fetch_seg(seg: str) -> bytes:
        async with sem:
            try:
                resp = await client.get(
                    "https://translate.googleapis.com/translate_tts",
                    params={"ie": "UTF-8", "tl": "vi", "client": "tw-ob", "q": seg},
                )
                return resp.content if resp.status_code == 200 else b""
            except Exception:
                return b""

    tasks = [asyncio.create_task(fetch_seg(s)) for s in segments]
    try:
        for task in tasks:
            data = await task
            if data:
                yield data
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()


async def _stream_single_segment(
    segment: str,
    voice: str,
    rate: str,
) -> AsyncIterator[bytes]:
    """Stream từng frame MP3 của một phân đoạn đơn lẻ (có LRU cache & Circuit Breaker)."""
    global _edge_tts_cooldown_until

    cache_key = (segment, voice, rate)
    cached = _cache_get(cache_key)
    if cached is not None:
        yield cached
        return

    collected = bytearray()
    now = time.monotonic()

    # Nếu Edge-TTS không đang trong thời gian ngắt mạch (cooldown), thử gọi với timeout ngắn
    if now >= _edge_tts_cooldown_until:
        async with _EDGE_TTS_SEM:
            if time.monotonic() >= _edge_tts_cooldown_until:
                audio_received = False
                try:
                    communicate = edge_tts.Communicate(
                        text=segment,
                        voice=voice,
                        rate=rate,
                        connect_timeout=2,
                        receive_timeout=2,
                    )
                    stream_iter = communicate.stream().__aiter__()
                    while not audio_received:
                        chunk = await asyncio.wait_for(stream_iter.__anext__(), timeout=1.0)
                        if chunk.get("type") == "audio" and chunk.get("data"):
                            audio_received = True
                            data = chunk["data"]
                            collected.extend(data)
                            yield data
                    async for chunk in stream_iter:
                        if chunk.get("type") == "audio" and chunk.get("data"):
                            data = chunk["data"]
                            collected.extend(data)
                            yield data
                    if audio_received:
                        _cache_put(cache_key, bytes(collected))
                        return
                except StopAsyncIteration:
                    if audio_received:
                        _cache_put(cache_key, bytes(collected))
                        return
                    _edge_tts_cooldown_until = time.monotonic() + _EDGE_TTS_COOLDOWN_SECONDS
                    log.warning(
                        "Edge-TTS không trả dữ liệu âm thanh, kích hoạt Circuit Breaker %ds sang Google TTS.",
                        int(_EDGE_TTS_COOLDOWN_SECONDS),
                    )
                except Exception as e:
                    if audio_received:
                        _cache_put(cache_key, bytes(collected))
                        return
                    _edge_tts_cooldown_until = time.monotonic() + _EDGE_TTS_COOLDOWN_SECONDS
                    log.warning(
                        "Edge-TTS phản hồi chậm/lỗi (%s), kích hoạt Circuit Breaker %ds sang Google TTS.",
                        type(e).__name__,
                        int(_EDGE_TTS_COOLDOWN_SECONDS),
                    )

    async for chunk in _fallback_google_tts_chunks(segment):
        collected.extend(chunk)
        yield chunk
    if collected:
        _cache_put(cache_key, bytes(collected))


async def _prefetch_segment_bytes(segment: str, voice: str, rate: str) -> bytes:
    """Tổng hợp ngầm 1 phân đoạn câu ở nền để sẵn sàng phát ngay khi câu trước vừa đọc xong."""
    buf = bytearray()
    async for chunk in _stream_single_segment(segment, voice=voice, rate=rate):
        buf.extend(chunk)
    return bytes(buf)


async def stream_speech_chunks(
    text: str,
    voice: str = cfg.TTS_VOICE,
    rate: str = cfg.TTS_RATE,
    max_retries: int = 1,
) -> AsyncIterator[bytes]:
    """
    Stream âm thanh MP3 độ trễ thấp từ server:
      1. Câu đầu tiên (<= 120 ký tự) được stream ngay khi có chunk đầu tiên (~0.25–0.4s).
      2. Ngay khi câu đầu tiên phát ra byte đầu tiên, câu tiếp theo được prefetch song song ở nền.
      3. Các câu đã đọc được lưu vào LRU cache để phản hồi tức thì (0ms) nếu gặp lại.
    """
    cleaned_text = normalize_academic_text(text)
    if not is_speakable_text(cleaned_text):
        return

    segments = _split_stream_segments(cleaned_text)
    if not segments:
        return

    next_task: Optional[asyncio.Task[bytes]] = None
    try:
        # Phân đoạn 0: stream trực tiếp từng frame, và kích hoạt prefetch đoạn 1 ngay khi có chunk đầu tiên
        async for chunk in _stream_single_segment(segments[0], voice=voice, rate=rate):
            if next_task is None and len(segments) > 1:
                next_task = asyncio.create_task(
                    _prefetch_segment_bytes(segments[1], voice=voice, rate=rate)
                )
            yield chunk

        # Nếu đoạn 0 rỗng mà vẫn còn đoạn sau
        if next_task is None and len(segments) > 1:
            next_task = asyncio.create_task(
                _prefetch_segment_bytes(segments[1], voice=voice, rate=rate)
            )

        # Các phân đoạn tiếp theo: lấy từ task prefetch cuốn chiếu (pipeline)
        for idx in range(1, len(segments)):
            current_task = next_task
            if idx + 1 < len(segments):
                next_task = asyncio.create_task(
                    _prefetch_segment_bytes(segments[idx + 1], voice=voice, rate=rate)
                )
            else:
                next_task = None

            if current_task is not None:
                seg_bytes = await current_task
                if seg_bytes:
                    yield seg_bytes
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()


async def synthesize_speech_bytes(
    text: str,
    voice: str = cfg.TTS_VOICE,
    rate: str = cfg.TTS_RATE,
) -> bytes:
    """Tổng hợp giọng nói Tiếng Việt và trả về toàn bộ buffer MP3 (dùng khi cần buffer đầy đủ hoặc test)."""
    audio_chunks = bytearray()
    async for data in stream_speech_chunks(text=text, voice=voice, rate=rate):
        audio_chunks.extend(data)
    return bytes(audio_chunks)
