"""Module xử lý Voice: VAD cắt khoảng lặng, Whisper STT với Từ điển Học vụ, 
Sửa lỗi đồng âm tiếng Việt, Chuẩn hóa văn bản pháp quy và Tổng hợp TTS.
"""
import io
import logging
import os
import re
import time
import wave
from typing import AsyncGenerator, List, Optional, Tuple

import edge_tts
import numpy as np

from . import config as cfg
from . import groq_client

log = logging.getLogger("uvicorn.error")

# ── 1. Từ điển Ngữ cảnh Học vụ (Domain Lexicon Prompt cho Whisper) ───────────
# Giúp Whisper định hướng xác suất âm học về các từ khóa đặc thù trường đại học
DOMAIN_PROMPT = (
    "Học bổng khuyến khích, điểm rèn luyện, GPA, tín chỉ, học kỳ, tổ hợp xét tuyển "
    "A00, A01, B00, C00, D01, chứng chỉ IELTS, TOEIC, THPT, Trường Đại học Hoa Sen, HSU, "
    "học phần, điểm F, thi lại, miễn giảm học phí, trợ cấp xã hội, xét tuyển học bạ."
)

# ── 2. Bảng quy tắc sửa lỗi nhận dạng đồng âm tiếng Việt ──────────────────────
PHONETIC_RULES: List[Tuple[re.Pattern, str]] = [
    # Tổ hợp môn xét tuyển
    (re.compile(r"\b(tổ\s*hợp\s+)?a\s*(0|không|khong)\s*(0|không|khong)\b", re.IGNORECASE), "A00"),
    (re.compile(r"\b(tổ\s*hợp\s+)?a\s*(0|không|khong)\s*(1|một|mot)\b", re.IGNORECASE), "A01"),
    (re.compile(r"\b(tổ\s*hợp\s+)?b\s*(0|không|khong)\s*(0|không|khong)\b", re.IGNORECASE), "B00"),
    (re.compile(r"\b(tổ\s*hợp\s+)?d\s*(0|không|khong)\s*(1|một|mot)\b", re.IGNORECASE), "D01"),
    (re.compile(r"\b(tổ\s*hợp\s+)?c\s*(0|không|khong)\s*(0|không|khong)\b", re.IGNORECASE), "C00"),
    # Điểm số & Thuật ngữ tiếng Anh
    (re.compile(r"\b(gờ\s*pê\s*a|di\s*pi\s*ây|d\s*p\s*a|ji\s*pi\s*ây)\b", re.IGNORECASE), "GPA"),
    (re.compile(r"\b(ai\s*eo|ai\s*el|ai\s*en|ai\s*oét)\b", re.IGNORECASE), "IELTS"),
    (re.compile(r"\b(tô\s*ích|tô\s*ít|toe\s*ic)\b", re.IGNORECASE), "TOEIC"),
    (re.compile(r"\b(điểm\s+)?(ép|e-pờ|f|ef)\b", re.IGNORECASE), "điểm F"),
    # Tên trường
    (re.compile(r"\b(đại\s*học\s+)?(hoa\s*xem|hoa\s*sơn)\b", re.IGNORECASE), "Hoa Sen"),
]


def apply_phonetic_fix(text: str, *, enable_phonetic_fix: bool = True) -> str:
    """Sửa lỗi đồng âm các thuật ngữ học vụ thường bị STT nhận dạng sai."""
    if not enable_phonetic_fix or not text:
        return text

    corrected = text
    for pattern, replacement in PHONETIC_RULES:
        corrected = pattern.sub(replacement, corrected)
    return re.sub(r"\s+", " ", corrected).strip()


# ── 3. Thuật toán VAD cắt khoảng lặng & lọc ồn (Dùng thư viện chuẩn + NumPy) ─
def preprocess_audio_vad(
    audio_bytes: bytes,
    *,
    enable_vad: bool = True,
) -> Tuple[bytes, bool, float]:
    """Cắt khoảng lặng đầu/cuối và kiểm tra có tiếng người hay không.
    
    Trả về: (audio_bytes_đã_lọc, is_speech: bool, trimmed_ms: float)
    - Chỉ chạy với file .wav (PCM 16-bit). Nếu là .mp3/.webm thì giữ nguyên byte.
    """
    if not enable_vad or not audio_bytes:
        return audio_bytes, True, 0.0

    # Kiểm tra Header RIFF/WAVE
    if not audio_bytes.startswith(b"RIFF") or b"WAVE" not in audio_bytes[:16]:
        return audio_bytes, True, 0.0

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw_frames = wf.readframes(n_frames)

        # Chỉ xử lý PCM 16-bit (sample_width == 2)
        if sample_width != 2 or framerate <= 0 or n_frames == 0:
            return audio_bytes, True, 0.0

        samples = np.frombuffer(raw_frames, dtype=np.int16)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

        total_ms = (len(samples) / framerate) * 1000.0
        frame_len = int(framerate * (cfg.VAD_FRAME_MS / 1000.0))
        if frame_len <= 0:
            return audio_bytes, True, 0.0

        n_chunks = len(samples) // frame_len
        if n_chunks == 0:
            return audio_bytes, True, 0.0

        # Tính năng lượng RMS (Root Mean Square) chuẩn hóa về [0, 1]
        chunks = samples[: n_chunks * frame_len].reshape(n_chunks, frame_len).astype(np.float32)
        rms = np.sqrt(np.mean((chunks / 32768.0) ** 2, axis=1))
        speech_mask = rms >= cfg.VAD_ENERGY_THRESHOLD

        speech_indices = np.where(speech_mask)[0]
        # Không có khung nào đạt ngưỡng hoặc tổng thời gian nói < ngưỡng tối thiểu
        if len(speech_indices) == 0:
            return b"", False, total_ms

        speech_duration_ms = len(speech_indices) * cfg.VAD_FRAME_MS
        if speech_duration_ms < cfg.VAD_MIN_SPEECH_MS:
            return b"", False, total_ms

        # Thêm đệm 100ms trước và sau để không bị cụt âm tiết đầu/cuối
        pad_frames = int(100 / cfg.VAD_FRAME_MS)
        start_idx = max(0, speech_indices[0] - pad_frames) * frame_len
        end_idx = min(len(samples), (speech_indices[-1] + 1 + pad_frames) * frame_len)

        trimmed_samples = samples[start_idx:end_idx]
        trimmed_ms = total_ms - ((len(trimmed_samples) / framerate) * 1000.0)

        # Đóng gói lại thành file WAV hợp lệ
        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(2)
            out_wf.setframerate(framerate)
            out_wf.writeframes(trimmed_samples.tobytes())

        return out_buf.getvalue(), True, max(0.0, trimmed_ms)

    except Exception as e:
        log.warning(f"Lỗi khi chạy VAD trên WAV: {e}, bỏ qua VAD.")
        return audio_bytes, True, 0.0


# ── 4. Bộ điều phối Nhận dạng Giọng nói (STT) ─────────────────────────────────
def transcribe_speech(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    *,
    enable_vad: bool = True,
    enable_domain_prompt: bool = True,
    enable_phonetic_fix: bool = True,
) -> dict:
    """Xử lý nhận dạng âm thanh thành văn bản với đầy đủ thông số đo lường."""
    start_time = time.perf_counter()

    # Bước 1: VAD Lọc khoảng lặng & chặn file rỗng/tiếng ồn
    processed_bytes, is_speech, trimmed_ms = preprocess_audio_vad(audio_bytes, enable_vad=enable_vad)
    if not is_speech or len(processed_bytes) == 0:
        return {
            "is_speech": False,
            "raw_text": "",
            "corrected_text": "",
            "vad_trimmed_ms": round(trimmed_ms, 2),
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }

    # Bước 2: Gọi Groq Whisper
    # Lưu ý: Whisper dùng get_client() nhưng KHÔNG dùng chung RateLimiter TPM của Qwen
    client = groq_client.get_client()
    clean_name = filename if filename and "." in filename else "audio.wav"
    file_tuple = (clean_name, processed_bytes)

    prompt = DOMAIN_PROMPT if enable_domain_prompt else None
    resp = client.audio.transcriptions.create(
        file=file_tuple,
        model=cfg.GROQ_STT_MODEL,
        prompt=prompt,
        language="vi",
        response_format="json",
        temperature=0.0,
    )
    raw_text = getattr(resp, "text", "") or ""
    raw_text = raw_text.strip()

    # Bước 3: Sửa lỗi đồng âm học vụ
    final_text = apply_phonetic_fix(raw_text, enable_phonetic_fix=enable_phonetic_fix)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "is_speech": True,
        "raw_text": raw_text,
        "corrected_text": final_text,
        "vad_trimmed_ms": round(trimmed_ms, 2),
        "latency_ms": round(elapsed_ms, 2),
    }


# ── 5. Chuẩn hóa Văn bản Quy chế & Tách câu cho TTS ──────────────────────────
def normalize_academic_text(text: str, *, enable_academic_norm: bool = True) -> str:
    """Lược bỏ nhãn file PDF, markdown và chuẩn hóa số điểm cho giọng đọc tự nhiên."""
    if not enable_academic_norm or not text:
        return text

    s = text

    # Xóa ký hiệu định dạng markdown
    s = re.sub(r"[*#_`~>]", "", s)

    # Chuyển trích dẫn [TenFile.pdf, Điều 1, Khoản 2] -> "theo Điều 1, Khoản 2"
    s = re.sub(r"\[[^\]]*\.pdf,\s*([^\]]+)\]", r"theo \1", s, flags=re.IGNORECASE)
    # Xóa các nhãn trích dẫn chỉ có tên file [TenFile.pdf]
    s = re.sub(r"\[[^\]]*\.pdf[^\]]*\]", "", s, flags=re.IGNORECASE)

    # Chuẩn hóa thang điểm phân số: "15/30" -> "15 trên 30"
    s = re.sub(r"\b(\d+(?:[.,]\d+)?)\s*/\s*(10|30|100)\b", r"\1 trên \2", s)

    # Chuẩn hóa dải điểm GPA: "3.60 - 4.00" -> "từ 3 chấm 60 đến 4 chấm 0"
    def _norm_range(m):
        return f"từ {m.group(1).replace('.', ' chấm ')} đến {m.group(2).replace('.', ' chấm ')}"

    s = re.sub(r"\b(\d+\.\d{1,2})\s*[-–]\s*(\d+\.\d{1,2})\b", _norm_range, s)

    # Số thập phân đơn lẻ: "3.60" -> "3 chấm 60"
    s = re.sub(r"\b(\d+)\.(\d+)\b", r"\1 chấm \2", s)

    # Mở rộng từ viết tắt phổ biến
    s = re.sub(r"\bTHPT\b", "Trung học phổ thông", s)
    s = re.sub(r"\bĐHCNTT\b", "Đại học Công nghệ Thông tin", s)
    s = re.sub(r"\bĐH\b", "Đại học", s)
    s = re.sub(r"\bQĐ\b", "Quyết định", s)
    s = re.sub(r"\bGPA\b", "G P A", s)
    s = re.sub(r"\bIELTS\b", "Ai-eo", s)
    s = re.sub(r"\b(\d+)\s*%\b", r"\1 phần trăm", s)

    return re.sub(r"\s+", " ", s).strip()


def split_into_sentences(text: str) -> List[str]:
    """Tách đoạn văn thành từng câu hoàn chỉnh (tránh cắt nhầm số thập phân 3.60 hay Điều 1.)."""
    if not text:
        return []
    # Cắt theo dấu chấm, chấm phẩy, hai chấm, dấu hỏi, dấu than hoặc xuống dòng
    raw_sentences = re.split(r"(?<=[.!?:;\n])\s+", text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
    return sentences if sentences else [text.strip()]


async def synthesize_speech_stream(
    text: str, voice: Optional[str] = None
) -> AsyncGenerator[bytes, None]:
    """Stream từng chunk âm thanh MP3 từ Edge-TTS."""
    v = voice or cfg.TTS_VOICE
    communicate = edge_tts.Communicate(text, v)
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            yield chunk["data"]