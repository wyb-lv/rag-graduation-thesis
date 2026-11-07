import io
import math
import os
import re
import struct
import wave
from typing import AsyncIterator, Dict, Any, List, Tuple

import edge_tts

from . import config
from .llm.groq_client import get_client as get_groq_client


class AudioTooLargeError(ValueError):
    """File âm thanh vượt quá dung lượng tối đa cho phép (MAX_AUDIO_BYTES)."""


class UnsupportedAudioFormatError(ValueError):
    """Định dạng file không phải là âm thanh hợp lệ."""


# ==============================================================================
# 1. HẰNG SỐ DOMAIN LEXICON & PHONETIC CORRECTION (DÀNH CHO STT HỌC VỤ HSU)
# ==============================================================================
DOMAIN_PROMPT = (
    "HSU, Đại học Hoa Sen, tín chỉ, học phần, "
    "rút học phần, học phí, học bổng, điểm rèn luyện, GPA, điểm trung bình tích lũy, "
    "khóa luận tốt nghiệp, thực tập doanh nghiệp, chuẩn đầu ra, TOEIC, bảo lưu, "
    "cảnh báo học vụ, buộc thôi học, niên chế, học kỳ, đăng ký học phần, "
    "phòng Đào tạo, công tác sinh viên, quy chế đào tạo."
)

# Bảng luật sửa lỗi đồng âm / phát âm địa phương sang thuật ngữ học vụ chuẩn
PHONETIC_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\btính\s+chỉ\b", re.IGNORECASE), "tín chỉ"),
    (re.compile(r"\btín\s+chĩ\b", re.IGNORECASE), "tín chỉ"),
    (re.compile(r"\bhọc\s+bỗng\b", re.IGNORECASE), "học bổng"),
    (re.compile(r"\bđiểm\s+rèn\s+liện\b", re.IGNORECASE), "điểm rèn luyện"),
    (re.compile(r"\bkhóa\s+lận\b", re.IGNORECASE), "khóa luận"),
    (re.compile(r"\bbão\s+lưu\b", re.IGNORECASE), "bảo lưu"),
    (re.compile(r"\bquy\s+chế\s+tín\s+chĩ\b", re.IGNORECASE), "quy chế tín chỉ"),
    (re.compile(r"\bhắt\s*ét\s*u\b", re.IGNORECASE), "HSU"),
    (re.compile(r"\bhắt\s*ét\s*iu\b", re.IGNORECASE), "HSU"),
    (re.compile(r"\bgi\s*pi\s*ây\b", re.IGNORECASE), "GPA"),
    (re.compile(r"\btô\s*ích\b", re.IGNORECASE), "TOEIC"),
]

# ==============================================================================
# 2. HẰNG SỐ CHUẨN HÓA VĂN BẢN HỌC VỤ (DÀNH CHO TTS PHÁP QUY HSU)
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
    # Loại bỏ ký tự trang trí Markdown làm TTS đọc ngắc ngứ
    (re.compile(r"[*_`#>-]{2,}"), " "),
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
]

def _has_valid_audio_signature(data: bytes) -> bool:
    """Kiểm tra Magic Bytes nhận diện các định dạng container âm thanh phổ biến."""
    if len(data) < 4:
        return False
    # WAV: RIFF....WAVE
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WAVE":
        return True
    # MP3: ID3 tag hoặc MPEG sync word (0xFFE0 mask)
    if data.startswith(b"ID3") or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return True
    # OGG / Opus / Vorbis
    if data.startswith(b"OggS"):
        return True
    # FLAC
    if data.startswith(b"fLaC"):
        return True
    # WebM / Matroska (EBML header)
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return True
    # MP4 / M4A / AAC (ISO Base Media File Format: ....ftyp)
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return True
    return False


def validate_audio_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> None:
    """
    Kiểm tra tính hợp lệ và bảo mật của dữ liệu âm thanh đầu vào:
      - Không được rỗng (ValueError -> HTTP 400).
      - Không vượt quá MAX_AUDIO_BYTES 5MB (AudioTooLargeError -> HTTP 413).
      - Phải đúng Magic Bytes âm thanh hợp lệ (UnsupportedAudioFormatError -> HTTP 415).
    """
    if not audio_bytes:
        raise ValueError("File âm thanh rỗng.")

    if len(audio_bytes) > config.MAX_AUDIO_BYTES:
        max_mb = config.MAX_AUDIO_BYTES / (1024 * 1024)
        raise AudioTooLargeError(f"File âm thanh vượt quá giới hạn tối đa {max_mb:.1f} MB.")

    if not _has_valid_audio_signature(audio_bytes):
        raise UnsupportedAudioFormatError(
            f"Định dạng file '{filename}' không được hỗ trợ hoặc không phải là file âm thanh hợp lệ "
            "(chỉ chấp nhận WAV, MP3, M4A, WebM, OGG, FLAC)."
        )


def apply_phonetic_fix(text: str) -> str:
    """Sửa lỗi đồng âm / phát âm địa phương sang từ vựng học vụ chuẩn."""
    if not text:
        return text
    result = text
    for pattern, replacement in PHONETIC_RULES:
        result = pattern.sub(replacement, result)
    return re.sub(r"\s+", " ", result).strip()


def _compute_rms_db(pcm_bytes: bytes, sample_width: int) -> float:
    """Tính năng lượng âm thanh RMS (dBFS) cho frame PCM 16-bit."""
    if not pcm_bytes or sample_width != 2:
        return 0.0
    count = len(pcm_bytes) // 2
    if count == 0:
        return -100.0
    shorts = struct.unpack(f"<{count}h", pcm_bytes[: count * 2])
    sum_sq = sum(s * s for s in shorts)
    rms = math.sqrt(sum_sq / count) / 32768.0
    return 20.0 * math.log10(max(rms, 1e-9))


def preprocess_audio_vad(audio_bytes: bytes) -> Tuple[bytes, bool]:
    """
    Lọc khoảng lặng đầu/cuối và kiểm tra có tiếng nói hay không (Voice Activity Detection).
    Chỉ áp dụng trực tiếp trên định dạng WAV chuẩn PCM 16-bit; các định dạng nén (webm/mp3/ogg/m4a)
    sẽ đi thẳng qua nếu dung lượng vượt ngưỡng tối thiểu.
    Trả về: (processed_bytes, is_speech)
    """
    if len(audio_bytes) < 512:
        return audio_bytes, False

    if not audio_bytes.startswith(b"RIFF"):
        return audio_bytes, True

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()

            if sample_width != 2 or n_frames == 0:
                return audio_bytes, True

            frame_ms = 30
            frames_per_chunk = max(1, int(framerate * frame_ms / 1000))
            bytes_per_frame = n_channels * sample_width

            raw_pcm = wf.readframes(n_frames)
            chunk_byte_size = frames_per_chunk * bytes_per_frame
            chunks = [
                raw_pcm[i : i + chunk_byte_size]
                for i in range(0, len(raw_pcm), chunk_byte_size)
            ]

            voiced_indices = [
                idx
                for idx, chunk in enumerate(chunks)
                if _compute_rms_db(chunk, sample_width) > config.STT_VAD_ENERGY_THRESHOLD_DB
            ]

            if not voiced_indices:
                return audio_bytes, False

            start_idx = max(0, voiced_indices[0] - 2)
            end_idx = min(len(chunks), voiced_indices[-1] + 3)
            trimmed_pcm = b"".join(chunks[start_idx:end_idx])

            out_buf = io.BytesIO()
            with wave.open(out_buf, "wb") as out_wf:
                out_wf.setnchannels(n_channels)
                out_wf.setsampwidth(sample_width)
                out_wf.setframerate(framerate)
                out_wf.writeframes(trimmed_pcm)

            return out_buf.getvalue(), True
    except Exception:
        return audio_bytes, True


def transcribe_speech(
    audio_bytes: bytes,
    filename: str = "audio.wav",
) -> Dict[str, Any]:
    """
    Chuyển giọng nói thành văn bản (STT) qua 3 tầng:
      1. Kiểm tra bảo mật & VAD lọc khoảng lặng, chặn nhiễu rỗng trước khi gọi API.
      2. Groq Whisper kèm Domain Prompt từ vựng học vụ HSU.
      3. Hậu xử lý ngữ âm (Phonetic Post-Correction).
    """
    validate_audio_bytes(audio_bytes, filename=filename)

    processed_bytes, is_speech = preprocess_audio_vad(audio_bytes)
    if not is_speech:
        return {
            "is_speech": False,
            "text": "",
        }

    client = get_groq_client()
    audio_file = (filename, io.BytesIO(processed_bytes))

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model=os.environ["STT_MODEL"],  # model STT trên Groq, đặt trong ai-service/.env
        language=config.STT_LANGUAGE,
        prompt=DOMAIN_PROMPT,
        response_format="json",
        temperature=0.0,
    )

    raw_text = (getattr(transcription, "text", "") or "").strip()
    text = apply_phonetic_fix(raw_text)

    return {
        "is_speech": bool(text),
        "text": text,
    }


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


async def stream_speech_chunks(
    text: str,
    voice: str = config.TTS_VOICE,
    rate: str = config.TTS_RATE,
) -> AsyncIterator[bytes]:
    """
    Stream từng chunk âm thanh MP3 trực tiếp từ Edge-TTS Neural ngay khi vừa sinh ra,
    giúp trình duyệt phát âm thanh đầu tiên (TTFA) nhanh mà không cần gom toàn bộ vào RAM.
    """
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            yield chunk["data"]


async def synthesize_speech_bytes(
    text: str,
    voice: str = config.TTS_VOICE,
    rate: str = config.TTS_RATE,
) -> bytes:
    """Tổng hợp giọng nói Tiếng Việt (Edge-TTS Neural) và trả về toàn bộ buffer MP3."""
    audio_chunks = bytearray()
    async for data in stream_speech_chunks(text=text, voice=voice, rate=rate):
        audio_chunks.extend(data)
    return bytes(audio_chunks)
