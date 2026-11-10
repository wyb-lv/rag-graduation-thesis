"""Module STT (Speech-to-Text): kiểm tra bảo mật audio, lọc khoảng lặng VAD, Groq Whisper & sửa lỗi đồng âm học vụ HSU."""
import io
import math
import re
import struct
import wave
from typing import Any, Dict, List, Tuple

from .. import config as cfg, texts
from ..llm.groq_client import get_client as get_groq_client


class AudioTooLargeError(ValueError):
    """File âm thanh vượt quá dung lượng tối đa cho phép (MAX_AUDIO_BYTES)."""


class UnsupportedAudioFormatError(ValueError):
    """Định dạng file không phải là âm thanh hợp lệ."""


# ==============================================================================
# HẰNG SỐ DOMAIN LEXICON & PHONETIC CORRECTION (DÀNH CHO STT HỌC VỤ HSU)
# ==============================================================================
DOMAIN_PROMPT = texts.read_text("stt_domain_prompt.txt")

# Bảng luật sửa lỗi đồng âm / phát âm địa phương sang thuật ngữ học vụ chuẩn
PHONETIC_RULES: List[Tuple[re.Pattern, str]] = texts.read_rules("stt_phonetic_rules.jsonl", "replace")


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

    if len(audio_bytes) > cfg.MAX_AUDIO_BYTES:
        max_mb = cfg.MAX_AUDIO_BYTES / (1024 * 1024)
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
                if _compute_rms_db(chunk, sample_width) > cfg.STT_VAD_ENERGY_THRESHOLD_DB
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
    # get_groq_client() đặt max_retries=0 cho chat(); dùng with_options để bật retry riêng cho Whisper STT
    stt_client = client.with_options(max_retries=2, timeout=20.0) if hasattr(client, "with_options") else client
    # Truyền thẳng processed_bytes (kiểu bytes) thay vì BytesIO để không bị lệch con trỏ EOF khi SDK tự động retry
    audio_file = (filename, processed_bytes)
    transcription = stt_client.audio.transcriptions.create(
        file=audio_file,
        model=cfg.STT_MODEL,
        language=cfg.STT_LANGUAGE,
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
