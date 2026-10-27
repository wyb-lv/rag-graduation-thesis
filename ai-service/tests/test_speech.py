import io
import math
import struct
import wave
import pytest

from app import speech, config


def make_wav_bytes(freq: float = 440.0, duration_s: float = 0.5, amplitude: int = 12000) -> bytes:
    """Tạo dữ liệu WAV PCM 16-bit mono phục vụ kiểm thử."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            val = int(amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def test_apply_phonetic_fix_replaces_academic_homophones():
    raw = "Cho em hỏi quy chế tính chỉ và điều kiện xét học bỗng tại hắt ét u"
    fixed = speech.apply_phonetic_fix(raw)
    assert "tín chỉ" in fixed
    assert "học bổng" in fixed
    assert "HSU" in fixed


def test_normalize_academic_text_expands_abbreviations_and_symbols():
    raw = "Theo QĐ-ĐHHS năm 2024-2025, SV ngành CNTT cần ĐTBCTL >= 2.5/4.0 và đạt >= 50% số TC."
    normalized = speech.normalize_academic_text(raw)
    assert "Quyết định Đại học Hoa Sen" in normalized
    assert "2024 đến 2025" in normalized
    assert "sinh viên" in normalized
    assert "Công nghệ thông tin" in normalized
    assert "điểm trung bình chung tích lũy" in normalized
    assert "lớn hơn hoặc bằng 2.5 chấm" not in normalized
    assert "2 chấm 5 trên 4 chấm 0" in normalized
    assert "50 phần trăm" in normalized
    assert "tín chỉ" in normalized


def test_preprocess_audio_vad_detects_silence_and_voiced_audio():
    silent_wav = make_wav_bytes(amplitude=0)
    _, is_speech_silent = speech.preprocess_audio_vad(silent_wav)
    assert is_speech_silent is False

    voiced_wav = make_wav_bytes(amplitude=14000)
    processed_bytes, is_speech_voiced = speech.preprocess_audio_vad(voiced_wav)
    assert is_speech_voiced is True
    assert len(processed_bytes) > 44


def test_validate_audio_bytes_rejects_empty_oversize_and_invalid_magic():
    with pytest.raises(ValueError):
        speech.validate_audio_bytes(b"", filename="empty.wav")

    oversized = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * (config.MAX_AUDIO_BYTES + 10)
    with pytest.raises(speech.AudioTooLargeError):
        speech.validate_audio_bytes(oversized, filename="huge.wav")

    with pytest.raises(speech.UnsupportedAudioFormatError):
        speech.validate_audio_bytes(b"%PDF-1.7 fake document content", filename="fake.wav")
