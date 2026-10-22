import io
import wave
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app import speech


# ── 1. Test Sửa lỗi đồng âm tiếng Việt (Phonetic Fix) ────────────────────────
def test_phonetic_fix_admission_codes():
    assert speech.apply_phonetic_fix("xét tuyển tổ hợp a không không") == "xét tuyển A00"
    assert speech.apply_phonetic_fix("tổ hợp a không một") == "A01"
    assert speech.apply_phonetic_fix("tổ hợp b không không") == "B00"
    assert speech.apply_phonetic_fix("tổ hợp d không một") == "D01"
    assert speech.apply_phonetic_fix("tổ hợp c không không") == "C00"


def test_phonetic_fix_academic_terms():
    assert speech.apply_phonetic_fix("điểm gờ pê a") == "điểm GPA"
    assert speech.apply_phonetic_fix("chứng chỉ ai eo") == "chứng chỉ IELTS"
    assert speech.apply_phonetic_fix("chứng chỉ tô ích") == "chứng chỉ TOEIC"
    assert speech.apply_phonetic_fix("bị điểm ép") == "bị điểm F"
    assert speech.apply_phonetic_fix("trường đại học hoa xem") == "trường Hoa Sen"


def test_phonetic_fix_disabled():
    raw = "tổ hợp a không một"
    assert speech.apply_phonetic_fix(raw, enable_phonetic_fix=False) == raw


# ── 2. Test Chuẩn hóa văn bản đọc (Text Normalization) ────────────────────────
def test_normalize_academic_text():
    # Điểm thập phân
    assert "G P A 3 chấm 50" in speech.normalize_academic_text("GPA 3.50")
    # Phân số tín chỉ
    assert "15 trên 30 tín chỉ" in speech.normalize_academic_text("15/30 tín chỉ")
    # Viết tắt học vụ
    assert "Trung học phổ thông" in speech.normalize_academic_text("kỳ thi THPT")
    # Nhãn trích dẫn tài liệu bị loại bỏ
    norm = speech.normalize_academic_text("Xem chi tiết [quy_che.pdf:12].")
    assert "[quy_che.pdf:12]" not in norm
    assert "Xem chi tiết" in norm


# ── 3. Test Tách câu cho TTS Prefetching ──────────────────────────────────────
def test_split_into_sentences():
    text = "Câu thứ nhất! Câu thứ hai? Câu thứ ba. Và câu cuối cùng."
    sentences = speech.split_into_sentences(text)
    assert len(sentences) == 4
    assert sentences[0] == "Câu thứ nhất!"
    assert sentences[1] == "Câu thứ hai?"
    assert sentences[2] == "Câu thứ ba."
    assert sentences[3] == "Và câu cuối cùng."


# ── 4. Test Thuật toán VAD (Voice Activity Detection) ─────────────────────────
def _create_wav(duration_sec: float, is_silence: bool = False, sample_rate: int = 16000) -> bytes:
    n_samples = int(duration_sec * sample_rate)
    if is_silence:
        samples = np.zeros(n_samples, dtype=np.int16)
    else:
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def test_vad_active_speech():
    active_wav = _create_wav(1.5, is_silence=False)
    processed, is_speech, trimmed_ms = speech.preprocess_audio_vad(active_wav)
    assert is_speech is True
    assert len(processed) > 0


def test_vad_silence_detection():
    silent_wav = _create_wav(1.5, is_silence=True)
    processed, is_speech, trimmed_ms = speech.preprocess_audio_vad(silent_wav)
    assert is_speech is False
    assert len(processed) == 0
    assert trimmed_ms >= 1400.0


def test_vad_non_wav_bypass():
    mp3_fake = b"ID3\x03\x00\x00\x00fake_mp3_content"
    processed, is_speech, trimmed_ms = speech.preprocess_audio_vad(mp3_fake)
    assert processed == mp3_fake
    assert is_speech is True
    assert trimmed_ms == 0.0


# ── 5. Test Pipeline Transcribe Speech với Mock Groq ──────────────────────────
def test_transcribe_speech_silence_short_circuit():
    silent_wav = _create_wav(1.0, is_silence=True)
    result = speech.transcribe_speech(silent_wav, "silent.wav")
    assert result["is_speech"] is False
    assert result["raw_text"] == ""
    assert result["corrected_text"] == ""


@patch("app.groq_client.get_client")
def test_transcribe_speech_with_mock_client(mock_get_client):
    mock_client = MagicMock()
    mock_transcription = MagicMock()
    mock_transcription.text = "Em muốn thi tổ hợp a không một"
    mock_client.audio.transcriptions.create.return_value = mock_transcription
    mock_get_client.return_value = mock_client

    active_wav = _create_wav(1.0, is_silence=False)
    result = speech.transcribe_speech(active_wav, "active.wav", enable_phonetic_fix=True)

    assert result["is_speech"] is True
    assert result["raw_text"] == "Em muốn thi tổ hợp a không một"
    assert result["corrected_text"] == "Em muốn thi A01"
    assert "latency_ms" in result
