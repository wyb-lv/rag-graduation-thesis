import io
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.api import app


@pytest.fixture
def client():
    return TestClient(app)


def _dummy_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((np.zeros(16000, dtype=np.int16)).tobytes())
    return buf.getvalue()


# ── 1. Test POST /tts/split ──────────────────────────────────────────
def test_tts_split_endpoint(client):
    payload = {
        "text": "Thí sinh có GPA 3.5 sẽ được ưu tiên. Xem Điều 10, Khoản 2."
    }
    resp = client.post("/tts/split", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "sentences" in data
    assert len(data["sentences"]) == 2
    assert "G P A 3 chấm 5" in data["sentences"][0]


# ── 2. Test POST /tts ────────────────────────────────────────────────
@patch("app.speech.synthesize_speech_stream")
def test_tts_endpoint(mock_synth, client):
    async def mock_generator(*args, **kwargs):
        yield b"chunk_1"
        yield b"chunk_2"

    mock_synth.return_value = mock_generator()

    resp = client.post("/tts", json={"text": "Xin chào thí sinh", "optimized": True})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/mpeg")
    assert resp.content == b"chunk_1chunk_2"


# ── 3. Test POST /stt ────────────────────────────────────────────────
@patch("app.speech.transcribe_speech")
def test_stt_endpoint_success(mock_transcribe, client):
    mock_transcribe.return_value = {
        "is_speech": True,
        "raw_text": "To hop a khong mot",
        "corrected_text": "A01",
        "vad_trimmed_ms": 120.0,
        "latency_ms": 450.0,
    }

    files = {"file": ("audio.wav", _dummy_wav(), "audio/wav")}
    resp = client.post("/stt?optimized=true", files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "optimized"
    assert data["is_speech"] is True
    assert data["corrected_text"] == "A01"
    assert data["latency_ms"] == 450.0


def test_stt_endpoint_exceed_max_size(client):
    # Tạo payload vượt quá giới hạn 5MB
    large_audio = b"0" * (5 * 1024 * 1024 + 100)
    files = {"file": ("huge.wav", large_audio, "audio/wav")}
    resp = client.post("/stt", files=files)
    assert resp.status_code == 413
    assert "vượt quá giới hạn" in resp.json()["detail"]
