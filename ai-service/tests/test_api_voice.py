from types import SimpleNamespace
from fastapi.testclient import TestClient

import app.api as api
import app.speech as speech
import app.rag.query_model as qm
import app.config as cfg
from test_speech import make_wav_bytes  # tests/ khong phai package (pytest them no vao sys.path); `tests` tren sys.path co the la goi khac


client = TestClient(api.app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_tts_endpoint_returns_audio_mpeg(monkeypatch):
    async def fake_synthesize_speech_bytes(text: str, voice: str = "", rate: str = "") -> bytes:
        assert "Quyết định Đại học Hoa Sen" in text
        return b"ID3-FAKE-MP3-BYTES"

    monkeypatch.setattr(speech, "synthesize_speech_bytes", fake_synthesize_speech_bytes)

    resp = client.post(
        "/tts",
        json={"text": "Theo QĐ-ĐHHS, SV cần đóng học phí đúng hạn."},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    assert resp.content.startswith(b"ID3-FAKE-MP3-BYTES")


def test_tts_empty_or_whitespace_returns_400_or_422():
    resp = client.post("/tts", json={"text": "   "})
    assert resp.status_code == 400


def test_stt_endpoint_with_silence_skips_groq():
    silent_wav = make_wav_bytes(amplitude=0)
    resp = client.post(
        "/stt",
        files={"file": ("silence.wav", silent_wav, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_speech"] is False
    assert data["text"] == ""


def test_stt_endpoint_rejects_non_audio_file_with_415():
    resp = client.post(
        "/stt",
        files={"file": ("malware.exe", b"MZ\x90\x00\x03\x00\x00\x00not-an-audio-file", "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_stt_endpoint_rejects_oversized_file_with_413(monkeypatch):
    monkeypatch.setattr(cfg, "MAX_AUDIO_BYTES", 1024)
    big_wav = make_wav_bytes(duration_s=0.2, amplitude=12000)
    assert len(big_wav) > 1024

    resp = client.post(
        "/stt",
        files={"file": ("large.wav", big_wav, "audio/wav")},
    )
    assert resp.status_code == 413


def test_stt_endpoint_with_voiced_wav_applies_phonetic_fix(monkeypatch):
    class FakeTranscriptions:
        def create(self, **kwargs):
            return SimpleNamespace(text="Cho em hỏi điều kiện xét học bỗng và tính chỉ ở hắt ét u")

    class FakeAudio:
        transcriptions = FakeTranscriptions()

    class FakeGroqClient:
        audio = FakeAudio()

    monkeypatch.setattr(speech, "get_groq_client", lambda: FakeGroqClient())

    voiced_wav = make_wav_bytes(amplitude=14000)
    resp = client.post(
        "/stt",
        files={"file": ("voice.wav", voiced_wav, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_speech"] is True
    assert "học bổng" in data["text"]
    assert "tín chỉ" in data["text"]
    assert "HSU" in data["text"]


def test_ask_voice_stream_endpoint_streams_sse_from_audio(monkeypatch):
    class FakeTranscriptions:
        def create(self, **kwargs):
            return SimpleNamespace(text="Điều kiện nhận học bỗng tại hắt ét u")

    class FakeAudio:
        transcriptions = FakeTranscriptions()

    class FakeGroqClient:
        audio = FakeAudio()

    monkeypatch.setattr(speech, "get_groq_client", lambda: FakeGroqClient())
    monkeypatch.setattr(
        qm,
        "find_hits",
        lambda question, **kwargs: [
            {
                "source": "Quy chế học vụ HSU",
                "text": "Sinh viên HSU có GPA >= 3.2 được xét học bổng.",
                "rerank_score": 0.91,
                "retrieval_score": 0.85,
                "payload": {},
            }
        ],
    )
    monkeypatch.setattr(
        qm,
        "generate_answer_stream",
        lambda question, hits: iter(["Sinh viên HSU ", "cần GPA từ 3.2 trở lên."]),
    )

    voiced_wav = make_wav_bytes(amplitude=14000)
    resp = client.post(
        "/ask/voice/stream",
        files={"file": ("question.wav", voiced_wav, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert '"event": "transcription"' in body
    assert "Điều kiện nhận học bổng tại HSU" in body
    assert '"event": "sources"' in body
    assert '"event": "delta"' in body
    assert "data: [DONE]" in body
