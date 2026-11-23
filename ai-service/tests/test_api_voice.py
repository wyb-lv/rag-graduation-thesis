import json
import uuid
from types import SimpleNamespace
from fastapi.testclient import TestClient

import app.api as api
import app.config as cfg
import app.rag.query_model as qm
from app import session
from app.router import router_llm
from app.speech import stt, tts
try:
    from test_speech import make_wav_bytes
except ImportError:
    from tests.test_speech import make_wav_bytes


client = TestClient(api.app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_tts_endpoint_returns_audio_mpeg(monkeypatch):
    async def fake_stream_speech_chunks(text: str, voice: str = "", rate: str = ""):
        assert "Quyết định Đại học Hoa Sen" in text
        yield b"ID3-FAKE-MP3-BYTES"

    monkeypatch.setattr(tts, "stream_speech_chunks", fake_stream_speech_chunks)

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

    monkeypatch.setattr(stt, "get_groq_client", lambda: FakeGroqClient())

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

    monkeypatch.setattr(stt, "get_groq_client", lambda: FakeGroqClient())
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


def _parse_sse(resp):
    return [
        block[len("data: "):]
        if block.startswith("data: [DONE]")
        else json.loads(block[len("data: "):])
        for block in resp.text.split("\n\n")
        if block.startswith("data: ")
    ]


def _fake_groq_stt(monkeypatch, transcripts):
    queue = list(transcripts)

    class FakeTranscriptions:
        def create(self, **kwargs):
            assert queue, "STT duoc goi ngoai du kien"
            return SimpleNamespace(text=queue.pop(0))

    class FakeAudio:
        transcriptions = FakeTranscriptions()

    class FakeGroqClient:
        audio = FakeAudio()

    monkeypatch.setattr(stt, "get_groq_client", lambda: FakeGroqClient())


def test_call_greeting_returns_session_id_and_listen_action(monkeypatch):
    async def fake_stream_speech_chunks(text: str, voice: str = "", rate: str = ""):
        assert text == cfg.CALL_GREETING_TEXT
        yield b"ID3-GREETING"

    monkeypatch.setattr(tts, "stream_speech_chunks", fake_stream_speech_chunks)

    resp = client.post("/call/greeting")
    assert resp.status_code == 200
    assert resp.headers["x-call-action"] == "listen"
    sid = resp.headers["x-session-id"]
    assert str(uuid.UUID(sid, version=4)) == sid
    expose = resp.headers.get("access-control-expose-headers", "")
    assert "X-Call-Action" in expose and "X-Session-Id" in expose
    assert resp.content == b"ID3-GREETING"

    assert client.post("/call/greeting", data={"session_id": "invalid"}).status_code == 400


def test_call_turn_silence_returns_204_with_session_id():
    silent_wav = make_wav_bytes(amplitude=0)
    sid = "11111111-1111-4111-8111-111111111111"
    resp = client.post(
        "/call/turn",
        files={"file": ("silence.wav", silent_wav, "audio/wav")},
        data={"session_id": sid.upper()},
    )
    assert resp.status_code == 204
    assert resp.headers["x-call-action"] == "listen"
    assert resp.headers["x-session-id"] == sid
    expose = resp.headers.get("access-control-expose-headers", "")
    assert "X-Call-Action" in expose and "X-Session-Id" in expose


def test_call_turn_preserves_session_and_resolves_followup(monkeypatch):
    monkeypatch.setattr(session, "_store", None)
    _fake_groq_stt(monkeypatch, ["Học bổng thế nào?", "Vậy giỏi được nhiều hơn khá đúng không?"])

    router_calls = []
    standalone_queries = [
        "Điều kiện và mức học bổng HSU",
        "Mức học bổng loại giỏi so với loại khá tại HSU",
    ]

    async def fake_resolve(question, history, session_facts, previous_was_clarify=False):
        router_calls.append((question, list(history), dict(session_facts), previous_was_clarify))
        idx = len(router_calls) - 1
        return router_llm.Routed(
            decision=router_llm.RouteDecision(
                intent="rag",
                confidence=0.95,
                standalone_query=standalone_queries[idx],
                missing=None,
            )
        )

    find_calls = []

    def fake_find_hits(question, **kwargs):
        find_calls.append((question, kwargs))
        return [
            {
                "source": "Quy chế học bổng HSU",
                "text": "Học bổng loại giỏi cao hơn loại khá.",
                "rerank_score": 0.92,
                "retrieval_score": 0.85,
                "payload": {},
            }
        ]

    stream_calls = []

    def fake_generate_answer_stream(question, hits, **kwargs):
        stream_calls.append((question, kwargs))
        if len(stream_calls) == 1:
            return iter(["Học bổng gồm ", "loại khá và loại giỏi."])
        return iter(["Đúng rồi, loại giỏi cao hơn loại khá."])

    async def fake_stream_speech_from_text_iter(text_iter, voice: str = "", rate: str = ""):
        parts = list(text_iter)
        assert parts
        yield b"ID3-CALL-MP3"

    monkeypatch.setattr(router_llm, "resolve", fake_resolve)
    monkeypatch.setattr(qm, "find_hits", fake_find_hits)
    monkeypatch.setattr(qm, "generate_answer_stream", fake_generate_answer_stream)
    monkeypatch.setattr(tts, "stream_speech_from_text_iter", fake_stream_speech_from_text_iter)

    voiced_wav = make_wav_bytes(amplitude=14000)
    resp1 = client.post(
        "/call/turn",
        files={"file": ("turn1.wav", voiced_wav, "audio/wav")},
    )
    assert resp1.status_code == 200
    assert resp1.headers["x-call-action"] == "listen"
    sid = resp1.headers["x-session-id"]
    assert str(uuid.UUID(sid, version=4)) == sid
    assert resp1.content == b"ID3-CALL-MP3"

    resp2 = client.post(
        "/call/turn",
        files={"file": ("turn2.wav", voiced_wav, "audio/wav")},
        data={"session_id": sid},
    )
    assert resp2.status_code == 200
    assert resp2.headers["x-call-action"] == "listen"
    assert resp2.headers["x-session-id"] == sid
    assert resp2.content == b"ID3-CALL-MP3"

    assert len(router_calls) == 2
    assert router_calls[0][1] == []
    turn2_history = router_calls[1][1]
    assert len(turn2_history) == 2
    assert turn2_history[0] == ("user", "Học bổng thế nào?")
    assert turn2_history[1] == ("assistant", "Học bổng gồm loại khá và loại giỏi.")
    assert find_calls[0][0] == standalone_queries[0]
    assert find_calls[1][0] == standalone_queries[1]
    assert stream_calls[0][1] == {"voice_mode": True}
    assert stream_calls[1][1] == {"history": turn2_history, "voice_mode": True}


def test_call_turn_returns_hangup_on_goodbye(monkeypatch):
    monkeypatch.setattr(session, "_store", None)
    _fake_groq_stt(monkeypatch, ["Cảm ơn bạn, tạm biệt nhé"])

    async def fake_resolve(question, history, session_facts, previous_was_clarify=False):
        return router_llm.Routed(
            decision=router_llm.RouteDecision(
                intent="goodbye",
                confidence=0.98,
                standalone_query=None,
                missing=None,
            )
        )

    spoken_texts = []

    async def fake_stream_speech_chunks(text: str, voice: str = "", rate: str = ""):
        spoken_texts.append(text)
        yield b"ID3-GOODBYE-MP3"

    monkeypatch.setattr(router_llm, "resolve", fake_resolve)
    monkeypatch.setattr(tts, "stream_speech_chunks", fake_stream_speech_chunks)
    monkeypatch.setattr(
        qm,
        "find_hits",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("khong duoc goi RAG khi goodbye")),
    )

    voiced_wav = make_wav_bytes(amplitude=14000)
    resp = client.post(
        "/call/turn",
        files={"file": ("bye.wav", voiced_wav, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["x-call-action"] == "hangup"
    sid = resp.headers["x-session-id"]
    assert str(uuid.UUID(sid, version=4)) == sid
    expose = resp.headers.get("access-control-expose-headers", "")
    assert "X-Call-Action" in expose and "X-Session-Id" in expose
    assert resp.content == b"ID3-GOODBYE-MP3"
    assert spoken_texts == [cfg.GUARD_REPLY_GOODBYE]


def test_ask_voice_stream_includes_session_id_and_preserves_context(monkeypatch):
    monkeypatch.setattr(session, "_store", None)
    _fake_groq_stt(monkeypatch, ["Học bổng thế nào?", "Vậy loại giỏi thì sao?"])

    router_calls = []
    standalone_queries = ["Điều kiện học bổng HSU", "Mức học bổng loại giỏi HSU"]

    async def fake_resolve(question, history, session_facts, previous_was_clarify=False):
        router_calls.append((question, list(history), dict(session_facts), previous_was_clarify))
        idx = len(router_calls) - 1
        return router_llm.Routed(
            decision=router_llm.RouteDecision(
                intent="rag",
                confidence=0.95,
                standalone_query=standalone_queries[idx],
                missing=None,
            )
        )

    find_calls = []

    def fake_find_hits(question, **kwargs):
        find_calls.append((question, kwargs))
        return [
            {
                "source": "Quy chế học bổng HSU",
                "text": "Học bổng gồm loại khá và loại giỏi.",
                "rerank_score": 0.91,
                "retrieval_score": 0.84,
                "payload": {},
            }
        ]

    stream_calls = []

    def fake_generate_answer_stream(question, hits, **kwargs):
        stream_calls.append((question, kwargs))
        if len(stream_calls) == 1:
            return iter(["Học bổng ", "có nhiều mức."])
        return iter(["Loại giỏi ", "cao hơn loại khá."])

    monkeypatch.setattr(router_llm, "resolve", fake_resolve)
    monkeypatch.setattr(qm, "find_hits", fake_find_hits)
    monkeypatch.setattr(qm, "generate_answer_stream", fake_generate_answer_stream)

    voiced_wav = make_wav_bytes(amplitude=14000)
    resp1 = client.post(
        "/ask/voice/stream",
        files={"file": ("q1.wav", voiced_wav, "audio/wav")},
    )
    assert resp1.status_code == 200
    events1 = _parse_sse(resp1)
    assert events1[0] == {"event": "transcription", "text": "Học bổng thế nào?"}
    sources1 = events1[1]
    assert sources1["event"] == "sources" and sources1["in_scope"] is True
    sid = sources1["session_id"]
    assert str(uuid.UUID(sid, version=4)) == sid
    assert events1[-1] == "[DONE]"

    resp2 = client.post(
        "/ask/voice/stream",
        files={"file": ("q2.wav", voiced_wav, "audio/wav")},
        data={"session_id": sid},
    )
    assert resp2.status_code == 200
    events2 = _parse_sse(resp2)
    assert events2[0] == {"event": "transcription", "text": "Vậy loại giỏi thì sao?"}
    assert events2[1]["event"] == "sources" and events2[1]["session_id"] == sid
    assert events2[-1] == "[DONE]"

    assert len(router_calls) == 2
    assert router_calls[0][1] == []
    turn2_history = router_calls[1][1]
    assert turn2_history == [
        ("user", "Học bổng thế nào?"),
        ("assistant", "Học bổng có nhiều mức."),
    ]
    assert find_calls[1][0] == standalone_queries[1]
    assert stream_calls[0][1] == {}
    assert stream_calls[1][1] == {"history": turn2_history}
