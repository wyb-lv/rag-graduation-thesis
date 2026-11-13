import pytest

from app.ingest import embed_service

TEXT = "Điều 1. Sinh viên phải nộp hồ sơ trước ngày 15 hằng tháng.\nĐiều 2. Hồ sơ nộp muộn không được xem xét."


@pytest.fixture
def calls(monkeypatch):
    log = []
    monkeypatch.setattr(embed_service, "_get_qdrant", lambda: object())
    monkeypatch.setattr(embed_service, "_ensure_collection", lambda client: None)
    monkeypatch.setattr(embed_service, "delete_document_vectors", lambda document_id: log.append(("delete", document_id)))
    return log


def test_a_failing_encode_keeps_the_old_vectors(calls, monkeypatch):
    def boom(texts):
        raise RuntimeError("CUDA error: unknown error")

    monkeypatch.setattr(embed_service.encoders, "encode_texts", boom)
    with pytest.raises(RuntimeError):
        embed_service.embed_document(document_id=7, text=TEXT, source="Quy chế thử")
    assert calls == []  # không xóa gì trước khi có vector mới
