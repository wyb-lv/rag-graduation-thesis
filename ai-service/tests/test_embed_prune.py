from types import SimpleNamespace

import pytest

from app.ingest import embed_consumer, embed_service


class FakeQdrant:
    """Hai trang scroll: tài liệu 1, 2 ở trang đầu, 2 (chunk khác) và 3 ở trang sau."""

    def __init__(self, exists=True):
        self.exists = exists
        self.pages = {None: ([1, 2], "next"), "next": ([2, 3], None)}

    def collection_exists(self, name):
        return self.exists

    def scroll(self, collection, limit, offset, with_payload, with_vectors):
        ids, nxt = self.pages[offset]
        return [SimpleNamespace(payload={"document_id": i}) for i in ids], nxt


@pytest.fixture
def deleted(monkeypatch):
    calls = []
    monkeypatch.setattr(embed_service, "_get_qdrant", lambda: FakeQdrant())
    monkeypatch.setattr(embed_service, "delete_document_vectors", lambda document_id: calls.append(document_id) or 1)
    return calls


def test_prune_deletes_vectors_of_documents_that_no_longer_exist(deleted):
    assert embed_service.prune_vectors({1, 3}) == [2]
    assert deleted == [2]


def test_prune_reads_every_page_of_the_collection(deleted):
    assert embed_service.prune_vectors({1}) == [2, 3]


def test_prune_keeps_everything_when_every_document_is_still_listed(deleted):
    assert embed_service.prune_vectors({1, 2, 3, 4}) == []
    assert deleted == []


def test_prune_on_a_collection_that_does_not_exist_yet_is_a_no_op(monkeypatch):
    monkeypatch.setattr(embed_service, "_get_qdrant", lambda: FakeQdrant(exists=False))
    assert embed_service.prune_vectors({1, 2}) == []


def test_prune_with_an_empty_list_deletes_nothing(deleted):
    # danh sách rỗng thường là DB bị reset; xóa theo sẽ mất bản duy nhất còn lại
    assert embed_service.prune_vectors(set()) == []
    assert deleted == []


def test_consumer_routes_a_prune_job_with_its_keep_list(monkeypatch):
    seen = []
    monkeypatch.setattr(embed_service, "prune_vectors", lambda keep: seen.append(keep))
    embed_consumer._process_job({"action": "PRUNE", "documentId": None, "keepIds": [4, 5]})
    assert seen == [{4, 5}]
