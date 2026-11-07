"""Chay voi Qdrant that o QDRANT_URL. Embed vao collection tam qua embed_service.embed_document (encoder la ban gia)."""
import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from app import bm25
from app import config as cfg
from app import qdrant_setup
from app.ingest import chunker, embed_service
from app.models import encoders
from app.rag import query_model as qm

pytestmark = pytest.mark.integration

DOC_V1 = [
    "[Quy chế học bổng – Điều 16. Học bổng tài năng]\n1. Sinh viên đạt giải olympic quốc tế được xét học bổng tài năng.",
    "[Quy chế học bổng – Điều 17. Hồ sơ]\n1. Hồ sơ gồm đơn đề nghị và bảng điểm.",
]
DOC_V2 = [
    "[Quy chế học bổng – Điều 16. Kỷ luật]\n1. Sinh viên bị kỷ luật cảnh cáo không được xét học bổng.",
    "[Quy chế học bổng – Điều 17. Hồ sơ]\n1. Hồ sơ nộp tại phòng công tác sinh viên.",
]


def unit(i):
    v = np.zeros(cfg.EMBED_DIM, dtype="float32")
    v[i % cfg.EMBED_DIM] = 1.0
    return v


@pytest.fixture(scope="module")
def client():
    c = QdrantClient(url=cfg.QDRANT_URL, timeout=60)  # mac dinh 5s: de timeout khi may dang tai nang
    try:
        c.get_collections()
    except Exception as e:
        pytest.skip(f"Khong ket noi duoc Qdrant {cfg.QDRANT_URL}: {e}")
    return c


@pytest.fixture
def target(client, monkeypatch):
    """Tro embed_service vao ten collection tam (chua tao), encoder gia; xoa collection khi xong."""
    name = f"test_embed_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(cfg, "QDRANT_COLLECTION", name)
    monkeypatch.setattr(embed_service, "_qdrant", client)
    monkeypatch.setattr(
        encoders, "encode_texts", lambda texts, batch_size=None: np.stack([unit(i) for i, _ in enumerate(texts)])
    )
    yield name
    assert name.startswith("test_")
    if client.collection_exists(name):
        client.delete_collection(name)


def use_chunks(monkeypatch, texts):
    monkeypatch.setattr(
        chunker, "chunk_document", lambda text, source: [chunker.Chunk(t, dieu=f"Điều {16 + i}") for i, t in enumerate(texts)]
    )


def points(client, name, document_id=None):
    flt = None
    if document_id is not None:
        flt = models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])
    found, _ = client.scroll(name, scroll_filter=flt, limit=100, with_payload=True, with_vectors=True)
    return found


def test_embed_creates_the_collection_with_all_legs_and_indexes_before_the_first_upsert(client, target, monkeypatch):
    order = []

    class Recording(QdrantClient):
        def create_collection(self, *a, **k):
            order.append("create_collection")
            return super().create_collection(*a, **k)

        def create_payload_index(self, collection_name, field_name, *a, **k):
            order.append(f"index:{field_name}")
            return super().create_payload_index(collection_name, field_name, *a, **k)

        def upsert(self, *a, **k):
            order.append("upsert")
            return super().upsert(*a, **k)

    monkeypatch.setattr(embed_service, "_qdrant", Recording(url=cfg.QDRANT_URL))
    use_chunks(monkeypatch, DOC_V1)
    result = embed_service.embed_document(document_id=7, text="x", source="Quy chế học bổng")

    assert result == {"upserted": 2, "paragraphs": 2}
    assert order[0] == "create_collection" and order[-1] == "upsert"
    assert {o for o in order if o.startswith("index:")} >= {"index:kho", "index:loai_van_ban", "index:khoa_tu"}
    assert qdrant_setup.vector_names(client, target) == {"dense", "bm25"}


@pytest.mark.parametrize("legs", [("dense", "bm25"), ("dense",), ("bm25",)])
def test_embed_writes_dense_and_bm25_regardless_of_hybrid_legs(client, target, monkeypatch, legs):
    monkeypatch.setattr(cfg, "HYBRID_LEGS", legs)
    use_chunks(monkeypatch, DOC_V1)
    embed_service.embed_document(document_id=7, text="x", source="Quy chế học bổng")

    found = points(client, target)
    assert len(found) == 2
    for p in found:
        assert set(p.vector) == {"dense", "bm25"}
        assert p.vector["bm25"].indices == bm25.bm25_doc_vector(p.payload["text"]).indices  # cùng hàm với lúc truy vấn
        assert p.payload["document_id"] == 7 and p.payload["loai_van_ban"] == "quy_che"


def test_bm25_leg_finds_the_embedded_text(client, target, monkeypatch):
    use_chunks(monkeypatch, DOC_V1)
    embed_service.embed_document(document_id=7, text="x", source="Quy chế học bổng")
    monkeypatch.setattr(qm, "_qdrant", client)
    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("bm25",))
    assert qm.retrieve("olympic")[0]["text"] == DOC_V1[0]


def test_reembedding_a_document_leaves_one_set_of_points_and_replaces_bm25(client, target, monkeypatch):
    monkeypatch.setattr(qm, "_qdrant", client)
    use_chunks(monkeypatch, DOC_V1)
    embed_service.embed_document(document_id=7, text="x", source="Quy chế học bổng")
    use_chunks(monkeypatch, ["[Quy chế khác – Điều 1]\n1. Nội dung của một tài liệu khác về thư viện."])
    embed_service.embed_document(document_id=8, text="x", source="Quy chế khác")
    assert client.count(target, exact=True).count == 3

    embed_service.embed_document(document_id=8, text="x", source="Quy chế khác")  # cùng nội dung: không nhân đôi
    assert client.count(target, exact=True).count == 3

    use_chunks(monkeypatch, DOC_V2)
    embed_service.embed_document(document_id=7, text="x", source="Quy chế học bổng")  # nội dung mới cho tài liệu 7
    assert client.count(target, exact=True).count == 3  # 2 (tài liệu 7) + 1 (tài liệu 8), không dư điểm cũ
    assert sorted(p.payload["text"] for p in points(client, target, 7)) == sorted(DOC_V2)
    assert len(points(client, target, 8)) == 1  # tài liệu khác không bị đụng tới

    monkeypatch.setattr(cfg, "HYBRID_LEGS", ("bm25",))
    assert qm.retrieve("olympic") == []  # bm25 của nội dung cũ đã bị thay
    assert "kỷ luật" in qm.retrieve("kỷ luật")[0]["text"]


def test_chunk_without_tokens_does_not_break_the_upsert(client, target, monkeypatch):
    use_chunks(monkeypatch, [DOC_V1[0], "?!... —"])
    result = embed_service.embed_document(document_id=7, text="x", source="Quy chế học bổng")
    assert result == {"upserted": 2, "paragraphs": 2}
    by_text = {p.payload["text"]: p for p in points(client, target)}
    assert set(by_text["?!... —"].vector) == {"dense"}  # không có bm25, vector dense vẫn ghi
    assert set(by_text[DOC_V1[0]].vector) == {"dense", "bm25"}
