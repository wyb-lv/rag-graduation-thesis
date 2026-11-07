"""Module upsert (embed + index) đoạn văn bản vào Qdrant.

Được gọi từ:
  - API endpoint POST /embed  (FE hoặc admin trigger thủ công)
  - RabbitMQ consumer: nhận job từ file-service sau khi OCR xong

Mỗi tài liệu được làm sạch và chia chunk theo Chương/Điều/khoản (xem chunker.py), encode bằng BGE-M3 (dense, sentence-transformers) và tính vector BM25 (bm25.py),
rồi upsert vào Qdrant collection `quy_dinh_hybrid_v2`.
"""
import logging
import uuid
from typing import Optional

from qdrant_client import QdrantClient, models

from .. import bm25
from .. import config as cfg
from .. import qdrant_setup
from ..models import encoders
from . import chunker

log = logging.getLogger("uvicorn.error")

# Qdrant client dùng chung (lazy init)
_qdrant: Optional[QdrantClient] = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=cfg.QDRANT_URL)
    return _qdrant


def _ensure_collection(client: QdrantClient):
    """Tạo collection nếu chưa tồn tại."""
    existing = {c.name for c in client.get_collections().collections}
    if cfg.QDRANT_COLLECTION in existing:
        for field in (cfg.FIELD_KHOA_TU, cfg.FIELD_KHOA_DEN):  # idempotent; bổ sung cho collection tạo trước đây
            client.create_payload_index(cfg.QDRANT_COLLECTION, field_name=field, field_schema=models.PayloadSchemaType.INTEGER)
        return
    log.info("Tạo collection '%s'...", cfg.QDRANT_COLLECTION)
    qdrant_setup.create_collection(client, cfg.QDRANT_COLLECTION)


def embed_document(
    *,
    document_id: int,
    text: str,
    source: str,
    kho: Optional[str] = None,
    hieu_luc_tu: Optional[str] = None,
    hieu_luc_den: Optional[str] = None,
) -> dict:
    """Embed toàn bộ văn bản của một tài liệu và upsert vào Qdrant.

    Args:
        document_id: ID tài liệu trong file-service (dùng để idempotent delete-then-insert).
        text: Toàn bộ văn bản đã trích xuất (từ OCR hoặc parse).
        source: Tên tài liệu hiển thị cho người dùng (VD: "Quy chế đào tạo 2024").
        kho: Nhãn kho (VD: "tuyen_sinh", "dao_tao").
        hieu_luc_tu / hieu_luc_den: Ngày hiệu lực ISO-8601 (VD: "2024-01-01T00:00:00").

    Returns:
        {"upserted": <số đoạn đã index>, "paragraphs": <tổng số đoạn>}
    """
    if not text or not text.strip():
        log.warning("embed_document: tài liệu %d không có văn bản, bỏ qua.", document_id)
        return {"upserted": 0, "paragraphs": 0}

    chunks = chunker.chunk_document(text, source)
    paragraphs = [c.text for c in chunks]
    meta = chunker.infer_metadata(source)
    if not paragraphs:
        return {"upserted": 0, "paragraphs": 0}

    log.info("Bắt đầu embed tài liệu id=%d source='%s' (%d đoạn)", document_id, source, len(paragraphs))

    client = _get_qdrant()
    _ensure_collection(client)
    # Embed lại (sau OCR lại) phải thay vector cũ, không được chồng lên
    delete_document_vectors(document_id)
    total_upserted = 0
    for batch_start in range(0, len(paragraphs), cfg.UPSERT_BATCH_SIZE):
        batch = paragraphs[batch_start: batch_start + cfg.UPSERT_BATCH_SIZE]
        dense_vecs = encoders.encode_texts(batch)

        points = []
        for i, (para, dvec) in enumerate(zip(batch, dense_vecs)):
            payload = {
                "document_id": document_id,
                "source": source,
                "text": para,
                "paragraph_index": batch_start + i,
                **meta,
            }
            if chunks[batch_start + i].dieu:
                payload["dieu"] = chunks[batch_start + i].dieu
            if chunks[batch_start + i].chuong:
                payload["chuong"] = chunks[batch_start + i].chuong
            if kho:
                payload[cfg.FIELD_KHO] = kho
            if hieu_luc_tu:
                payload[cfg.FIELD_HIEU_LUC_TU] = hieu_luc_tu
            if hieu_luc_den:
                payload[cfg.FIELD_HIEU_LUC_DEN] = hieu_luc_den

            vector = {cfg.DENSE_VECTOR_NAME: dvec.tolist(), **bm25.doc_vectors(para)}

            points.append(models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))

        client.upsert(collection_name=cfg.QDRANT_COLLECTION, points=points)
        total_upserted += len(points)
        log.info("  Upsert batch %d-%d xong (%d/%d đoạn)",
                 batch_start, batch_start + len(batch) - 1, total_upserted, len(paragraphs))

    log.info("Embed tài liệu %d hoàn tất: %d/%d đoạn đã index.", document_id, total_upserted, len(paragraphs))
    return {"upserted": total_upserted, "paragraphs": len(paragraphs)}


def delete_document_vectors(document_id: int) -> int:
    """Xóa toàn bộ vector của tài liệu khỏi Qdrant. Trả về số điểm đã xóa (ước tính)."""
    client = _get_qdrant()
    try:
        result = client.delete(
            collection_name=cfg.QDRANT_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
                )
            ),
        )
        log.info("Đã xóa vector tài liệu %d khỏi Qdrant (status=%s)", document_id, result.status)
        return 1
    except Exception as e:
        log.error("Lỗi khi xóa vector tài liệu %d: %s", document_id, e)
        return 0
