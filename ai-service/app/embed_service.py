"""Module upsert (embed + index) đoạn văn bản vào Qdrant.

Được gọi từ:
  - API endpoint POST /embed  (FE hoặc admin trigger thủ công)
  - RabbitMQ consumer: nhận job từ file-service sau khi OCR xong

Mỗi tài liệu được chia thành các đoạn (paragraph), encode bằng BGE-M3 (dense + sparse)
và upsert vào Qdrant collection `quy_dinh_hybrid`.
"""
import logging
import re
import uuid
from typing import Optional

from qdrant_client import QdrantClient, models

from . import config as cfg
from . import encoders

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
        return
    log.info("Tạo collection '%s'...", cfg.QDRANT_COLLECTION)
    client.create_collection(
        collection_name=cfg.QDRANT_COLLECTION,
        vectors_config={
            cfg.DENSE_VECTOR_NAME: models.VectorParams(
                size=cfg.EMBED_DIM,
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=cfg.HNSW_M, ef_construct=cfg.HNSW_EF_CONSTRUCT),
                on_disk=False,
            )
        },
        sparse_vectors_config={
            cfg.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=cfg.INDEXING_THRESHOLD),
    )
    # Tạo payload index để filter nhanh
    for field in (cfg.FIELD_KHO, cfg.FIELD_DOI_TUONG):
        client.create_payload_index(
            cfg.QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    for field in (cfg.FIELD_HIEU_LUC_TU, cfg.FIELD_HIEU_LUC_DEN):
        client.create_payload_index(
            cfg.QDRANT_COLLECTION,
            field_name=field,
            field_schema=models.PayloadSchemaType.DATETIME,
        )


def _split_paragraphs(text: str, min_chars: int = 50) -> list[str]:
    """Tách văn bản thành các đoạn dựa trên dòng trống hoặc dấu xuống dòng kép.
    Đoạn quá ngắn (< min_chars) được ghép vào đoạn trước.
    """
    raw = re.split(r"\n{2,}", text.strip())
    paragraphs = []
    for p in raw:
        p = p.strip().replace("\n", " ")
        if not p:
            continue
        if paragraphs and len(p) < min_chars:
            paragraphs[-1] += " " + p
        else:
            paragraphs.append(p)
    return paragraphs


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

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return {"upserted": 0, "paragraphs": 0}

    log.info("Bắt đầu embed tài liệu id=%d source='%s' (%d đoạn)", document_id, source, len(paragraphs))

    client = _get_qdrant()
    _ensure_collection(client)

    # Xóa các vector cũ của tài liệu này (idempotent re-embed)
    client.delete(
        collection_name=cfg.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
            )
        ),
    )

    total_upserted = 0
    for batch_start in range(0, len(paragraphs), cfg.UPSERT_BATCH_SIZE):
        batch = paragraphs[batch_start: batch_start + cfg.UPSERT_BATCH_SIZE]
        dense_vecs, lexical_weights = encoders.encode_texts(batch)

        points = []
        for i, (para, dvec, lex) in enumerate(zip(batch, dense_vecs, lexical_weights)):
            payload = {
                "document_id": document_id,
                "source": source,
                "text": para,
                "paragraph_index": batch_start + i,
            }
            if kho:
                payload[cfg.FIELD_KHO] = kho
            if hieu_luc_tu:
                payload[cfg.FIELD_HIEU_LUC_TU] = hieu_luc_tu
            if hieu_luc_den:
                payload[cfg.FIELD_HIEU_LUC_DEN] = hieu_luc_den

            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        cfg.DENSE_VECTOR_NAME: dvec.tolist(),
                        cfg.SPARSE_VECTOR_NAME: encoders.to_sparse_vector(lex),
                    },
                    payload=payload,
                )
            )

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
