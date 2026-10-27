"""Tạo collection Qdrant và kiểm tra các nhánh hybrid có trong collection.

Chỉ cần qdrant_client + config (không nạp model).
"""
import logging
from typing import Iterable, Optional

from qdrant_client import QdrantClient, models

from . import config as cfg

log = logging.getLogger("uvicorn.error")


def create_collection(client: QdrantClient, name: str):
    """Tạo collection mới đủ dense + sparse và payload index. Gọi khi collection chưa tồn tại."""
    client.create_collection(
        collection_name=name,
        vectors_config={
            cfg.DENSE_VECTOR_NAME: models.VectorParams(
                size=cfg.EMBED_DIM, 
                distance=models.Distance.COSINE,
                hnsw_config=models.HnswConfigDiff(m=cfg.HNSW_M, ef_construct=cfg.HNSW_EF_CONSTRUCT),
                on_disk=False,
            )
        },
        sparse_vectors_config={
            # Trọng số học được của BGE-M3: không dùng IDF
            cfg.SPARSE_VECTOR_NAME: models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False)),
        },
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=cfg.INDEXING_THRESHOLD),
    )
    # Tạo payload index để filter nhanh
    for field in (cfg.FIELD_KHO, cfg.FIELD_DOI_TUONG, cfg.FIELD_LOAI_VAN_BAN):
        client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)
    for field in (cfg.FIELD_HIEU_LUC_TU, cfg.FIELD_HIEU_LUC_DEN):
        client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.DATETIME)
    for field in (cfg.FIELD_KHOA_TU, cfg.FIELD_KHOA_DEN):
        client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.INTEGER)


def vector_names(client: QdrantClient, name: str) -> set[str]:
    """Tên các vector (dense và sparse) mà collection khai báo."""
    params = client.get_collection(name).config.params
    dense = set(params.vectors) if isinstance(params.vectors, dict) else set()  # vector không tên thì không có khóa
    return dense | set(params.sparse_vectors or {})


def missing_legs(client: QdrantClient, name: str, legs: Iterable[str]) -> list[str]:
    present = vector_names(client, name)
    return [leg for leg in legs if cfg.HYBRID_LEG_VECTORS[leg] not in present]


def validate_legs(client: QdrantClient, name: Optional[str] = None, legs: Optional[Iterable[str]] = None):
    """Lúc khởi động: mọi nhánh trong HYBRID_LEGS phải có vector tương ứng trong collection, nếu không thì báo lỗi.

    Qdrant chưa sẵn sàng hoặc collection chưa tồn tại thì chỉ cảnh báo/ghi log, không chặn khởi động
    (collection sẽ được tạo đủ dense/sparse ở lần embed đầu tiên).
    """
    name = name or cfg.QDRANT_COLLECTION
    legs = tuple(legs or cfg.HYBRID_LEGS)
    try:
        if not client.collection_exists(name):
            log.info("Collection '%s' chưa tồn tại, sẽ tạo ở lần embed đầu tiên. Nhánh hybrid: %s", name, ", ".join(legs))
            return
        missing = missing_legs(client, name, legs)
    except Exception as e:
        log.warning("Không kiểm tra được nhánh hybrid của collection '%s' (Qdrant %s): %s", name, cfg.QDRANT_URL, e)
        return
    if missing:
        raise RuntimeError(
            f"Collection '{name}' thiếu vector cho nhánh {', '.join(missing)} "
            f"(HYBRID_LEGS={','.join(legs)}). Embed lại tài liệu vào collection tạo bằng qdrant_setup.create_collection, "
            f"hoặc bỏ nhánh đó khỏi HYBRID_LEGS."
        )
    log.info("Nhánh hybrid đang dùng: %s (collection '%s')", ", ".join(legs), name)
