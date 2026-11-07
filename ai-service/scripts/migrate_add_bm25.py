"""Chuyển sang nhánh BM25: sao chép collection cũ `quy_dinh_hybrid` sang collection mới (dense + bm25).

Chạy (cần Qdrant; không cần GPU và không nạp model vì vector dense được chép nguyên, bm25 tính từ payload `text`):
    python -m scripts.migrate_add_bm25 [--source quy_dinh_hybrid] [--target quy_dinh_hybrid_v2] [--tokenizer syllable_bigram]

Vector dense của sentence-transformers và của FlagEmbedding cùng một model BGE-M3 (pooling CLS + chuẩn hóa L2) nên dùng
lẫn nhau được, không phải embed lại. Chỉ vector dense được chép; mọi vector khác của collection cũ bị bỏ.

Lượt 1 đọc payload để đo BM25_AVG_LEN (số token BM25 trung bình mỗi chunk). Lượt 2 tạo collection đích (nếu chưa có),
chép từng điểm với cùng id, giữ nguyên vector dense và thêm bm25 tính từ payload `text`.

Không bao giờ sửa hay xóa collection nguồn. Chạy lại an toàn: cùng id nên upsert chỉ ghi đè chính điểm đó, chạy dở
thì chạy lại là xong. Cuối cùng so số điểm hai bên và in các giá trị cần đặt trong app/config.py.
"""
import argparse
import sys

from qdrant_client import QdrantClient, models

from app import bm25
from app import config as cfg
from app import qdrant_setup

COPIED_VECTORS = (cfg.DENSE_VECTOR_NAME,)
LEGACY_COLLECTION = "quy_dinh_hybrid"  # collection cũ, tạo trước khi chuyển sang BM25


def scroll_batches(client, name, batch_size, **kwargs):
    offset = None
    while True:
        points, offset = client.scroll(name, limit=batch_size, offset=offset, **kwargs)
        yield points
        if offset is None:
            return


def text_of(point) -> str:
    return (point.payload or {}).get("text") or ""


def measure_avg_len(client, source, tokenizer, batch_size):
    """(số chunk, số token BM25 trung bình mỗi chunk) trên collection nguồn, chỉ đọc payload."""
    chunks = tokens = 0
    for points in scroll_batches(client, source, batch_size, with_payload=["text"], with_vectors=False):
        for p in points:
            tokens += len(bm25.bm25_tokens(text_of(p), tokenizer))
            chunks += 1
    return chunks, (tokens / chunks if chunks else 0.0)


def copy_points(client, source, target, avg_len, tokenizer, batch_size):
    copied = 0
    # Chỉ lấy vector cần chép (dense), không kéo các vector khác của collection cũ về
    for points in scroll_batches(client, source, batch_size, with_payload=True, with_vectors=list(COPIED_VECTORS)):
        batch = [
            models.PointStruct(
                id=p.id, vector={**p.vector, **bm25.doc_vectors(text_of(p), avg_len, tokenizer)}, payload=p.payload
            )
            for p in points
        ]
        if batch:
            client.upsert(target, points=batch, wait=True)
        copied += len(batch)
        print(f"  đã chép {copied} điểm", flush=True)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="Chép collection cũ sang collection mới (dense + bm25)")
    ap.add_argument("--source", default=LEGACY_COLLECTION)
    ap.add_argument("--target", default=cfg.QDRANT_COLLECTION, help="mặc định: QDRANT_COLLECTION")
    ap.add_argument("--tokenizer", choices=cfg.BM25_TOKENIZERS, default=cfg.BM25_TOKENIZER)
    ap.add_argument("--batch-size", type=int, default=cfg.UPSERT_BATCH_SIZE)
    ap.add_argument("--url", default=cfg.QDRANT_URL)
    args = ap.parse_args(argv)

    if args.source == args.target:
        print("--target phải khác --source: script không bao giờ ghi vào collection nguồn.", file=sys.stderr)
        return 2
    client = QdrantClient(url=args.url, timeout=120)  # mac dinh 5s qua ngan cho scroll/upsert lo lon tren dia cham
    if not client.collection_exists(args.source):
        print(f"Không thấy collection nguồn '{args.source}' ở {args.url}.", file=sys.stderr)
        return 2
    missing = set(COPIED_VECTORS) - qdrant_setup.vector_names(client, args.source)
    if missing:
        print(f"Collection nguồn '{args.source}' thiếu vector {sorted(missing)}, không có gì để chép.", file=sys.stderr)
        return 2

    chunks, mean_len = measure_avg_len(client, args.source, args.tokenizer, args.batch_size)
    if not chunks:
        print(f"Collection nguồn '{args.source}' rỗng, không có gì để chép.", file=sys.stderr)
        return 2
    avg_len = round(mean_len, 2)  # đúng giá trị này vừa dùng để tính bm25 vừa in ra cho BM25_AVG_LEN
    print(f"Lượt 1: {chunks} chunk, trung bình {mean_len:.2f} token BM25/chunk (tokenizer={args.tokenizer})")

    if client.collection_exists(args.target):
        missing = {*COPIED_VECTORS, cfg.BM25_VECTOR_NAME} - qdrant_setup.vector_names(client, args.target)
        if missing:
            print(f"Collection đích '{args.target}' đã có nhưng thiếu vector {sorted(missing)}.", file=sys.stderr)
            return 2
        print(f"Lượt 2: collection đích '{args.target}' đã có, ghi đè các điểm cùng id")
    else:
        qdrant_setup.create_collection(client, args.target)
        print(f"Lượt 2: đã tạo collection đích '{args.target}'")
    copy_points(client, args.source, args.target, avg_len, args.tokenizer, args.batch_size)

    source_count = client.count(args.source, exact=True).count
    target_count = client.count(args.target, exact=True).count
    if source_count != target_count:
        print(
            f"SỐ ĐIỂM KHÔNG KHỚP: nguồn {source_count}, đích {target_count}. Nguồn đổi trong lúc chép, hoặc đích có điểm "
            f"cũ không còn ở nguồn: xóa collection đích '{args.target}' rồi chạy lại.",
            file=sys.stderr,
        )
        return 1
    print(f"\nXong: {target_count} điểm, '{args.source}' -> '{args.target}' (collection nguồn không bị thay đổi).")
    print("Đặt trong ai-service/app/config.py rồi build lại (docker-compose up -d --build ai-service):")
    print(f"  BM25_AVG_LEN = {avg_len}")
    if args.target != cfg.QDRANT_COLLECTION:  # các giá trị còn lại đã là mặc định của config.py
        print(f'  QDRANT_COLLECTION = "{args.target}"')
    if args.tokenizer != cfg.BM25_TOKENIZER:
        print(f'  BM25_TOKENIZER = "{args.tokenizer}"')
    print(f"(bm25 đã tính với BM25_K1={cfg.BM25_K1}, BM25_B={cfg.BM25_B}; đổi K1/B/tokenizer thì chạy lại script.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
