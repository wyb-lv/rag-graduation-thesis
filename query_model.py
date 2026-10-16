"""Tra loi cau hoi bang RAG hybrid: BGE-M3 (dense + sparse) -> Qdrant RRF -> jina-reranker-v3.5 -> Groq Qwen.

Ba mode (RAG_MODE / --mode / tham so `mode`), khong can sua code:
  dense          : chi vector `dense`
  hybrid         : dense + sparse (prefetch moi nhanh `top_k_retrieval`, hop nhat RRF)
  hybrid_rerank  : hybrid, roi jina-reranker-v3.5 giu `top_k_rerank` doan tot nhat
Nap du lieu vao Qdrant bang index_qdrant.py (chay docker compose up -d truoc).
"""
import argparse
import sys
from datetime import date, datetime, time

from qdrant_client import QdrantClient, models

import config as cfg
import encoders
import groq_client

NO_RESULT_ANSWER = "Không tìm thấy quy định liên quan đến câu hỏi trong các văn bản được cung cấp."

SYSTEM_PROMPT = (
    "Bạn là trợ lý hỏi đáp quy định của Trường Đại học Hoa Sen. "
    "Chỉ trả lời dựa trên các đoạn trích được cung cấp, không dùng kiến thức bên ngoài.\n"
    "Quy tắc:\n"
    "1. Trả lời bằng tiếng Việt, ngắn gọn, chính xác.\n"
    "2. Mỗi khẳng định phải kèm trích dẫn ngay sau đó, dạng [Tên văn bản, Điều X, Khoản Y]. "
    "Tên văn bản lấy từ nhãn \"Tài liệu\" của đoạn trích; số Điều/Khoản lấy từ chính nội dung đoạn trích. "
    "Nếu đoạn trích không nêu số Điều/Khoản thì chỉ ghi tên văn bản, tuyệt đối không tự đặt ra số Điều/Khoản.\n"
    "3. Nếu các đoạn trích không chứa thông tin để trả lời, hãy nói rõ: "
    "\"Các đoạn trích được cung cấp không chứa thông tin để trả lời câu hỏi này.\" và không suy đoán."
)

_qdrant = None


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=cfg.QDRANT_URL)
    return _qdrant


def _to_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    return datetime.fromisoformat(value)


def _open_bound(field, date_range):
    """Dieu kien theo moc ngay; thieu field/null (moc de trong) coi nhu khong gioi han."""
    return models.Filter(should=[
        models.IsEmptyCondition(is_empty=models.PayloadField(key=field)),
        models.FieldCondition(key=field, range=date_range),
    ])


def build_filter(kho=None, reference_date=None):
    """Filter payload: dung kho (neu co) va van ban con hieu luc tai `reference_date` (mac dinh hom nay)."""
    ref = _to_datetime(reference_date or date.today())
    must = [
        _open_bound(cfg.FIELD_HIEU_LUC_TU, models.DatetimeRange(lte=ref)),
        _open_bound(cfg.FIELD_HIEU_LUC_DEN, models.DatetimeRange(gte=ref)),
    ]
    if kho:
        must.insert(0, models.FieldCondition(key=cfg.FIELD_KHO, match=models.MatchValue(value=kho)))
    return models.Filter(must=must)


def retrieve(question, mode=None, kho=None, reference_date=None, top_k=None):
    """Tim `top_k` (mac dinh top_k_retrieval) ung vien; moi hit: id, source, text, payload, retrieval/rerank_score."""
    mode = _check_mode(mode)
    top_k = top_k or cfg.TOP_K_RETRIEVAL
    dense_vecs, lexical = encoders.encode_texts([question])
    dense = dense_vecs[0].tolist()
    sparse = encoders.to_sparse_vector(lexical[0])
    flt = build_filter(kho, reference_date)
    ef = models.SearchParams(hnsw_ef=cfg.HNSW_EF)

    if mode == "dense" or not sparse.indices:  # cau hoi khong co token sparse -> chi dung nhanh dense
        res = get_qdrant().query_points(
            cfg.QDRANT_COLLECTION,
            query=dense,
            using=cfg.DENSE_VECTOR_NAME,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
            search_params=ef,
        )
    else:
        res = get_qdrant().query_points(
            cfg.QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense, using=cfg.DENSE_VECTOR_NAME, filter=flt, limit=top_k, params=ef),
                models.Prefetch(query=sparse, using=cfg.SPARSE_VECTOR_NAME, filter=flt, limit=top_k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
    return [
        {
            "id": p.id,
            "source": p.payload.get("source"),
            "text": p.payload["text"],
            "payload": p.payload,
            "retrieval_score": float(p.score),  # dense: cosine | hybrid: diem RRF (khong phai cosine)
            "rerank_score": None,
        }
        for p in res.points
    ]


def rerank(question, hits, top_k=None):
    """Cham lai moi cap (question, text) bang reranker, sap giam dan theo diem, giu `top_k` (mac dinh top_k_rerank)."""
    scores = encoders.rerank_scores(question, [h["text"] for h in hits])
    ranked = sorted(zip(hits, scores), key=lambda hs: hs[1], reverse=True)
    return [{**h, "rerank_score": s} for h, s in ranked[:top_k or cfg.TOP_K_RERANK]]


def _check_mode(mode):
    mode = mode or cfg.RAG_MODE
    if mode not in cfg.RAG_MODES:
        raise ValueError(f"mode phai la mot trong {cfg.RAG_MODES}, nhan duoc '{mode}'")
    return mode


def _relevant(hit, mode):
    """Nguong tuong dong theo tung mode (xem config.SCORE_THRESHOLD / RERANK_MIN_SCORE)."""
    if mode == "dense":
        return hit["retrieval_score"] >= cfg.SCORE_THRESHOLD
    if mode == "hybrid_rerank":
        return hit["rerank_score"] >= cfg.RERANK_MIN_SCORE
    return True


def build_messages(question, hits):
    context = "\n\n".join(f"[Đoạn {i}] Tài liệu: {h['source']}\n{h['text']}" for i, h in enumerate(hits, 1))
    user = f"Các đoạn trích:\n\n{context}\n\nCâu hỏi: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def generate_answer(question, hits):
    return groq_client.chat(build_messages(question, hits))


def generate_answer_stream(question, hits):
    """Generator cac doan text cua cau tra loi (xem groq_client.chat_stream)."""
    return groq_client.chat_stream(build_messages(question, hits))


def find_hits(question, top_k=None, mode=None, kho=None, reference_date=None):
    """Truy van + rerank + loc nguong: cac doan se dua vao LLM. Rong = cau hoi ngoai pham vi tai lieu."""
    mode = _check_mode(mode)
    top_k = top_k or cfg.TOP_K_RERANK
    hits = retrieve(question, mode, kho, reference_date)
    if hits and mode == "hybrid_rerank":
        hits = rerank(question, hits, top_k)
    return [h for h in hits[:top_k] if _relevant(h, mode)]


def ask(question, top_k=None, mode=None, kho=None, reference_date=None):
    """Tra ve (answer, hits, in_scope). Khong con doan nao sau loc/nguong -> khong goi Groq, hits rong."""
    hits = find_hits(question, top_k, mode, kho, reference_date)
    if not hits:
        return NO_RESULT_ANSWER, [], False
    return generate_answer(question, hits), hits, True


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Hoi dap RAG hybrid")
    parser.add_argument("question", nargs="*", help="cau hoi; bo trong de nhap tuong tac")
    parser.add_argument("--mode", choices=cfg.RAG_MODES, default=None, help=f"mac dinh: {cfg.RAG_MODE}")
    parser.add_argument("--kho", default=None, help="chi tim trong kho nay")
    parser.add_argument(
        "--date", dest="reference_date", default=None, help="ngay tham chieu YYYY-MM-DD (mac dinh hom nay)"
    )
    args = parser.parse_args()

    if args.question:
        questions = [" ".join(args.question)]
    else:
        print("Nhap cau hoi (dong trong hoac Ctrl+C de thoat):")
        questions = iter(lambda: input("> ").strip(), "")

    for question in questions:
        if not question:
            continue
        answer, hits, _ = ask(question, mode=args.mode, kho=args.kho, reference_date=args.reference_date)
        for h in hits:
            score = h["rerank_score"] if h["rerank_score"] is not None else h["retrieval_score"]
            print(f"Nguon: {h['source']} | score: {score:.4f}")
        print(answer)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        pass
