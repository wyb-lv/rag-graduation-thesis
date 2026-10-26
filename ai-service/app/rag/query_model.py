"""Tra loi cau hoi bang RAG hybrid: loc metadata -> BGE-M3 (dense + sparse) -> Qdrant RRF -> jina-reranker-v3.5 -> Groq Qwen.
Bo loc metadata (kho, hieu luc, khoa, loai_van_ban) gan vao TUNG lan tim vector nen Qdrant thu hep tai lieu truoc khi lay top-k.
Tuy chon (QUERY_REWRITE_ENABLED): Qwen viet lai cau hoi, tim dense + sparse cho ca hai cau, RRF mot lan tren moi danh sach
(Qdrant gop trung theo id) roi moi rerank theo cau goc.

Ba mode (RAG_MODE / --mode / tham so `mode`), khong can sua code:
  dense          : chi vector `dense`
  hybrid         : cac nhanh trong HYBRID_LEGS (dense, sparse; prefetch moi nhanh `top_k_retrieval`, hop nhat RRF)
  hybrid_rerank  : hybrid, roi jina-reranker-v3.5 giu `top_k_rerank` doan tot nhat
Nap du lieu vao Qdrant qua POST /embed hoac RabbitMQ (ingest/embed_service.py); chay docker compose up -d truoc.
"""
import argparse
import sys
from datetime import date, datetime, time

from qdrant_client import QdrantClient, models

from .. import config as cfg
from ..router import guard
from ..llm import groq_client
from ..models import encoders
from . import query_rewrite

NO_RESULT_ANSWER = "Không tìm thấy quy định liên quan đến câu hỏi trong các văn bản được cung cấp."

BASE_SYSTEM_PROMPT = (
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
# Them quy tac chong injection + canary (guard.py). BASE_SYSTEM_PROMPT la ban cu, dung khi GUARD_ENABLED=false.
SYSTEM_PROMPT = guard.harden_prompt(BASE_SYSTEM_PROMPT)

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


def _open_bound(field, bound):
    """Dieu kien theo moc (ngay hoac so); thieu field/null (moc de trong) coi nhu khong gioi han."""
    return models.Filter(should=[
        models.IsEmptyCondition(is_empty=models.PayloadField(key=field)),
        models.FieldCondition(key=field, range=bound),
    ])


def build_filter(kho=None, reference_date=None, khoa=None, loai_van_ban=None):
    """Filter payload: dung kho (neu co), van ban con hieu luc tai `reference_date` (mac dinh hom nay),
    neu co `khoa` cua sinh vien thi chi van ban ap dung cho khoa do (hoac khong gioi han khoa),
    va neu co `loai_van_ban` (danh sach, vd ["thong_bao"]) thi chi cac loai van ban do."""
    ref = _to_datetime(reference_date or date.today())
    must = [
        _open_bound(cfg.FIELD_HIEU_LUC_TU, models.DatetimeRange(lte=ref)),
        _open_bound(cfg.FIELD_HIEU_LUC_DEN, models.DatetimeRange(gte=ref)),
    ]
    if kho:
        must.insert(0, models.FieldCondition(key=cfg.FIELD_KHO, match=models.MatchValue(value=kho)))
    if khoa:
        must += [
            _open_bound(cfg.FIELD_KHOA_TU, models.Range(lte=khoa)),
            _open_bound(cfg.FIELD_KHOA_DEN, models.Range(gte=khoa)),
        ]
    if loai_van_ban:
        must.append(models.FieldCondition(key=cfg.FIELD_LOAI_VAN_BAN, match=models.MatchAny(any=list(loai_van_ban))))
    return models.Filter(must=must)


def _query_legs(question, legs):
    """{ten nhanh: vector truy van} cho cac nhanh co token; nhanh khong co token (vd cau hoi toan dau cau) bi bo.
    Ca hai nhanh dung chung mot lan encode BGE-M3 (GPU)."""
    dense_vecs, lexical_weights = encoders.encode_texts([question])
    queries = {}
    for leg in legs:
        if leg == "dense":
            queries[leg] = dense_vecs[0].tolist()
        elif leg == "sparse":
            sparse = encoders.to_sparse_vector(lexical_weights[0])
            if sparse.indices:
                queries[leg] = sparse
        else:
            raise ValueError(f"HYBRID_LEGS chi nhan 'dense' va 'sparse', nhan duoc '{leg}'")
    return queries


def _legs(question, mode):
    """{ten nhanh: vector truy van} cua mot cau; moi nhanh deu khong co token -> chi dung nhanh dense nhu truoc day."""
    return _query_legs(question, ("dense",) if mode == "dense" else cfg.HYBRID_LEGS) or _query_legs(question, ("dense",))


def _leg_prefetch(leg, query, flt, ef, top_k):
    return models.Prefetch(
        query=query,
        using=cfg.HYBRID_LEG_VECTORS[leg],
        filter=flt,
        limit=top_k,
        **({"params": ef} if leg == "dense" else {}),
    )


def retrieve(question, mode=None, kho=None, reference_date=None, top_k=None, khoa=None, loai_van_ban=None, alt_question=None):
    """Tim `top_k` (mac dinh top_k_retrieval) ung vien; moi hit: id, source, text, payload, retrieval/rerank_score.

    Mode `dense` chi truy van vector dense; `hybrid`/`hybrid_rerank` truy van cac nhanh trong HYBRID_LEGS roi hop nhat
    RRF phia Qdrant. Cung mot filter duoc gan vao TUNG prefetch: nhanh nao thieu filter se tra ve ca doan ngoai pham vi.
    Chi con mot nhanh co token thi truy van thang nhanh do (diem la diem cua nhanh, khong phai RRF).

    `alt_question` (cau viet lai, khac `question`): tim moi nhanh cho CA HAI cau, roi RRF mot lan tren tat ca danh sach
    (dense, sparse x 2 cau) trong mot request Qdrant. Filter van nam o tung nhanh, nen cau viet lai khong keo ve doan ngoai pham vi."""
    mode = _check_mode(mode)
    top_k = top_k or cfg.TOP_K_RETRIEVAL
    texts = [question] + ([alt_question] if alt_question and alt_question != question else [])
    legs = [(leg, query) for text in texts for leg, query in _legs(text, mode).items()]
    flt = build_filter(kho, reference_date, khoa, loai_van_ban)
    ef = models.SearchParams(hnsw_ef=cfg.HNSW_EF)  # efSearch chi co nghia voi nhanh dense (HNSW)

    if len(legs) == 1:
        (leg, query), = legs
        res = get_qdrant().query_points(
            cfg.QDRANT_COLLECTION,
            query=query,
            using=cfg.HYBRID_LEG_VECTORS[leg],
            query_filter=flt,
            limit=top_k,
            with_payload=True,
            **({"search_params": ef} if leg == "dense" else {}),
        )
    else:
        res = get_qdrant().query_points(
            cfg.QDRANT_COLLECTION,
            prefetch=[_leg_prefetch(leg, query, flt, ef, top_k) for leg, query in legs],
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


def build_messages(question, hits, history=None):
    """`history`: [(role, text)] cua cac luot truoc (session); chi vao prompt khi GUARD_ENABLED (prompt cu khong co khai niem nay)."""
    if cfg.GUARD_ENABLED:
        user = guard.build_user_message(question, hits, history)
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    context = "\n\n".join(f"[Đoạn {i}] Tài liệu: {h['source']}\n{h['text']}" for i, h in enumerate(hits, 1))
    user = f"Các đoạn trích:\n\n{context}\n\nCâu hỏi: {question}"
    return [{"role": "system", "content": BASE_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def generate_answer(question, hits, history=None):
    return groq_client.chat(build_messages(question, hits, history))


def generate_answer_stream(question, hits, history=None):
    """Generator cac doan text cua cau tra loi (xem groq_client.chat_stream); guard dau ra co the nem guard.OutputBlocked."""
    return guard.vet_stream(groq_client.chat_stream(build_messages(question, hits, history)))


def find_hits(question, top_k=None, mode=None, kho=None, reference_date=None, khoa=None, loai_van_ban=None):
    """Truy van + rerank + loc nguong: cac doan se dua vao LLM. Rong = cau hoi ngoai pham vi tai lieu.
    QUERY_REWRITE_ENABLED (chi hybrid / hybrid_rerank, vi mode dense loc theo cosine): Qwen viet lai `question`, dense + sparse cua ca
    hai cau vao MOT lan RRF; rerank van cham theo `question` goc, nen cau viet lai chi mo rong ung vien, khong doi tieu chi."""
    mode = _check_mode(mode)
    top_k = top_k or cfg.TOP_K_RERANK
    alt = query_rewrite.rewrite(question) if cfg.QUERY_REWRITE_ENABLED and mode != "dense" else None
    hits = retrieve(question, mode, kho, reference_date, khoa=khoa, loai_van_ban=loai_van_ban, alt_question=alt)
    if hits and mode == "hybrid_rerank":
        hits = rerank(question, hits, top_k)
    return [h for h in hits[:top_k] if _relevant(h, mode)]


def ask(question, top_k=None, mode=None, kho=None, reference_date=None, khoa=None, loai_van_ban=None,
        search_query=None, history=None):
    """Tra ve (answer, hits, in_scope). Khong con doan nao sau loc/nguong -> khong goi Groq, hits rong.
    Guard dau ra chan cau tra loi -> (GUARD_REFUSAL_ANSWER, [], False).
    `search_query` (cau doc lap do router viet lai) dung de encode / truy van / rerank; prompt tra loi van mang `question` goc."""
    hits = find_hits(search_query or question, top_k, mode, kho, reference_date, khoa, loai_van_ban)
    if not hits:
        return NO_RESULT_ANSWER, [], False
    answer = generate_answer(question, hits, history)
    if not guard.vet_answer(answer):
        return cfg.GUARD_REFUSAL_ANSWER, [], False
    return answer, hits, True


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Hoi dap RAG hybrid")
    parser.add_argument("question", nargs="*", help="cau hoi; bo trong de nhap tuong tac")
    parser.add_argument("--mode", choices=cfg.RAG_MODES, default=None, help=f"mac dinh: {cfg.RAG_MODE}")
    parser.add_argument("--kho", default=None, help="chi tim trong kho nay")
    parser.add_argument("--khoa", type=int, default=None, help="khoa cua sinh vien (vd 2024): loc van ban ap dung cho khoa nay")
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
        answer, hits, _ = ask(question, mode=args.mode, kho=args.kho, reference_date=args.reference_date, khoa=args.khoa)
        for h in hits:
            score = h["rerank_score"] if h["rerank_score"] is not None else h["retrieval_score"]
            print(f"Nguon: {h['source']} | score: {score:.4f}")
        print(answer)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        pass
