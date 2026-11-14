"""Tra loi cau hoi bang RAG hybrid + rerank: loc metadata -> model embedding (dense) + BM25 -> Qdrant RRF -> CrossEncoder reranker -> Groq Qwen.
Bo loc metadata (kho, hieu luc, khoa, loai_van_ban) gan vao TUNG lan tim vector nen Qdrant thu hep tai lieu truoc khi lay top-k.
Cac nhanh trong HYBRID_LEGS (dense, bm25) lay moi nhanh `top_k_retrieval` ung vien, hop nhat RRF phia Qdrant, roi
reranker giu `top_k_rerank` doan tot nhat. Qwen viet lai cau hoi (query_rewrite.py), tim dense + bm25
cho ca hai cau, RRF mot lan tren moi danh sach (Qdrant gop trung theo id) roi moi rerank: reranker cham CA HAI cau, diem la trung binh.
Nap du lieu vao Qdrant qua POST /embed hoac RabbitMQ (ingest/embed_service.py); chay docker compose up -d truoc.
"""
import argparse
import sys
from datetime import date, datetime, time

from qdrant_client import QdrantClient, models

from .. import bm25
from .. import config as cfg
from .. import texts
from ..router import guard
from ..llm import groq_client
from ..models import encoders
from . import query_rewrite

NO_RESULT_ANSWER = cfg.NO_RESULT_ANSWER

BASE_SYSTEM_PROMPT = texts.read_text("answer_system.txt")
# Them quy tac chong injection + canary (guard.py).
SYSTEM_PROMPT = guard.harden_prompt(BASE_SYSTEM_PROMPT)

VOICE_CALL_SYSTEM_PROMPT = guard.harden_prompt(
    BASE_SYSTEM_PROMPT
    + "\n4. ĐÂY LÀ CUỘC GỌI THOẠI TRỰC TIẾP (người dùng chỉ nghe âm thanh, không nhìn màn hình): "
    "Hãy trả lời ngắn gọn trong 2 đến 3 câu súc tích, đi thẳng vào thông tin quan trọng nhất, "
    "dùng văn phong nói tự nhiên, KHÔNG dùng danh sách gạch đầu dòng hay ký hiệu Markdown."
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
    Chi nap model embedding (GPU) khi co nhanh dense; bm25 chay CPU (bm25.py)."""
    queries = {}
    for leg in legs:
        if leg == "dense":
            queries[leg] = encoders.encode_texts([question])[0].tolist()
        elif leg == "bm25":
            vec = bm25.bm25_query_vector(question)
            if vec.indices:
                queries[leg] = vec
        else:
            raise ValueError(f"HYBRID_LEGS chi nhan 'dense' va 'bm25', nhan duoc '{leg}'")
    return queries


def _legs(question):
    """{ten nhanh: vector truy van} cua mot cau; moi nhanh deu khong co token -> chi dung nhanh dense."""
    return _query_legs(question, cfg.HYBRID_LEGS) or _query_legs(question, ("dense",))


def _leg_prefetch(leg, query, flt, ef, top_k):
    return models.Prefetch(
        query=query,
        using=cfg.HYBRID_LEG_VECTORS[leg],
        filter=flt,
        limit=top_k,
        **({"params": ef} if leg == "dense" else {}),
    )


def retrieve(question, kho=None, reference_date=None, top_k=None, khoa=None, loai_van_ban=None, alt_question=None):
    """Tim `top_k` (mac dinh top_k_retrieval) ung vien; moi hit: id, source, text, payload, retrieval/rerank_score.

    Truy van cac nhanh trong HYBRID_LEGS roi hop nhat RRF phia Qdrant. Cung mot filter duoc gan vao TUNG prefetch:
    nhanh nao thieu filter se tra ve ca doan ngoai pham vi.
    Chi con mot nhanh co token thi truy van thang nhanh do (diem la diem cua nhanh, khong phai RRF).

    `alt_question` (cau viet lai, khac `question`): tim moi nhanh cho CA HAI cau, roi RRF mot lan tren tat ca danh sach
    (dense, bm25 x 2 cau) trong mot request Qdrant. Filter van nam o tung nhanh, nen cau viet lai khong keo ve doan ngoai pham vi."""
    top_k = top_k or cfg.TOP_K_RETRIEVAL
    texts = [question] + ([alt_question] if alt_question and alt_question != question else [])
    legs = [(leg, query) for text in texts for leg, query in _legs(text).items()]
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
            "retrieval_score": float(p.score),  # diem RRF (khong phai cosine); chi con mot nhanh thi la diem cua nhanh do
            "rerank_score": None,
        }
        for p in res.points
    ]


def rerank(question, hits, top_k=None, alt_question=None):
    """Cham lai moi cap (question, text) bang reranker, sap giam dan theo diem, giu `top_k` (mac dinh top_k_rerank).
    Co `alt_question` (cau viet lai) thi cham CA HAI cau va lay trung binh. Cham rieng cau goc thi tu long ("rot mon", "no mon") lam diem
    doan dung chi con ~0.1 (reranker khong hieu), duoi RERANK_MIN_SCORE nen bi loai het; cau viet lai dung tu chinh thuc cho ~0.9.
    Trung binh (khong lay max) de mot cau viet lai sai khong mot minh keo diem len."""
    texts = [h["text"] for h in hits]
    scores = encoders.rerank_scores(question, texts)
    if alt_question:
        scores = [(a + b) / 2 for a, b in zip(scores, encoders.rerank_scores(alt_question, texts))]
    ranked = sorted(zip(hits, scores), key=lambda hs: hs[1], reverse=True)
    return [{**h, "rerank_score": s} for h, s in ranked[:top_k or cfg.TOP_K_RERANK]]


def build_messages(question, hits, history=None, voice_mode=False):
    """`history`: [(role, text)] cua cac luot truoc (session); `voice_mode`: prompt khau ngu ngan gon cho cuoc goi thoai."""
    user = guard.build_user_message(question, hits, history)
    return [{"role": "system", "content": VOICE_CALL_SYSTEM_PROMPT if voice_mode else SYSTEM_PROMPT}, {"role": "user", "content": user}]


def generate_answer(question, hits, history=None):
    return groq_client.chat(build_messages(question, hits, history))

def generate_answer_stream(question, hits, history=None, voice_mode: bool = False):
    """Generator các đoạn text của câu trả lời; hỗ trợ voice_mode cho cuộc gọi trực tiếp."""
    return guard.vet_stream(groq_client.chat_stream(build_messages(question, hits, history, voice_mode=voice_mode)))


def find_hits(question, top_k=None, kho=None, reference_date=None, khoa=None, loai_van_ban=None):
    """Truy van hybrid + rerank + loc nguong (RERANK_MIN_SCORE): cac doan se dua vao LLM. Rong = cau hoi ngoai pham vi tai lieu.
    Qwen viet lai `question` (query_rewrite.rewrite), dense + bm25 cua ca hai cau vao MOT lan RRF; rerank cham CA HAI cau (diem trung binh),
    de cau viet lai dung tu chinh thuc cuu duoc cau hoi dung tu long."""
    top_k = top_k or cfg.TOP_K_RERANK
    alt = query_rewrite.rewrite(question)
    hits = retrieve(question, kho, reference_date, khoa=khoa, loai_van_ban=loai_van_ban, alt_question=alt)
    hits = rerank(question, hits, top_k, alt_question=alt) if hits else []
    return [h for h in hits if h["rerank_score"] >= cfg.RERANK_MIN_SCORE]


def ask(question, top_k=None, kho=None, reference_date=None, khoa=None, loai_van_ban=None,
        search_query=None, history=None):
    """Tra ve (answer, hits, in_scope). Khong con doan nao sau loc/nguong -> khong goi Groq, hits rong.
    Guard dau ra chan cau tra loi -> (GUARD_REFUSAL_ANSWER, [], False).
    `search_query` (cau doc lap do router viet lai) dung de encode / truy van / rerank; prompt tra loi van mang `question` goc."""
    hits = find_hits(search_query or question, top_k, kho, reference_date, khoa, loai_van_ban)
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
        answer, hits, _ = ask(question, kho=args.kho, reference_date=args.reference_date, khoa=args.khoa)
        for h in hits:
            print(f"Nguon: {h['source']} | score: {h['rerank_score']:.4f}")
        print(answer)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        pass
