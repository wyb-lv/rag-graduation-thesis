"""Chay model that (BGE-M3 ~1.1GB, Vietnamese_Reranker ~1.1GB VRAM). Bo qua bang: pytest -m "not integration"."""
import numpy as np
import pytest

from app import config as cfg
from app.models import encoders
from app.rag import query_model as qm

pytestmark = pytest.mark.integration

TEXT = "Thí sinh được đăng ký tối đa 15 nguyện vọng theo quy chế tuyển sinh đại học năm 2026."


def test_encode_returns_normalized_1024_dim_dense_vectors():
    dense = encoders.encode_texts([TEXT, "câu khác"])

    assert dense.shape == (2, cfg.EMBED_DIM) == (2, 1024)
    assert np.allclose(np.linalg.norm(dense.astype("float32"), axis=1), 1.0, atol=1e-2)  # cosine -> da chuan hoa


def test_models_are_loaded_once_per_process():
    assert encoders.get_embedder() is encoders.get_embedder()
    assert encoders.get_reranker() is encoders.get_reranker()


def test_encoding_is_deterministic_between_ingest_and_query():
    first = encoders.encode_texts([TEXT])
    second = encoders.encode_texts([TEXT, "câu khác"])
    assert np.allclose(first[0].astype("float32"), second[0].astype("float32"), atol=1e-2)


QUESTION = "Thí sinh được đăng ký tối đa bao nhiêu nguyện vọng?"
PASSAGES = [
    "Học bổng Tinh hoa HSU giảm 100% học phí cho sinh viên xuất sắc.",
    TEXT,  # doan tra loi dung cau hoi
    "Thời tiết hôm nay nhiều mây, có mưa rào vào buổi chiều.",
    "Điểm sàn xét học bạ là 15/30 theo quy chế tuyển sinh mới.",
    "Công thức nấu phở bò gồm xương bò, hành nướng và gừng.",
    "Thí sinh chủ động chiến lược để giữ cơ hội vào đại học.",
    "Lịch thi tốt nghiệp THPT công bố vào tháng sáu hằng năm.",
    "Giảm một nửa điểm cộng với chứng chỉ IELTS.",
]


def test_reranker_returns_exactly_five_chunks_ordered_by_score():
    hits = [{"id": i, "source": "quy_che.pdf", "text": t, "payload": {"source": "quy_che.pdf", "text": t},
             "retrieval_score": 0.0, "rerank_score": None} for i, t in enumerate(PASSAGES)]

    top = qm.rerank(QUESTION, hits)

    assert len(top) == 5
    scores = [h["rerank_score"] for h in top]
    assert scores == sorted(scores, reverse=True)
    assert top[0]["id"] == 1  # doan tra loi dung cau hoi
    assert top[0]["payload"]["source"] == "quy_che.pdf"  # giu payload de trich dan


def test_reranker_scores_are_probabilities_and_only_the_answer_clears_the_floor():
    scores = encoders.rerank_scores(QUESTION, PASSAGES)

    assert len(scores) == len(PASSAGES) and all(0.0 <= s <= 1.0 for s in scores)
    assert scores[1] >= cfg.RERANK_MIN_SCORE
    assert all(s < cfg.RERANK_MIN_SCORE for i, s in enumerate(scores) if i != 1)


def test_reranker_scores_follow_the_input_order():
    forward = encoders.rerank_scores(QUESTION, PASSAGES)
    backward = encoders.rerank_scores(QUESTION, PASSAGES[::-1])

    assert len(backward) == len(PASSAGES)
    assert max(range(len(forward)), key=forward.__getitem__) == 1
    assert max(range(len(backward)), key=backward.__getitem__) == len(PASSAGES) - 2  # cung doan do o vi tri dao
