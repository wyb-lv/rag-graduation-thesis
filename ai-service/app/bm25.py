"""Nhánh lexical BM25 cho hybrid retrieval: token hóa tiếng Việt và tạo SparseVector cho Qdrant.

Chỉ chạy CPU, không dùng encoders.py (không dính GPU lock). Một bộ hàm duy nhất cho cả lúc index và lúc truy vấn,
nên token ở hai phía luôn khớp nhau.

Vì sao tự mã hóa thay vì BM25 có sẵn của Qdrant: bản đó mặc định stemming + stopwords tiếng Anh, còn tách theo
khoảng trắng làm vỡ từ ghép tiếng Việt ("học bổng", "điều 15").

IDF không tính ở đây: collection khai báo `modifier=IDF` cho vector `bm25` nên Qdrant tự nhân IDF khi truy vấn.
Phía code chỉ tính phần tf + chuẩn hóa độ dài của tài liệu.
"""
import hashlib
import re
import threading
import unicodedata
from collections import Counter
from typing import Optional

from qdrant_client import models

from . import config as cfg

_WORD = re.compile(r"[^\W_]+")  # chữ + số theo Unicode, bỏ dấu câu và "_"
_pyvi_lock = threading.Lock()  # CRF tagger của pyvi giữ trạng thái nội bộ, không an toàn khi gọi song song


def _syllables(text: str) -> list[str]:
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


def _pyvi_words(text: str) -> list[str]:
    try:
        from pyvi import ViTokenizer  # nạp lười: kéo theo scikit-learn, chỉ cần khi BM25_TOKENIZER=pyvi
    except ImportError as e:
        raise RuntimeError("BM25_TOKENIZER=pyvi cần thư viện pyvi (pip install pyvi)") from e
    with _pyvi_lock:
        segmented = ViTokenizer.tokenize(unicodedata.normalize("NFC", text))
    return re.findall(r"\w+", segmented.lower())  # \w giữ "_" nên từ ghép "học_bổng" không bị tách lại


def bm25_tokens(text: str, tokenizer: Optional[str] = None) -> list[str]:
    """Token BM25 của `text`, giữ dấu tiếng Việt và chữ số.

    syllable         : từng âm tiết ("học", "bổng", "điều", "15")
    syllable_bigram  : âm tiết + cặp âm tiết liền nhau nối bằng "_" ("học_bổng", "điều_15")
    pyvi             : từ ghép do pyvi tách ("học_bổng"); chậm hơn và cho kết quả phụ thuộc ngữ cảnh câu
    """
    tokenizer = tokenizer or cfg.BM25_TOKENIZER
    if tokenizer == "pyvi":
        return _pyvi_words(text)
    syllables = _syllables(text)
    if tokenizer == "syllable":
        return syllables
    if tokenizer == "syllable_bigram":
        return syllables + [f"{a}_{b}" for a, b in zip(syllables, syllables[1:])]
    raise ValueError(f"tokenizer phải là một trong {cfg.BM25_TOKENIZERS}, nhận được '{tokenizer}'")


def _index(token: str) -> int:
    """Chỉ số sparse 32 bit ổn định giữa các tiến trình (hash() của Python bị ngẫu nhiên hóa theo từng lần chạy)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "little")


def _sparse(weights: dict[str, float]) -> models.SparseVector:
    merged: dict[int, float] = {}
    for token, weight in weights.items():
        i = _index(token)
        merged[i] = merged.get(i, 0.0) + weight  # hai token trùng hash gộp làm một, chỉ số phải duy nhất
    indices = sorted(merged)
    return models.SparseVector(indices=indices, values=[merged[i] for i in indices])


def bm25_doc_vector(
    text: str, avg_len: Optional[float] = None, tokenizer: Optional[str] = None
) -> models.SparseVector:
    """Trọng số tf của chunk: tf*(k1+1) / (tf + k1*(1 - b + b*len/avg_len)). Không có token thì trả vector rỗng.

    `avg_len` mặc định BM25_AVG_LEN. Giá trị này đi vào trọng số lúc ghi nên phải giống nhau giữa các lần index.
    """
    tokens = bm25_tokens(text, tokenizer)
    if not tokens:
        return models.SparseVector(indices=[], values=[])
    k1, b = cfg.BM25_K1, cfg.BM25_B
    norm = k1 * (1 - b + b * len(tokens) / (avg_len or cfg.BM25_AVG_LEN))
    return _sparse({token: tf * (k1 + 1) / (tf + norm) for token, tf in Counter(tokens).items()})


def bm25_query_vector(text: str) -> models.SparseVector:
    """Trọng số 1.0 cho mỗi token duy nhất của câu hỏi (IDF do Qdrant nhân thêm)."""
    return _sparse({token: 1.0 for token in set(bm25_tokens(text))})
