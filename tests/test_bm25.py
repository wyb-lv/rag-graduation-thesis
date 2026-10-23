import os
import subprocess
import sys
import unicodedata

import pytest

from app import bm25
from app import config as cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_syllable_bigram_keeps_diacritics_digits_and_makes_compounds():
    tokens = bm25.bm25_tokens("Học bổng KHUYẾN KHÍCH — Điều 15, khoản 2.", "syllable_bigram")
    assert {"học", "bổng", "khuyến", "khích", "điều", "15", "khoản", "2"} <= set(tokens)  # đơn vị, còn dấu
    assert {"học_bổng", "khuyến_khích", "điều_15", "khoản_2"} <= set(tokens)  # cặp liền nhau
    assert "hoc_bong" not in tokens and "học bổng" not in tokens


def test_syllable_tokenizer_has_unigrams_only():
    tokens = bm25.bm25_tokens("Điều 15 học bổng", "syllable")
    assert tokens == ["điều", "15", "học", "bổng"]


def test_tokenizer_defaults_to_config(monkeypatch):
    monkeypatch.setattr(cfg, "BM25_TOKENIZER", "syllable")
    assert bm25.bm25_tokens("học bổng") == ["học", "bổng"]


def test_unicode_is_normalized_to_nfc():
    decomposed = unicodedata.normalize("NFD", "Điều 15 học bổng")
    assert decomposed != "Điều 15 học bổng"
    assert bm25.bm25_tokens(decomposed) == bm25.bm25_tokens("Điều 15 học bổng")


def test_unknown_tokenizer_is_rejected():
    with pytest.raises(ValueError):
        bm25.bm25_tokens("học bổng", "whitespace")


def test_doc_vector_uses_bm25_term_frequency_and_length_normalization(monkeypatch):
    monkeypatch.setattr(cfg, "BM25_K1", 1.2)
    monkeypatch.setattr(cfg, "BM25_B", 0.75)
    vec = bm25.bm25_doc_vector("học học bổng", avg_len=4.0, tokenizer="syllable")  # 3 token, "học" tf=2, "bổng" tf=1
    weights = dict(zip(vec.indices, vec.values))
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / 4.0)
    assert weights[bm25._index("học")] == pytest.approx(2 * 2.2 / (2 + norm))
    assert weights[bm25._index("bổng")] == pytest.approx(1 * 2.2 / (1 + norm))


def test_longer_chunk_gets_smaller_weight_for_the_same_term():
    short = bm25.bm25_doc_vector("học bổng", avg_len=10, tokenizer="syllable")
    long = bm25.bm25_doc_vector("học bổng " + "khác " * 30, avg_len=10, tokenizer="syllable")
    i = bm25._index("học")
    assert dict(zip(long.indices, long.values))[i] < dict(zip(short.indices, short.values))[i]


def test_doc_vector_avg_len_defaults_to_config(monkeypatch):
    monkeypatch.setattr(cfg, "BM25_AVG_LEN", 50.0)
    assert bm25.bm25_doc_vector("học bổng") == bm25.bm25_doc_vector("học bổng", avg_len=50.0)


def test_query_vector_is_one_per_unique_token(monkeypatch):
    monkeypatch.setattr(cfg, "BM25_TOKENIZER", "syllable")
    vec = bm25.bm25_query_vector("học học bổng")
    assert vec.values == [1.0, 1.0] and len(vec.indices) == 2


def test_query_for_an_article_contains_the_article_bigram():
    vec = bm25.bm25_query_vector("Điều 15")
    assert len(vec.indices) == 3 and bm25._index("điều_15") in vec.indices  # điều, 15, điều_15


@pytest.mark.parametrize("text", ["", "   ", "?!... — ()"])
def test_text_without_tokens_gives_empty_vectors(text):
    assert bm25.bm25_doc_vector(text).indices == [] and bm25.bm25_doc_vector(text).values == []
    assert bm25.bm25_query_vector(text).indices == []


def test_indices_are_unique_sorted_and_fit_u32():
    vec = bm25.bm25_doc_vector("quy chế học bổng điều 15 khoản 2 học bổng khuyến khích " * 20)
    assert vec.indices == sorted(set(vec.indices))
    assert all(0 <= i < 2**32 for i in vec.indices)
    assert len(vec.indices) == len(vec.values)


def test_hash_collisions_are_merged_into_one_index(monkeypatch):
    monkeypatch.setattr(bm25, "_index", lambda token: 7)
    vec = bm25.bm25_query_vector("học bổng")  # 3 token (học, bổng, học_bổng) cùng một chỉ số
    assert vec.indices == [7] and vec.values == [3.0]
    doc = bm25.bm25_doc_vector("học bổng", avg_len=1)
    assert doc.indices == [7]


def test_token_hash_scheme_is_pinned():
    # Đổi cách băm làm hỏng toàn bộ vector bm25 đã index (chỉ số cũ không còn khớp chỉ số mới).
    assert bm25._index("học_bổng") == 1024466451
    assert bm25._index("điều_15") == 781069008
    assert bm25._index("olympic") == 3157598511


def test_indices_do_not_depend_on_python_hash_seed():
    code = (
        "from app import bm25; "
        "v = bm25.bm25_doc_vector('Học bổng khuyến khích, Điều 15'); "
        "print(v.indices, [round(x, 6) for x in v.values])"
    )
    outputs = set()
    for seed in ("1", "12345", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        outputs.add(out.stdout.strip())
    assert len(outputs) == 1


def test_pyvi_tokenizer_segments_compound_words():
    pytest.importorskip("pyvi")
    tokens = bm25.bm25_tokens("Sinh viên được xét học bổng khuyến khích học tập theo Điều 15.", "pyvi")
    assert "học_bổng" in tokens and "khuyến_khích" in tokens
    assert "15" in tokens and "." not in tokens
