"""Reranker jina-reranker-v3.5: thu tu/nhom/cat doan, cach nap model, khong chay code tai ve (khong can model that)."""
import ast
import glob
import hashlib
import os
import re

import numpy as np
import pytest
import torch

from app import config as cfg
from app.models import encoders

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDORED = os.path.join(REPO_DIR, "app", "models", "jina_reranker_modeling.py")
MARKER = "# ----- upstream modeling.py starts below -----\n"


class FakeJina:
    def __init__(self):
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append(list(documents))
        results = [
            {"document": d, "relevance_score": np.float32(len(d) / 10), "index": np.int64(i)}
            for i, d in enumerate(documents)
        ]
        return sorted(results, key=lambda r: -r["relevance_score"])  # nhu ban that: da sap giam dan


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False, truncation=True, max_length=None):
        return {"input_ids": [ord(c) for c in text][:max_length]}

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


@pytest.fixture
def fake(monkeypatch):
    model = FakeJina()
    monkeypatch.setattr(encoders, "_reranker", model)
    monkeypatch.setattr(encoders, "_rerank_tokenizer", CharTokenizer())
    monkeypatch.setattr(cfg, "RERANK_MAX_LENGTH", 1000)
    monkeypatch.setattr(cfg, "RERANK_BATCH_SIZE", 64)
    return model


def test_scores_come_back_in_input_order_as_python_floats(fake):
    passages = ["a" * n for n in (4, 9, 2, 7)]  # diem = do dai / 10, nen thu tu giam dan la 9, 7, 4, 2
    scores = encoders.rerank_scores("q", passages)
    assert scores == pytest.approx([0.4, 0.9, 0.2, 0.7])
    assert all(type(s) is float for s in scores)


def test_candidates_are_scored_together_in_one_prompt(fake):
    encoders.rerank_scores("q", [f"doan {i}" for i in range(20)])
    assert [len(c) for c in fake.calls] == [20]  # listwise: mot lan goi cho ca 20 ung vien


def test_candidates_are_split_into_groups_of_rerank_batch_size(fake, monkeypatch):
    monkeypatch.setattr(cfg, "RERANK_BATCH_SIZE", 2)
    passages = ["a" * n for n in (3, 8, 1, 6, 5)]
    scores = encoders.rerank_scores("q", passages)
    assert [len(c) for c in fake.calls] == [2, 2, 1]
    assert scores == pytest.approx([0.3, 0.8, 0.1, 0.6, 0.5])


def test_long_passages_are_cut_to_rerank_max_length_tokens(fake, monkeypatch):
    monkeypatch.setattr(cfg, "RERANK_MAX_LENGTH", 5)
    encoders.rerank_scores("q", ["abcdefghij", "abc"])
    assert fake.calls == [["abcde", "abc"]]


def test_reranker_loads_the_pinned_revision_without_modeling_py(monkeypatch):
    seen = {}
    monkeypatch.setattr(cfg, "RERANK_MODEL", "org/some-reranker")
    monkeypatch.setattr(cfg, "RERANK_REVISION", "abc123")
    monkeypatch.setattr(cfg, "RERANK_DEVICE", "cuda:0")
    monkeypatch.setattr(cfg, "RERANK_DTYPE", "bfloat16")

    def fake_download(repo, **kwargs):
        seen["download"] = (repo, kwargs)
        return "/snapshot/dir"

    class FakeModel:
        def to(self, device):
            seen["device"] = device
            return self

        def eval(self):
            return self

    class FakeClass:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen["from_pretrained"] = (path, kwargs)
            return FakeModel()

    monkeypatch.setattr(encoders, "snapshot_download", fake_download)
    monkeypatch.setattr(encoders, "JinaForRanking", FakeClass)
    monkeypatch.setattr(encoders.AutoTokenizer, "from_pretrained", lambda path: f"tokenizer:{path}")
    monkeypatch.setattr(encoders, "_reranker", None)

    encoders.get_reranker()

    repo, kwargs = seen["download"]
    assert repo == "org/some-reranker" and kwargs["revision"] == "abc123"
    assert "modeling.py" in kwargs["ignore_patterns"]  # code tu xa khong duoc tai ve
    assert seen["from_pretrained"] == ("/snapshot/dir", {"dtype": torch.bfloat16})
    assert "trust_remote_code" not in seen["from_pretrained"][1]
    assert seen["device"] == "cuda:0" and encoders._rerank_tokenizer == "tokenizer:/snapshot/dir"


def test_reranker_on_cpu_uses_float32(monkeypatch):
    loaded = {}

    def fake_load(device, dtype):
        loaded.update(device=device, dtype=dtype)
        return "model", "tokenizer"

    monkeypatch.setattr(cfg, "RERANK_DEVICE", "cpu")
    monkeypatch.setattr(cfg, "RERANK_DTYPE", "bfloat16")
    monkeypatch.setattr(encoders, "_reranker", None)
    monkeypatch.setattr(encoders, "_load_reranker", fake_load)
    encoders.get_reranker()
    assert loaded == {"device": "cpu", "dtype": torch.float32}


def test_local_model_directory_skips_the_download(monkeypatch, tmp_path):
    def no_download(*args, **kwargs):
        pytest.fail("khong duoc tai khi da co thu muc local")

    monkeypatch.setattr(cfg, "RERANK_MODEL", str(tmp_path))
    monkeypatch.setattr(encoders, "snapshot_download", no_download)
    assert encoders._reranker_path() == str(tmp_path)


def python_files():
    return [f for f in glob.glob(os.path.join(REPO_DIR, "**", "*.py"), recursive=True) if "__pycache__" not in f]


def test_no_code_passes_trust_remote_code_true():
    """Nap model bang class vendored, khong bao gio chay code tai ve luc chay (ke ca tokenizer trong file vendored)."""
    offenders = []
    for path in python_files():
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), path)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "trust_remote_code":
                if not (isinstance(node.value, ast.Constant) and node.value.value is False):
                    offenders.append(f"{os.path.relpath(path, REPO_DIR)}:{node.value.lineno}")
    assert offenders == []


def test_vendored_modeling_is_the_pinned_upstream_plus_the_documented_change():
    with open(VENDORED, encoding="utf-8", newline="") as f:
        text = f.read().replace("\r\n", "\n")  # git co the doi LF -> CRLF khi checkout tren Windows
    header, body = text.split(MARKER, 1)

    assert cfg.RERANK_MODEL != "jinaai/jina-reranker-v3.5" or cfg.RERANK_REVISION in header
    expected_oid = re.search(r"git blob : ([0-9a-f]{40})", header).group(1)

    # Hoan nguyen dong duy nhat da sua -> phai ra dung file upstream (git blob oid cua file goc).
    upstream = body.replace(
        "AutoTokenizer.from_pretrained(self.name_or_path)",
        "AutoTokenizer.from_pretrained(self.name_or_path, trust_remote_code=True)",
    ).encode("utf-8")
    assert hashlib.sha1(b"blob %d\0" % len(upstream) + upstream).hexdigest() == expected_oid


def test_reranker_class_comes_from_the_vendored_module():
    assert encoders.JinaForRanking.__module__ == "app.models.jina_reranker_modeling"
