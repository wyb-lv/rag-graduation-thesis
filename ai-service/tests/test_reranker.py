"""Reranker CrossEncoder (AITeamVN/Vietnamese_Reranker): thu tu/batch, cach nap model, khong chay code tai ve (khong can model that)."""
import ast
import glob
import os

import numpy as np
import pytest
import torch

from app import config as cfg
from app.models import encoders

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeCrossEncoder:
    """Gia lap CrossEncoder.predict(): diem = do dai doan / 10, tra ve ndarray float32 nhu ban that."""

    def __init__(self):
        self.calls = []

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        self.calls.append((list(pairs), batch_size))
        return np.array([len(p) / 10 for _, p in pairs], dtype=np.float32)


@pytest.fixture
def fake(monkeypatch):
    model = FakeCrossEncoder()
    monkeypatch.setattr(encoders, "_reranker", model)
    monkeypatch.setattr(cfg, "RERANK_BATCH_SIZE", 16)
    return model


def test_scores_come_back_in_input_order_as_python_floats(fake):
    passages = ["a" * n for n in (4, 9, 2, 7)]
    scores = encoders.rerank_scores("q", passages)
    assert scores == pytest.approx([0.4, 0.9, 0.2, 0.7])
    assert all(type(s) is float for s in scores)


def test_every_passage_is_scored_against_the_question_in_one_call(fake):
    passages = [f"doan {i}" for i in range(5)]
    encoders.rerank_scores("cau hoi", passages)
    (pairs, _), = fake.calls
    assert pairs == [("cau hoi", p) for p in passages]


def test_batch_size_is_rerank_batch_size_but_never_more_than_the_passages(fake, monkeypatch):
    encoders.rerank_scores("q", [f"d{i}" for i in range(40)])
    encoders.rerank_scores("q", ["d0", "d1", "d2"])
    monkeypatch.setattr(cfg, "RERANK_BATCH_SIZE", 4)
    encoders.rerank_scores("q", [f"d{i}" for i in range(10)])
    assert [bs for _, bs in fake.calls] == [16, 3, 4]


def test_reranker_loads_the_pinned_revision_on_the_configured_device_and_dtype(monkeypatch):
    seen = {}

    class RecordingCrossEncoder:
        def __init__(self, model, **kwargs):
            seen["model"], seen["kwargs"] = model, kwargs

    monkeypatch.setattr(cfg, "RERANK_MODEL", "org/some-reranker")
    monkeypatch.setattr(cfg, "RERANK_REVISION", "abc123")
    monkeypatch.setattr(cfg, "RERANK_DEVICE", "cuda:0")
    monkeypatch.setattr(cfg, "RERANK_DTYPE", "float16")
    monkeypatch.setattr(cfg, "RERANK_MAX_LENGTH", 384)
    monkeypatch.setattr(encoders, "CrossEncoder", RecordingCrossEncoder)
    monkeypatch.setattr(encoders, "_reranker", None)

    assert isinstance(encoders.get_reranker(), RecordingCrossEncoder)

    assert seen["model"] == "org/some-reranker"
    assert seen["kwargs"] == {
        "revision": "abc123", "device": "cuda:0", "max_length": 384, "model_kwargs": {"dtype": torch.float16},
    }
    assert "trust_remote_code" not in seen["kwargs"]  # model XLM-R chuan, khong chay code tai ve


def test_reranker_on_cpu_uses_float32(monkeypatch):
    loaded = {}

    def fake_load(device, dtype):
        loaded.update(device=device, dtype=dtype)
        return "model"

    monkeypatch.setattr(cfg, "RERANK_DEVICE", "cpu")
    monkeypatch.setattr(cfg, "RERANK_DTYPE", "float16")
    monkeypatch.setattr(encoders, "_reranker", None)
    monkeypatch.setattr(encoders, "_load_reranker", fake_load)
    encoders.get_reranker()
    assert loaded == {"device": "cpu", "dtype": torch.float32}


def test_local_model_directory_is_passed_straight_to_cross_encoder(monkeypatch, tmp_path):
    seen = {}

    class RecordingCrossEncoder:
        def __init__(self, model, **kwargs):
            seen["model"] = model

    monkeypatch.setattr(cfg, "RERANK_MODEL", str(tmp_path))
    monkeypatch.setattr(encoders, "CrossEncoder", RecordingCrossEncoder)
    encoders._load_reranker("cpu", torch.float32)
    assert seen["model"] == str(tmp_path)


def test_no_code_passes_trust_remote_code_true():
    """Khong bao gio chay code tai ve luc chay (model + tokenizer deu la lop chuan cua transformers)."""
    offenders = []
    for path in glob.glob(os.path.join(REPO_DIR, "**", "*.py"), recursive=True):
        if "__pycache__" in path:
            continue
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), path)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "trust_remote_code":
                if not (isinstance(node.value, ast.Constant) and node.value.value is False):
                    offenders.append(f"{os.path.relpath(path, REPO_DIR)}:{node.value.lineno}")
    assert offenders == []
