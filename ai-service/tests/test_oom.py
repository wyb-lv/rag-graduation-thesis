import numpy as np
import pytest
import torch

from app import config as cfg
from app.models import encoders


def oom():
    return torch.cuda.OutOfMemoryError("CUDA out of memory")


class FakeCrossEncoder:
    """Gia lap CrossEncoder.predict(): OOM khi batch_size > max_batch, khong thi diem = do dai doan / 10."""

    def __init__(self, max_batch=10**9):
        self.max_batch, self.batch_sizes = max_batch, []

    def predict(self, pairs, batch_size=32, show_progress_bar=None):
        self.batch_sizes.append(batch_size)
        if batch_size > self.max_batch:
            raise oom()
        return np.array([len(p) / 10 for _, p in pairs], dtype=np.float32)


@pytest.fixture(autouse=True)
def fake_state(monkeypatch):
    monkeypatch.setattr(cfg, "RERANK_BATCH_SIZE", 64)
    monkeypatch.setattr(encoders, "_reranker_on_cpu", False)


def test_oom_halves_batch_size_until_it_fits():
    tried = []

    def run(batch_size):
        tried.append(batch_size)
        if batch_size > 3:
            raise oom()
        return "ok"

    assert encoders._retry_on_oom(run, 16) == "ok"
    assert tried == [16, 8, 4, 2]


def test_oom_at_batch_size_one_is_raised():
    def run(batch_size):
        raise oom()

    with pytest.raises(torch.cuda.OutOfMemoryError):
        encoders._retry_on_oom(run, 4)


def test_reranker_oom_halves_the_batch_size_until_it_fits(monkeypatch):
    model = FakeCrossEncoder(max_batch=2)
    monkeypatch.setattr(encoders, "_reranker", model)
    passages = ["a" * n for n in (3, 9, 1, 7, 5)]

    scores = encoders.rerank_scores("q", passages)

    assert model.batch_sizes == [5, 2]  # 5 that bai (OOM) -> chia doi thanh 2 thi chay duoc
    assert scores == pytest.approx([0.3, 0.9, 0.1, 0.7, 0.5])  # van dung thu tu goc


def test_reranker_moves_to_cpu_on_oom_when_configured(monkeypatch):
    moved = []
    monkeypatch.setattr(cfg, "RERANK_CPU_ON_OOM", True)
    monkeypatch.setattr(encoders, "_reranker", FakeCrossEncoder(max_batch=0))

    def move():
        moved.append(True)
        encoders._reranker = FakeCrossEncoder()

    monkeypatch.setattr(encoders, "_move_reranker_to_cpu", move)
    assert encoders.rerank_scores("q", ["aa", "bbbb", "c"]) == pytest.approx([0.2, 0.4, 0.1])
    assert moved == [True]


def test_reranker_oom_is_raised_when_cpu_fallback_not_configured(monkeypatch):
    monkeypatch.setattr(cfg, "RERANK_CPU_ON_OOM", False)
    monkeypatch.setattr(encoders, "_reranker", FakeCrossEncoder(max_batch=0))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        encoders.rerank_scores("q", ["a", "b"])


def test_reranker_oom_is_raised_when_already_on_cpu(monkeypatch):
    monkeypatch.setattr(cfg, "RERANK_CPU_ON_OOM", True)
    monkeypatch.setattr(encoders, "_reranker_on_cpu", True)
    monkeypatch.setattr(encoders, "_reranker", FakeCrossEncoder(max_batch=0))
    monkeypatch.setattr(encoders, "_move_reranker_to_cpu", lambda: pytest.fail("khong duoc chuyen lai CPU"))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        encoders.rerank_scores("q", ["a", "b"])


def test_rerank_with_no_passages_skips_the_model(monkeypatch):
    monkeypatch.setattr(encoders, "get_reranker", lambda: pytest.fail("khong duoc nap reranker"))
    assert encoders.rerank_scores("q", []) == []
