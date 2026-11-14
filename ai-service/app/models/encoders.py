"""Nap BGE-M3 (dense, SentenceTransformer) va reranker (CrossEncoder) mot lan moi tien trinh, ca hai bang sentence-transformers.

Nhanh lexical la BM25 (bm25.py, chay CPU).
Ingest (embed_service.py) va truy van (query_model.py) deu di qua encode_texts() de dung cung mot duong ma hoa.
Reranker la model XLM-R chuan (...ForSequenceClassification) nen khong can va khong chay code tai ve luc chay.
"""
import os
import threading

import scipy.linalg  # noqa: F401  (must load BLAS before torch to avoid a Windows DLL/heap crash)
import sklearn.metrics.pairwise  # noqa: F401

import torch
from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer

from .. import config as cfg

_embedder = None
_reranker = None
_reranker_on_cpu = False
_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # model GPU dung chung giua cac thread cua FastAPI -> chay tuan tu


def _device(configured):
    return configured or ("cuda:0" if torch.cuda.is_available() else "cpu")


def _embedder_path():
    """Thu muc local cua BGE-M3 chi gom file can cho suy luan (SentenceTransformer tu tai ca repo, them ~2GB ONNX/anh)."""
    if os.path.exists(cfg.EMBED_MODEL):
        path = cfg.EMBED_MODEL
    else:
        path = snapshot_download(
            cfg.EMBED_MODEL,
            ignore_patterns=["onnx/*", "imgs/*", "*.jpg", "flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
        )

    # ── Tự động tạo model.safetensors nếu chỉ có pytorch_model.bin (Tránh CVE-2025-32434 check) ──
    bin_file = os.path.join(path, "pytorch_model.bin")
    st_file = os.path.join(path, "model.safetensors")
    if os.path.exists(bin_file) and not os.path.exists(st_file):
        from safetensors.torch import save_file
        save_file(torch.load(bin_file, map_location="cpu"), st_file)

    return path


def get_embedder():
    global _embedder
    with _load_lock:
        if _embedder is None:
            device = _device(cfg.EMBED_DEVICE)
            kwargs = {"dtype": torch.float16} if cfg.USE_FP16 and device.startswith("cuda") else {}
            model = SentenceTransformer(_embedder_path(), device=device, model_kwargs=kwargs)
            model.max_seq_length = cfg.EMBED_MAX_LENGTH
            _embedder = model
        return _embedder


def _load_reranker(device, dtype):
    """RERANK_MODEL la Hub id (weights, config, tokenizer lay dung RERANK_REVISION) hoac thu muc local."""
    return CrossEncoder(
        cfg.RERANK_MODEL,
        revision=cfg.RERANK_REVISION or None,
        device=device,
        max_length=cfg.RERANK_MAX_LENGTH,
        model_kwargs={"dtype": dtype},
    )


def get_reranker():
    global _reranker
    with _load_lock:
        if _reranker is None:
            device = _device(cfg.RERANK_DEVICE)
            dtype = torch.float32 if device.startswith("cpu") else getattr(torch, cfg.RERANK_DTYPE)
            _reranker = _load_reranker(device, dtype)
        return _reranker


def _move_reranker_to_cpu():
    """Tai lai reranker tren CPU o fp32 (ban fp16 dang nam tren GPU)."""
    global _reranker, _reranker_on_cpu
    with _load_lock:
        _reranker = None
        torch.cuda.empty_cache()
        _reranker = _load_reranker("cpu", torch.float32)
        _reranker_on_cpu = True
        return _reranker


def _retry_on_oom(run, batch_size):
    """Goi run(batch_size); khi CUDA OOM thi giam doi batch size cho den 1 roi moi nem loi."""
    while True:
        try:
            return run(batch_size)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size // 2)


def encode_texts(texts, batch_size=None):
    """Ma hoa dense bang BGE-M3: tra ve ndarray [n, 1024] da chuan hoa (cosine)."""
    embedder = get_embedder()

    def run(bs):
        return embedder.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    with _infer_lock:
        return _retry_on_oom(run, batch_size or cfg.EMBED_BATCH_SIZE)


def rerank_scores(question, passages):
    """Cham tung cap (cau hoi, doan) bang CrossEncoder: diem 0..1 (sigmoid, cao hon = lien quan hon), moi cap cham doc lap nen
    so sanh duoc giua moi doan. Cap dai hon RERANK_MAX_LENGTH token bi cat. Tra ve theo dung thu tu cua `passages`."""
    if not passages:
        return []
    pairs = [(question, p) for p in passages]

    def run(batch_size):
        return [float(s) for s in get_reranker().predict(pairs, batch_size=batch_size, show_progress_bar=False)]

    batch_size = min(cfg.RERANK_BATCH_SIZE, len(passages))
    with _infer_lock:
        try:
            return _retry_on_oom(run, batch_size)
        except torch.cuda.OutOfMemoryError:
            if not cfg.RERANK_CPU_ON_OOM or _reranker_on_cpu:
                raise
        # Ra khoi khoi except roi moi chuyen sang CPU: traceback cua loi OOM con giu model GPU, chuyen trong khoi
        # except thi VRAM chua duoc giai phong.
        _move_reranker_to_cpu()
        return run(batch_size)
