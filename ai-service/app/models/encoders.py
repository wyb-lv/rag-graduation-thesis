"""Nap BGE-M3 (dense + sparse, FlagEmbedding) va jina-reranker-v3.5 (transformers) mot lan moi tien trinh.

Ingest (index_qdrant.py) va truy van (query_model.py) deu di qua encode_texts() de dung cung mot duong ma hoa.
Reranker dung ban sao da duyet cua modeling.py (jina_reranker_modeling.py) nen khong chay code tai ve luc chay.
"""
import os
import threading

import scipy.linalg  # noqa: F401  (must load BLAS before torch to avoid a Windows DLL/heap crash)
import sklearn.metrics.pairwise  # noqa: F401

import torch
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download
from qdrant_client import models
from transformers import AutoTokenizer

from .. import config as cfg
from .jina_reranker_modeling import JinaForRanking

_embedder = None
_reranker = None
_rerank_tokenizer = None
_reranker_on_cpu = False
_load_lock = threading.Lock()
_infer_lock = threading.Lock()  # model GPU dung chung giua cac thread cua FastAPI -> chay tuan tu


def _device(configured):
    return configured or ("cuda:0" if torch.cuda.is_available() else "cpu")


def _embedder_path():
    """Thu muc local cua BGE-M3 chi gom file can cho suy luan.

    FlagEmbedding tu snapshot_download ca repo (them ~2GB ONNX/anh) va neu thieu sparse_linear.pt thi am tham
    khoi tao NGAU NHIEN dau sparse -> trong so lexical vo nghia. Tai san loai file thua va bao loi neu thieu.
    """
    if os.path.exists(cfg.EMBED_MODEL):
        path = cfg.EMBED_MODEL
    else:
        path = snapshot_download(
            cfg.EMBED_MODEL,
            ignore_patterns=["onnx/*", "imgs/*", "*.jpg", "flax_model.msgpack", "rust_model.ot", "tf_model.h5"],
        )
    for name in ("sparse_linear.pt", "colbert_linear.pt"):
        if not os.path.exists(os.path.join(path, name)):
            raise FileNotFoundError(f"Thieu {name} trong {path}: dau sparse cua BGE-M3 se bi khoi tao ngau nhien")

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
            _embedder = BGEM3FlagModel(
                _embedder_path(),
                use_fp16=cfg.USE_FP16,
                devices=_device(cfg.EMBED_DEVICE),
                batch_size=cfg.EMBED_BATCH_SIZE,
                query_max_length=cfg.EMBED_MAX_LENGTH,
                passage_max_length=cfg.EMBED_MAX_LENGTH,
            )
        return _embedder


def _reranker_path():
    """Thu muc local cua reranker o dung revision da ghim (weights, config, tokenizer).

    Bo qua modeling.py cua repo: ban dang chay la jina_reranker_modeling.py trong repo nay, nen khong can va khong
    nap code tai ve. Doi RERANK_REVISION thi phai duyet va vendor lai file do.
    """
    if os.path.exists(cfg.RERANK_MODEL):
        return cfg.RERANK_MODEL
    return snapshot_download(cfg.RERANK_MODEL, revision=cfg.RERANK_REVISION or None, ignore_patterns=["modeling.py"])


def _load_reranker(device, dtype):
    path = _reranker_path()
    model = JinaForRanking.from_pretrained(path, dtype=dtype)
    return model.to(device).eval(), AutoTokenizer.from_pretrained(path)


def get_reranker():
    global _reranker, _rerank_tokenizer
    with _load_lock:
        if _reranker is None:
            device = _device(cfg.RERANK_DEVICE)
            dtype = torch.float32 if device.startswith("cpu") else getattr(torch, cfg.RERANK_DTYPE)
            _reranker, _rerank_tokenizer = _load_reranker(device, dtype)
        return _reranker


def _move_reranker_to_cpu():
    """Tai lai reranker tren CPU o fp32 (ban bf16 dang nam tren GPU)."""
    global _reranker, _rerank_tokenizer, _reranker_on_cpu
    with _load_lock:
        _reranker = None
        torch.cuda.empty_cache()
        _reranker, _rerank_tokenizer = _load_reranker("cpu", torch.float32)
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
    """Ma hoa bang MOT lan encode(): tra ve (dense_vecs [n, 1024] da chuan hoa, lexical_weights list[dict])."""
    embedder = get_embedder()

    def run(bs):
        return embedder.encode(
            texts,
            batch_size=bs,
            max_length=cfg.EMBED_MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
        )

    with _infer_lock:
        out = _retry_on_oom(run, batch_size or cfg.EMBED_BATCH_SIZE)
    return out["dense_vecs"], out["lexical_weights"]


def to_sparse_vector(lexical_weights):
    """{token_id(str): weight} cua BGE-M3 -> SparseVector cua Qdrant (chi so tang dan)."""
    items = sorted((int(k), float(v)) for k, v in lexical_weights.items())
    return models.SparseVector(indices=[k for k, _ in items], values=[v for _, v in items])


def _truncate(text):
    """Cat doan ve toi da RERANK_MAX_LENGTH token (model Jina chi cat o 8192 token/doan, qua lon cho VRAM 4GB)."""
    ids = _rerank_tokenizer(
        text, add_special_tokens=False, truncation=True, max_length=cfg.RERANK_MAX_LENGTH
    )["input_ids"]
    return _rerank_tokenizer.decode(ids) if len(ids) >= cfg.RERANK_MAX_LENGTH else text


def rerank_scores(question, passages):
    """Cham diem listwise bang jina-reranker-v3.5: cau hoi va ca danh sach doan vao MOT prompt nen diem (cosine, -1..1,
    cao hon = lien quan hon) chi so sanh duoc giua cac doan cung lan goi. Tra ve theo dung thu tu cua `passages`."""
    if not passages:
        return []

    def run(group_size):
        model = get_reranker()
        scores = []
        for start in range(0, len(passages), group_size):
            group = [_truncate(p) for p in passages[start:start + group_size]]
            # rerank() tra ve da sap giam dan, kem `index` = vi tri trong `group` -> xep lai ve thu tu goc.
            by_index = {int(r["index"]): float(r["relevance_score"]) for r in model.rerank(question, group)}
            scores.extend(by_index[i] for i in range(len(group)))
        return scores

    group_size = min(cfg.RERANK_BATCH_SIZE, len(passages))
    with _infer_lock:
        try:
            return _retry_on_oom(run, group_size)
        except torch.cuda.OutOfMemoryError:
            if not cfg.RERANK_CPU_ON_OOM or _reranker_on_cpu:
                raise
        # Ra khoi khoi except roi moi chuyen sang CPU: traceback cua loi OOM con giu model GPU, chuyen trong khoi
        # except thi VRAM chua duoc giai phong.
        _move_reranker_to_cpu()
        return run(group_size)
