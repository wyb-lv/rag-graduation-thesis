"""Cau hinh tap trung cho pipeline RAG hybrid (BGE-M3 dense+sparse -> Qdrant RRF -> reranker -> Groq Qwen).

Moi gia tri co the doi bang bien moi truong / file .env, khong can sua code.
"""
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ---- Du lieu nguon (do export_model.py tao ra) ----
PARAGRAPHS_PATH = os.path.join(BASE_DIR, "data", "paragraphs.txt")

# ---- Model (tai mot lan moi tien trinh, xem encoders.py) ----
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
# Reranker listwise nap qua transformers bang jina_reranker_modeling.py (ban sao da duyet, khong trust_remote_code).
# Giay phep CC BY-NC 4.0 (phi thuong mai).
RERANK_MODEL = os.environ.get("RERANK_MODEL", "jinaai/jina-reranker-v3.5")
# Weights/config/tokenizer lay dung revision nay, va jina_reranker_modeling.py la ban sao modeling.py o revision nay.
# Doi revision = duyet lai modeling.py moi va vendor lai.
RERANK_REVISION = os.environ.get(
    "RERANK_REVISION",
    "e8a93f33f0b22108f8c2364f8484ce3422552fbc" if RERANK_MODEL == "jinaai/jina-reranker-v3.5" else "",
)
RERANK_DTYPE = os.environ.get("RERANK_DTYPE", "bfloat16")  # dtype goc cua checkpoint; tren CPU luon dung float32
USE_FP16 = _env_bool("USE_FP16", True)  # chi cho BGE-M3
# Thiet bi: "cuda:0", "cpu"... De trong = tu chon (cuda:0 neu co GPU, khong thi cpu).
EMBED_DEVICE = os.environ.get("EMBED_DEVICE") or None
# Dat RERANK_DEVICE=cpu de chay reranker bang CPU, nhuong ~1.1GB VRAM cho BGE-M3 (RTX 3050 4GB).
RERANK_DEVICE = os.environ.get("RERANK_DEVICE") or None
# Khi CUDA OOM ngay ca khi chi cham 1 doan/lan: chuyen reranker sang CPU thay vi bao loi.
RERANK_CPU_ON_OOM = _env_bool("RERANK_CPU_ON_OOM", False)
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "16"))
EMBED_MAX_LENGTH = int(os.environ.get("EMBED_MAX_LENGTH", "512"))
# Reranker listwise: cau hoi + ca danh sach doan vao MOT prompt (Jina: toi da 64 doan) nen "batch" la so doan cham
# chung. Nhieu hon (hoac khi OOM) thi chia nhom; diem giua cac nhom khong cung ngu canh nen chi so sanh gan dung.
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "64"))
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))  # token toi da moi doan (cat bot), de vua VRAM 4GB

# Nap model + chay thu 1 lan ngay khi server khoi dong (thread nen), de cau hoi dau tien khong phai cho ~16s.
WARMUP = _env_bool("WARMUP", True)

# ---- Qdrant ----
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "quy_dinh_hybrid")
# Collection dense-only cu: chi doc cau hinh HNSW tu day, khong bao gio ghi/xoa.
QDRANT_LEGACY_COLLECTION = os.environ.get("QDRANT_LEGACY_COLLECTION", "paragraphs")
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
EMBED_DIM = 1024
# Chi dung khi khong doc duoc collection cu: cung cau hinh da benchmark truoc day.
HNSW_M = 32
HNSW_EF_CONSTRUCT = 200
HNSW_EF = int(os.environ.get("HNSW_EF", "128"))  # efSearch: lon hon -> recall cao hon, cham hon
# Mac dinh Qdrant chi dung do thi HNSW khi segment >= 20000 KB, nho hon thi quet tuan tu.
# Ha nguong xuong de tap nho nay van duoc dung do thi HNSW that.
INDEXING_THRESHOLD = 1
UPSERT_BATCH_SIZE = int(os.environ.get("UPSERT_BATCH_SIZE", "64"))

# Ten field payload dung cho index va filter. Doi o day neu payload dung ten khac.
FIELD_KHO = os.environ.get("PAYLOAD_FIELD_KHO", "kho")
FIELD_DOI_TUONG = os.environ.get("PAYLOAD_FIELD_DOI_TUONG", "doi_tuong")
FIELD_HIEU_LUC_TU = os.environ.get("PAYLOAD_FIELD_HIEU_LUC_TU", "hieu_luc_tu")
FIELD_HIEU_LUC_DEN = os.environ.get("PAYLOAD_FIELD_HIEU_LUC_DEN", "hieu_luc_den")

# ---- Pipeline ----
TOP_K_RETRIEVAL = int(os.environ.get("TOP_K_RETRIEVAL", "20"))
TOP_K_RERANK = int(os.environ.get("TOP_K_RERANK", "5"))  # so doan dua vao ngu canh LLM (moi mode)
# dense: chi vector `dense` | hybrid: dense + sparse (RRF) | hybrid_rerank: hybrid + jina-reranker
RAG_MODE = os.environ.get("RAG_MODE", "hybrid_rerank")
RAG_MODES = ("dense", "hybrid", "hybrid_rerank")
# Chi mode `dense`: cosine toi thieu de coi cau hoi la co trong tai lieu (do tren bge-m3 + quy_che.pdf:
# cau trong pham vi >= 0.5, cau ngoai pham vi <= 0.56). Diem RRF khong phai cosine nen khong dung nguong nay.
SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.5"))
# Mode hybrid_rerank: diem rerank (cosine cua jina, -1..1) toi thieu de doan van duoc coi la lien quan. Do tren
# quy_che.pdf (22 cau): doan dung o cau trong pham vi >= 0.268, cau lac de <= 0.075, cau "gan chu de" nhung khong co
# dap an 0.17-0.23 (van toi LLM, prompt se tu choi). Dat -1 de tat. Doi model/corpus thi do lai.
RERANK_MIN_SCORE = float(os.environ.get("RERANK_MIN_SCORE", "0.1"))

# ---- Groq ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("API_KEY")
# Bat buoc: ID model Qwen tren Groq, vi du qwen/qwen3.8-27b (doc console.groq.com/docs/models).
GROQ_QWEN_MODEL = os.environ.get("GROQ_QWEN_MODEL")
# "none" tat reasoning (Qwen tren Groq ho tro none/default/low/medium/high). De trong = khong gui tham so.
GROQ_REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "none")
GROQ_MAX_TOKENS = int(os.environ.get("GROQ_MAX_TOKENS", "1024"))
GROQ_TEMPERATURE = float(os.environ.get("GROQ_TEMPERATURE", "0"))
GROQ_TIMEOUT = float(os.environ.get("GROQ_TIMEOUT", "60"))
GROQ_RPM = int(os.environ.get("GROQ_RPM", "30"))
GROQ_TPM = int(os.environ.get("GROQ_TPM", "8000"))
# Uoc luong so token tu so ky tu de giu cho truoc khi goi; sau khi co usage that thi thay bang so that.
GROQ_CHARS_PER_TOKEN = float(os.environ.get("GROQ_CHARS_PER_TOKEN", "2.5"))
GROQ_MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "5"))
GROQ_BACKOFF_BASE = float(os.environ.get("GROQ_BACKOFF_BASE", "2"))  # giay; nhan doi moi lan thu lai
GROQ_BACKOFF_MAX = float(os.environ.get("GROQ_BACKOFF_MAX", "60"))
GROQ_CACHE_ENABLED = _env_bool("GROQ_CACHE", True)
GROQ_CACHE_DIR = os.environ.get("GROQ_CACHE_DIR", os.path.join(BASE_DIR, ".groq_cache"))

# ---- Voice (STT & TTS) ----
# Model nhận dạng giọng nói trên Groq Cloud (hỗ trợ tiếng Việt rất tốt)
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3")
# Kích thước mỗi cửa sổ phân tích năng lượng âm thanh (20ms là chuẩn DSP cho tiếng nói)
VAD_FRAME_MS = int(os.environ.get("VAD_FRAME_MS", "20"))
# Ngưỡng năng lượng RMS tối thiểu: Khung âm thanh có năng lượng >= 0.01 được coi là có tiếng người
VAD_ENERGY_THRESHOLD = float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.01"))
# Thời lượng tiếng nói tối thiểu hợp lệ (250ms). Tiếng click chuột, tiếng thở ngắn < 250ms sẽ bị loại bỏ
VAD_MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "250"))
# Giọng đọc Neural mặc định cho tiếng Việt (Hoài My - Nữ miền Bắc, truyền cảm và rõ chữ)
TTS_VOICE = os.environ.get("TTS_VOICE", "vi-VN-HoaiMyNeural")

# Giới hạn dung lượng file âm thanh tối đa (5MB ~ 2.5 phút WAV 16kHz) để chống tràn RAM/OOM
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(5 * 1024 * 1024)))