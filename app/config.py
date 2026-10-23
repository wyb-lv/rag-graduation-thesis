"""Cau hinh tap trung cho pipeline RAG hybrid (BGE-M3 dense+sparse -> Qdrant RRF -> reranker -> Groq Qwen).

Chi bon bien doc tu moi truong (file .env, `environment` cua docker-compose, ENV cua Dockerfile):
GROQ_API_KEY, GROQ_QWEN_MODEL, QDRANT_URL, GROQ_CACHE_DIR. Moi gia tri khac la hang so, sua truc tiep o day
(container phai build lai: `docker-compose up -d --build ai-service`).
"""
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


# ---- Du lieu nguon (do export_model.py tao ra) ----
PARAGRAPHS_PATH = os.path.join(BASE_DIR, "data", "paragraphs.txt")

# ---- Model (tai mot lan moi tien trinh, xem encoders.py) ----
EMBED_MODEL = "BAAI/bge-m3"
# Reranker listwise nap qua transformers bang models/jina_reranker_modeling.py (ban sao da duyet, khong trust_remote_code).
# Giay phep CC BY-NC 4.0 (phi thuong mai).
RERANK_MODEL = "jinaai/jina-reranker-v3.5"
# Weights/config/tokenizer lay dung revision nay, va models/jina_reranker_modeling.py la ban sao modeling.py o revision nay.
# Doi revision = duyet lai modeling.py moi va vendor lai.
RERANK_REVISION = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
RERANK_DTYPE = "bfloat16"  # dtype goc cua checkpoint; tren CPU luon dung float32
USE_FP16 = True  # chi cho BGE-M3
# Thiet bi: "cuda:0", "cpu"... None = tu chon (cuda:0 neu co GPU, khong thi cpu).
EMBED_DEVICE = None
# Dat RERANK_DEVICE = "cpu" de chay reranker bang CPU, nhuong ~1.1GB VRAM cho BGE-M3 (RTX 3050 4GB).
RERANK_DEVICE = None
# Khi CUDA OOM ngay ca khi chi cham 1 doan/lan: chuyen reranker sang CPU thay vi bao loi.
RERANK_CPU_ON_OOM = False
EMBED_BATCH_SIZE = 16
EMBED_MAX_LENGTH = 512
# Reranker listwise: cau hoi + ca danh sach doan vao MOT prompt (Jina: toi da 64 doan) nen "batch" la so doan cham
# chung. Nhieu hon (hoac khi OOM) thi chia nhom; diem giua cac nhom khong cung ngu canh nen chi so sanh gan dung.
RERANK_BATCH_SIZE = 64
RERANK_MAX_LENGTH = 512  # token toi da moi doan (cat bot), de vua VRAM 4GB

# Nap model + chay thu 1 lan ngay khi server khoi dong (thread nen), de cau hoi dau tien khong phai cho ~16s.
WARMUP = True

# ---- Qdrant ----
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = "quy_dinh_hybrid"
# Collection dense-only cu: chi doc cau hinh HNSW tu day, khong bao gio ghi/xoa.
QDRANT_LEGACY_COLLECTION = "paragraphs"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
BM25_VECTOR_NAME = "bm25"
EMBED_DIM = 1024
# Chi dung khi khong doc duoc collection cu: cung cau hinh da benchmark truoc day.
HNSW_M = 32
HNSW_EF_CONSTRUCT = 200
HNSW_EF = 128  # efSearch: lon hon -> recall cao hon, cham hon
# Mac dinh Qdrant chi dung do thi HNSW khi segment >= 20000 KB, nho hon thi quet tuan tu.
# Ha nguong xuong de tap nho nay van duoc dung do thi HNSW that.
INDEXING_THRESHOLD = 1
UPSERT_BATCH_SIZE = 64

# Ten field payload dung cho index va filter. Doi o day neu payload dung ten khac.
FIELD_KHO = "kho"
FIELD_DOI_TUONG = "doi_tuong"
FIELD_HIEU_LUC_TU = "hieu_luc_tu"
FIELD_HIEU_LUC_DEN = "hieu_luc_den"
# Khoa ap dung cua van ban (so nguyen, de trong = ap dung moi khoa); loc theo khoa cua sinh vien.
FIELD_KHOA_TU = "khoa_tu"
FIELD_KHOA_DEN = "khoa_den"
# Loai van ban (quy_che, thong_bao, ...) do chunker.infer_metadata ghi san; loc theo truong nay qua `loai_van_ban` cua request.
FIELD_LOAI_VAN_BAN = "loai_van_ban"

# ---- Pipeline ----
TOP_K_RETRIEVAL = 20
TOP_K_RERANK = 5  # so doan dua vao ngu canh LLM (moi mode)
# dense: chi vector `dense` | hybrid: dense + sparse (RRF) | hybrid_rerank: hybrid + jina-reranker
RAG_MODE = "hybrid_rerank"
RAG_MODES = ("dense", "hybrid", "hybrid_rerank")


# Ten nhanh -> ten vector trong collection Qdrant
HYBRID_LEG_VECTORS = {"dense": DENSE_VECTOR_NAME, "sparse": SPARSE_VECTOR_NAME, "bm25": BM25_VECTOR_NAME}
HYBRID_LEG_NAMES = tuple(HYBRID_LEG_VECTORS)

# Cac nhanh prefetch cua mode `hybrid` / `hybrid_rerank`, hop nhat bang RRF phia Qdrant. Mode `dense` luon chi dung dense.
# Mac dinh dense,sparse (nhu truoc khi co BM25). Them "bm25" can collection co vector bm25: chay scripts/migrate_add_bm25.py.
HYBRID_LEGS = ("dense", "sparse")

# ---- BM25 (nhanh lexical, xem bm25.py). Doi tokenizer/K1/B/AVG_LEN thi phai nap lai (reindex) bm25 cua collection ----
BM25_TOKENIZERS = ("syllable", "syllable_bigram", "pyvi")
BM25_TOKENIZER = "syllable_bigram"
BM25_K1 = 1.2
BM25_B = 0.75
# So token BM25 trung binh moi chunk. 120 chi la gia tri tam: scripts/migrate_add_bm25.py do gia tri that tren du lieu
# va in ra de dat vao day (trong so cua chunk luu o Qdrant da gom do dai chunk / gia tri nay).
BM25_AVG_LEN = 120.0
# Chi mode `dense`: cosine toi thieu de coi cau hoi la co trong tai lieu (do tren bge-m3 + quy_che.pdf:
# cau trong pham vi >= 0.5, cau ngoai pham vi <= 0.56). Diem RRF khong phai cosine nen khong dung nguong nay.
SCORE_THRESHOLD = 0.5
# Mode hybrid_rerank: diem rerank (cosine cua jina, -1..1) toi thieu de doan van duoc coi la lien quan. Do tren
# quy_che.pdf (22 cau): doan dung o cau trong pham vi >= 0.268, cau lac de <= 0.075, cau "gan chu de" nhung khong co
# dap an 0.17-0.23 (van toi LLM, prompt se tu choi). Dat -1 de tat. Doi model/corpus thi do lai.
RERANK_MIN_SCORE = 0.1

# ---- Guard (guard.py, small_talk.py): chong prompt injection + tra loi co dinh cho chao hoi / hoi ve tro ly ----
GUARD_ENABLED = True  # False: hanh vi y het truoc khi co guard
GUARD_MODE = "enforce"  # "log_only": chi ghi log, khong chan/loai/sua gi
if GUARD_MODE not in ("enforce", "log_only"):  # guard so sanh == "enforce" khap noi: go sai se am tham het chan
    raise ValueError(f"GUARD_MODE phai la 'enforce' hoac 'log_only', nhan duoc '{GUARD_MODE}'")
GUARD_MAX_QUESTION_CHARS = 1000
# "" = guard.harden_prompt() suy ra tu noi dung system prompt (on dinh moi lan trien khai, khong ngau nhien theo
# tien trinh, nen cache dia cua Groq van trung). Dat tay neu muon doi canary ma khong sua prompt.
GUARD_CANARY = ""

# Cau tra loi co dinh. Giong giong dieu cua SYSTEM_PROMPT; khong nhac ten model / nha cung cap.
GUARD_REFUSAL_ANSWER = (
    "Tôi không thể thực hiện yêu cầu này. Bạn có thể hỏi tôi về quy chế, quy định, thông báo hoặc biểu mẫu "
    "của Trường Đại học Hoa Sen."
)
_ABOUT = (
    "Tôi là trợ lý tra cứu quy định của Trường Đại học Hoa Sen. Tôi tìm quy chế, quy định, thông báo, biểu mẫu và "
    "hướng dẫn trong các kho văn bản của trường, trả lời kèm trích dẫn (tên văn bản, Điều, Khoản) và nói rõ khi "
    "các văn bản không chứa câu trả lời. Với nội dung nằm ngoài các văn bản, bạn vui lòng liên hệ phòng, ban phụ "
    "trách."
)
GUARD_REPLY_GREETING = f"Xin chào! {_ABOUT} Bạn muốn tra cứu nội dung gì?"
GUARD_REPLY_CAPABILITY = f"{_ABOUT} Bạn cứ đặt câu hỏi về quy định nhé."
GUARD_REPLY_META = (
    "Tôi là trợ lý tra cứu quy định của Trường Đại học Hoa Sen và không chia sẻ thông tin cấu hình nội bộ. "
    "Bạn có thể hỏi tôi về quy chế, quy định, thông báo hoặc biểu mẫu của trường."
)
GUARD_REPLY_THANKS = "Không có gì! Bạn cần tra cứu thêm quy định nào cứ hỏi nhé."
GUARD_REPLY_GOODBYE = "Tạm biệt! Khi cần tra cứu quy định, bạn cứ quay lại nhé."

# ---- Groq ----
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
# Bat buoc: ID model Qwen tren Groq, vi du qwen/qwen3.8-27b (doc console.groq.com/docs/models).
GROQ_QWEN_MODEL = os.environ.get("GROQ_QWEN_MODEL")
# "none" tat reasoning (Qwen tren Groq ho tro none/default/low/medium/high). "" = khong gui tham so.
GROQ_REASONING_EFFORT = "none"
GROQ_MAX_TOKENS = 1024
GROQ_TEMPERATURE = 0.0
GROQ_TIMEOUT = 60.0
GROQ_RPM = 30
GROQ_TPM = 8000
# Uoc luong so token tu so ky tu de giu cho truoc khi goi; sau khi co usage that thi thay bang so that.
GROQ_CHARS_PER_TOKEN = 2.5
GROQ_MAX_RETRIES = 5
GROQ_BACKOFF_BASE = 2.0  # giay; nhan doi moi lan thu lai
GROQ_BACKOFF_MAX = 60.0
GROQ_CACHE_ENABLED = True
GROQ_CACHE_DIR = os.environ.get("GROQ_CACHE_DIR", os.path.join(BASE_DIR, ".groq_cache"))
