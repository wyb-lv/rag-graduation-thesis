"""Cau hinh tap trung cho pipeline RAG hybrid (BGE-M3 dense+sparse -> Qdrant RRF -> reranker -> Groq Qwen).

Bon bien doc tu moi truong (file .env, `environment` cua docker-compose, ENV cua Dockerfile):
GROQ_API_KEY, GROQ_QWEN_MODEL, QDRANT_URL, GROQ_CACHE_DIR; cong them cac bien ROUTER_* / SESSION_* / GEMINI_API_KEY / MONGODB_*
o cuoi file (router LLM, session va luu tru hoi thoai, deu tat mac dinh). Moi gia tri khac la hang so, sua truc tiep o day
(container phai build lai: `docker-compose up -d --build ai-service`).
"""
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


def _env(name, default, cast=str):
    raw = os.environ.get(name, "").strip()
    return cast(raw) if raw else default


def _flag(raw):
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"gia tri bool khong hop le: '{raw}' (dung true/false)")


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
HYBRID_LEG_VECTORS = {"dense": DENSE_VECTOR_NAME, "sparse": SPARSE_VECTOR_NAME}
HYBRID_LEG_NAMES = tuple(HYBRID_LEG_VECTORS)

# Cac nhanh cua mode `hybrid` / `hybrid_rerank`: moi nhanh cua moi cau (cau goc va cau viet lai) la mot danh sach, tat ca duoc hop
# nhat bang MOT lan RRF phia Qdrant (gop trung theo id). Mode `dense` luon chi dung dense.
HYBRID_LEGS = ("dense", "sparse")

# Chi mode `dense`: cosine toi thieu de coi cau hoi la co trong tai lieu (do tren bge-m3 + quy_che.pdf:
# cau trong pham vi >= 0.5, cau ngoai pham vi <= 0.56). Diem RRF khong phai cosine nen khong dung nguong nay.
SCORE_THRESHOLD = 0.5
# Mode hybrid_rerank: diem rerank (cosine cua jina, -1..1) toi thieu de doan van duoc coi la lien quan. Dat -1 de tat.
# Do lai 2026-10 tren kho 1297 chunk / 77 van ban (10 cau trong pham vi, 9 cau lac de): cau hoi NGAN ("Hoc phi bao nhieu?")
# bi reranker nen diem xuong, doan dung chi 0.01-0.06 nen nguong 0.1 cat con 0-1 doan va LLM tra "khong chua thong tin";
# them ten truong vao cau hoi chi la doc them chu (diem len, nhung xep hang lai ve doan chung chung, va cau lac de cung vuot
# nguong) nen KHONG la cach sua. Cau lac de cao nhat -0.032, doan tot nhat cua cau trong pham vi >= 0.2 -> 0.0 giu du doan
# ma van chan cau lac de. Doi model/corpus thi do lai (scripts/calibrate_rerank.py).
RERANK_MIN_SCORE = 0.0

# ---- Guard (guard.py): quet doan trich, prompt co rao + canary, guard dau ra; cau tra loi co dinh (small_talk.py) cho router ----
GUARD_ENABLED = True  # False: tat quet doan trich, prompt co rao, guard dau ra va chuan hoa cau hoi
GUARD_MODE = "enforce"  # "log_only": chi ghi log, khong loai doan / chan dau ra
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

OFF_TOPIC_ANSWER = (
    "Tôi chỉ hỗ trợ tra cứu quy chế, quy định, thông báo, biểu mẫu và hướng dẫn của Trường Đại học Hoa Sen. "
    "Bạn vui lòng đặt lại câu hỏi theo một chủ đề quy định của trường nhé."
)
CLARIFY_ANSWER_TOPIC = "Bạn muốn hỏi về quy định hoặc chủ đề nào của trường (ví dụ học bổng, học phí, đăng ký học phần)?"
CLARIFY_ANSWER_AUDIENCE = "Bạn là sinh viên khóa nào hoặc thuộc chương trình đào tạo nào để tôi tra cứu đúng quy định áp dụng?"
CLARIFY_ANSWER_TIME = "Bạn muốn hỏi cho học kỳ hoặc năm học nào?"

# ---- Router LLM (router_llm.py): Gemini (REST generateContent) tu suy luan chao hoi / tan cong / ngoai pham vi / hoi lai / RAG, khong con
# luat tu khoa nao. Tat hoac loi thi cau hoi di thang vao RAG. Tat mac dinh ----
ROUTER_ENABLED = _flag(os.environ.get("ROUTER_ENABLED") or "false")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # gui o header x-goog-api-key; khong bao gio log
# Id model Gemini, vi du gemini-3.5-flash-lite (ban stable, nhanh nhat cua 3.5; khong dung ban *-preview).
# Tien to `gemini/` kieu cu (thoi LiteLLM) van duoc chap nhan va bi bo.
ROUTER_MODEL = (os.environ.get("GEMINI_MODEL") or "").removeprefix("gemini/")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"  # {} = ROUTER_MODEL
# Do 2026-10 voi gemini-3.5-flash-lite: p50 ~2.4s, p95 ~8.8s moi lan goi, nen 10s. Xau nhat truoc khi roi vao RAG: (1 + ROUTER_MAX_RETRIES) x nay.
ROUTER_TIMEOUT_S = _env("ROUTER_TIMEOUT_S", 10.0, float)
ROUTER_MAX_RETRIES = _env("ROUTER_MAX_RETRIES", 1, int)
# Gioi han cua tien trinh nay (khong phai cua Google). 15 RPM la con so cu LiteLLM ghi cho gemini-3.5-flash-lite; Google khong cong bo
# free tier tren trang rate-limits ("xem trong AI Studio"), nen doi theo gia tri hien trong AI Studio cua project.
ROUTER_RPM = _env("ROUTER_RPM", 15, int)
ROUTER_MIN_CONFIDENCE = _env("ROUTER_MIN_CONFIDENCE", 0.5, float)  # y dinh co dinh duoi muc nay -> rag
ROUTER_BLOCK_CONFIDENCE = _env("ROUTER_BLOCK_CONFIDENCE", 0.7, float)  # injection tu muc nay tro len -> tu choi
ROUTER_OFFTOPIC_CONFIDENCE = _env("ROUTER_OFFTOPIC_CONFIDENCE", 0.8, float)
ROUTER_CLARIFY_CONFIDENCE = _env("ROUTER_CLARIFY_CONFIDENCE", 0.7, float)
ROUTER_CLARIFY_ENABLED = _flag(os.environ.get("ROUTER_CLARIFY_ENABLED") or "true")  # false: clarify luon thanh rag (do ablation)
ROUTER_HISTORY_TURNS = _env("ROUTER_HISTORY_TURNS", 3, int)
ROUTER_MAX_TOKENS = 512  # JSON that ~100 token; 512 chua cho token "suy nghi" (tinh vao max_output_tokens)
# Do 2026-10 voi gemini-3.5-flash-lite + structured output: "none" (= thinkingLevel minimal) lam standalone_query lap vo han tai
# ~500 token, 7/8 lan JSON bi cat, ~5s; "low" cho JSON hop le ~100 token, ~2s. Chi ha xuong "none" khi doi model va do lai.
ROUTER_REASONING_EFFORT = _env("ROUTER_REASONING_EFFORT", "low")
ROUTER_QUERY_MAX_CHARS = 300  # standalone_query dai hon bi cat
# Moi tin nhan lich su gui cho router bi cat con ngan nay. 1500: du thay cac muc 1, 2, 3... cua mot cau tra loi dai, de cau hoi kieu
# "giai thich ro dieu 2" tro duoc vao muc cua cau tra loi truoc (300 chi thay duoc muc 1 nen router hieu thanh Dieu 2 cua van ban)
ROUTER_HISTORY_MSG_CHARS = 1500
ROUTER_CACHE_SIZE = 512
ROUTER_CACHE_TTL_S = 3600

# ---- Session (session.py): lich su hoi thoai ngan han theo session_id (LlamaIndex Memory). Tat mac dinh ----
SESSION_ENABLED = _flag(os.environ.get("SESSION_ENABLED") or "false")
SESSION_STORE = _env("SESSION_STORE", "memory")  # memory: SQLite trong RAM, mat khi restart | sqlite: file, can volume
SESSION_DB_URI = _env("SESSION_DB_URI", "")  # vd sqlite+aiosqlite:////data/sessions/sessions.db (chi khi SESSION_STORE=sqlite)
SESSION_TTL_S = _env("SESSION_TTL_S", 1800, int)
SESSION_MAX_SESSIONS = _env("SESSION_MAX_SESSIONS", 200, int)
SESSION_MAX_TURNS = _env("SESSION_MAX_TURNS", 6, int)
SESSION_MAX_HISTORY_TOKENS = _env("SESSION_MAX_HISTORY_TOKENS", 1500, int)  # van nho vi Groq chi co ~8K TPM; 1500 chua du mot cau tra loi dai + cau hoi
SESSION_CHARS_PER_TOKEN = 2.0  # dem token bao thu cho tieng Viet (co dau): it ky tu / token hon uoc luong 2.5 cua Groq
SESSION_MESSAGE_MAX_CHARS = 1500  # moi tin nhan lich su dua vao prompt bi cat con ngan nay (cau tra loi dai khong chiem het ngan sach; xem ROUTER_HISTORY_MSG_CHARS)
SESSION_USER_HEADER = "X-User-Id"  # gateway gan sau khi xac thuc JWT (JwtAuthFilter); co header nay thi khoa session = user:session

# ---- Conversations (conversations.py): luu tru lau dai moi cuoc hoi thoai trong MongoDB (khac session: lich su ngan han cho prompt, het
# han sau SESSION_TTL_S). Bat khi co MONGODB_URI va SESSION_ENABLED (session_id la khoa cua cuoc hoi thoai) ----
MONGODB_URI = _env("MONGODB_URI", "")  # vd mongodb://user:pass@mongodb:27017/?authSource=admin
MONGODB_DB = _env("MONGODB_DB", "hsu_chatbot")
MONGODB_TIMEOUT_MS = _env("MONGODB_TIMEOUT_MS", 2000, int)  # Mongo chet thi moi luot hoi dap cham them toi chung nay
CONVERSATIONS_ENABLED = bool(MONGODB_URI)
CONVERSATIONS_LIST_MAX = 200
CONVERSATION_TITLE_CHARS = 80  # tieu de = cau hoi dau tien, bi cat

if ROUTER_ENABLED and not ROUTER_MODEL:
    raise ValueError("ROUTER_MODEL (bien moi truong GEMINI_MODEL) bat buoc khi ROUTER_ENABLED, vd gemini-3.5-flash-lite")
if SESSION_STORE not in ("memory", "sqlite"):
    raise ValueError(f"SESSION_STORE phai la 'memory' hoac 'sqlite', nhan duoc '{SESSION_STORE}'")
if SESSION_ENABLED and SESSION_STORE == "sqlite" and not SESSION_DB_URI.startswith("sqlite+aiosqlite:"):
    raise ValueError("SESSION_STORE=sqlite can SESSION_DB_URI dang sqlite+aiosqlite:///duong/dan/sessions.db")

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

# ---- Viet lai cau truy van (rag/query_rewrite.py): Qwen viet lai cau tim kiem, tim dense + sparse cho CA HAI cau roi RRF mot lan tren
# moi danh sach truoc khi rerank. Them mot lan goi Groq cho moi cau hoi RAG (chia doi ngan sach GROQ_RPM). Chi mode hybrid / hybrid_rerank.
# Bat mac dinh; dat QUERY_REWRITE_ENABLED=false de tat (vd khi het quota Groq) ----
QUERY_REWRITE_ENABLED = _flag(os.environ.get("QUERY_REWRITE_ENABLED") or "true")
QUERY_REWRITE_MAX_CHARS = 300  # cau viet lai dai hon bi cat

# ---- Voice (STT & TTS) ----
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "vi")
# Kích thước mỗi cửa sổ phân tích năng lượng âm thanh (20ms là chuẩn DSP cho tiếng nói)
VAD_FRAME_MS = int(os.environ.get("VAD_FRAME_MS", "20"))
# Ngưỡng năng lượng RMS tối thiểu: Khung âm thanh có năng lượng >= 0.01 được coi là có tiếng người
VAD_ENERGY_THRESHOLD = float(os.environ.get("VAD_ENERGY_THRESHOLD", "0.01"))
STT_VAD_ENERGY_THRESHOLD_DB = float(os.environ.get("STT_VAD_ENERGY_THRESHOLD_DB", "-38.0"))
# Thời lượng tiếng nói tối thiểu hợp lệ (250ms). Tiếng click chuột, tiếng thở ngắn < 250ms sẽ bị loại bỏ
VAD_MIN_SPEECH_MS = int(os.environ.get("VAD_MIN_SPEECH_MS", "250"))
# Giọng đọc Neural mặc định cho tiếng Việt (Hoài My - Nữ miền Bắc, truyền cảm và rõ chữ)
TTS_VOICE = os.environ.get("TTS_VOICE", "vi-VN-HoaiMyNeural")
TTS_RATE = os.environ.get("TTS_RATE", "+0%")

# Giới hạn dung lượng file âm thanh tối đa (5MB ~ 2.5 phút WAV 16kHz) để chống tràn RAM/OOM
MAX_AUDIO_BYTES = int(os.environ.get("MAX_AUDIO_BYTES", str(5 * 1024 * 1024)))
