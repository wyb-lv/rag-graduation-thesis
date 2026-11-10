"""Cau hinh tap trung cho pipeline RAG hybrid (BGE-M3 dense + BM25 -> Qdrant RRF -> reranker -> Groq Qwen).

Bon bien doc tu moi truong (file .env, `environment` cua docker-compose, ENV cua Dockerfile):
GROQ_API_KEY, GROQ_QWEN_MODEL, QDRANT_URL, GROQ_CACHE_DIR; cong them cac bien ROUTER_* / SESSION_* / GEMINI_API_KEY / MONGODB_*
o cuoi file (router LLM, session va luu tru hoi thoai; khong co cong tac bat / tat nao, tat ca luon chay). Moi gia tri khac la hang so, sua truc tiep o day
(container phai build lai: `docker-compose up -d --build ai-service`).
"""
import os

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)


def _env(name, default, cast=str):
    raw = os.environ.get(name, "").strip()
    return cast(raw) if raw else default


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

# Model luon duoc nap + chay thu 1 lan ngay khi server khoi dong (thread nen), de cau hoi dau tien khong phai cho ~16s.
# request /ask den luc dang warm-up doi toi da chung nay roi moi chay tiep (warm-up ket thuc som hon thi di ngay). Phai du cho lan tai model dau tien
# (~3GB) + khoi tao CUDA + bien dich triton: het gio som thi router / viet lai chay giua luc tai, timeout, con retrieve van phai doi model nap xong.
WARMUP_WAIT_S = _env("WARMUP_WAIT_S", 900.0, float)

# ---- Qdrant ----
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# dense (BGE-M3) + bm25. Collection `quy_dinh_hybrid` cu khong bi dong toi: chep sang day bang
# `python -m scripts.migrate_add_bm25` (khong can GPU, dense chep nguyen, bm25 tinh tu payload `text`).
QDRANT_COLLECTION = "quy_dinh_hybrid_v2"
# Collection dense-only cu: chi doc cau hinh HNSW tu day, khong bao gio ghi/xoa.
QDRANT_LEGACY_COLLECTION = "paragraphs"
DENSE_VECTOR_NAME = "dense"
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
TOP_K_RERANK = 5  # so doan dua vao ngu canh LLM

# Ten nhanh -> ten vector trong collection Qdrant
HYBRID_LEG_VECTORS = {"dense": DENSE_VECTOR_NAME, "bm25": BM25_VECTOR_NAME}

# Cac nhanh cua pipeline hybrid (luon roi rerank bang jina-reranker): moi nhanh cua moi cau (cau goc va cau viet lai) la mot danh sach,
# tat ca duoc hop nhat bang MOT lan RRF phia Qdrant (gop trung theo id).
HYBRID_LEGS = ("dense", "bm25")

# ---- BM25 (nhanh lexical, xem bm25.py). Doi tokenizer/K1/B/AVG_LEN thi phai nap lai (reindex) bm25 cua collection ----
BM25_TOKENIZERS = ("syllable", "syllable_bigram")
BM25_TOKENIZER = "syllable_bigram"
BM25_K1 = 1.2
BM25_B = 0.75
# So token BM25 trung binh moi chunk, do tren kho that (1319 chunk / 78 van ban, 2026-11-07). Trong so cua chunk luu o Qdrant da gom
# do dai chunk / gia tri nay, nen doi o day thi phai nap lai bm25: scripts/migrate_add_bm25.py do lai kho va in gia tri moi.
BM25_AVG_LEN = 370.13

# Diem rerank (cosine cua jina, -1..1) toi thieu de doan van duoc coi la lien quan. Dat -1 de tat.
# Do lai 2026-10 tren kho 1297 chunk / 77 van ban (10 cau trong pham vi, 9 cau lac de): cau hoi NGAN ("Hoc phi bao nhieu?")
# bi reranker nen diem xuong, doan dung chi 0.01-0.06 nen nguong 0.1 cat con 0-1 doan va LLM tra "khong chua thong tin";
# them ten truong vao cau hoi chi la doc them chu (diem len, nhung xep hang lai ve doan chung chung, va cau lac de cung vuot
# nguong) nen KHONG la cach sua. Cau lac de cao nhat -0.032, doan tot nhat cua cau trong pham vi >= 0.2 -> 0.0 giu du doan
# ma van chan cau lac de. Doi model/corpus thi do lai (scripts/calibrate_rerank.py).
RERANK_MIN_SCORE = 0.0

# ---- Guard (guard.py): quet doan trich, prompt co rao + canary, guard dau ra; cau tra loi co dinh (small_talk.py) cho router ----
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
# luat tu khoa nao. Luon bat; loi (timeout, quota, thieu GEMINI_API_KEY / GEMINI_MODEL) thi cau hoi di thang vao RAG ----
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

# ---- Session (session.py): lich su hoi thoai ngan han theo session_id (LlamaIndex Memory). Luon bat ----
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
# han sau SESSION_TTL_S). Luon bat (session_id la khoa cua cuoc hoi thoai) ----
MONGODB_URI = _env("MONGODB_URI", "mongodb://localhost:27017")  # vd mongodb://user:pass@mongodb:27017/?authSource=admin
MONGODB_DB = _env("MONGODB_DB", "hsu_chatbot")
MONGODB_TIMEOUT_MS = _env("MONGODB_TIMEOUT_MS", 2000, int)  # Mongo chet thi moi luot hoi dap cham them toi chung nay
CONVERSATIONS_LIST_MAX = 200
CONVERSATION_TITLE_CHARS = 80  # tieu de = cau hoi dau tien, bi cat

if SESSION_STORE not in ("memory", "sqlite"):
    raise ValueError(f"SESSION_STORE phai la 'memory' hoac 'sqlite', nhan duoc '{SESSION_STORE}'")
if SESSION_STORE == "sqlite" and not SESSION_DB_URI.startswith("sqlite+aiosqlite:"):
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
# Mang cua container hay roi goi SYN toi api.groq.com (do 2026-11: 17/45 lan ket noi cham 1-16 s theo chuoi truyen lai 1/3/7/15 s, Google thi khong).
# Ket noi TCP/TLS toi Groq bi cat o GROQ_CONNECT_TIMEOUT (moi dia chi IP) roi tu ket noi lai toi GROQ_CONNECT_RETRIES lan; chi tac dong luc
# DANG thiet lap ket noi, request da gui khong bao gio bi gui lai. Truoc day timeout 60 s ap dung ca cho buoc ket noi.
GROQ_CONNECT_TIMEOUT = 2.0
GROQ_CONNECT_RETRIES = 3
# Giu ket noi toi Groq song lau (httpx mac dinh dong sau 5 s roi) de cac cau hoi lien tiep dung lai ket noi thay vi bat tay lai qua duong dang roi goi.
# Do trong container, 25 lan viet lai cach nhau 6 s: loi 2 -> 0, trung vi 1,70 s -> 0,58 s, so lan > 1,5 s 15 -> 3.
GROQ_KEEPALIVE_S = 120.0
GROQ_RPM = 30
GROQ_TPM = 8000
# Uoc luong so token tu so ky tu de giu cho truoc khi goi; sau khi co usage that thi thay bang so that.
GROQ_CHARS_PER_TOKEN = 2.5
GROQ_MAX_RETRIES = 5
GROQ_BACKOFF_BASE = 2.0  # giay; nhan doi moi lan thu lai
GROQ_BACKOFF_MAX = 60.0
GROQ_CACHE_DIR = os.environ.get("GROQ_CACHE_DIR", os.path.join(BASE_DIR, ".groq_cache"))

# ---- Voice (speech/stt.py & speech/tts.py) ----
STT_MODEL = _env("STT_MODEL", _env("GROQ_STT_MODEL", "whisper-large-v3"))
STT_LANGUAGE = _env("STT_LANGUAGE", "vi")
STT_VAD_ENERGY_THRESHOLD_DB = _env("STT_VAD_ENERGY_THRESHOLD_DB", -38.0, float)
TTS_VOICE = _env("TTS_VOICE", "vi-VN-HoaiMyNeural")
TTS_RATE = _env("TTS_RATE", "+0%")

EDGE_TTS_ENABLED = _env("EDGE_TTS_ENABLED", False, lambda raw: raw.lower() in ("1", "true", "yes", "on"))

# Giới hạn dung lượng file âm thanh tối đa (5MB ~ 2.5 phút WAV 16kHz) để chống tràn RAM/OOM
MAX_AUDIO_BYTES = _env("MAX_AUDIO_BYTES", 5 * 1024 * 1024, int)

# ---- Viet lai cau truy van (rag/query_rewrite.py): Qwen viet lai cau tim kiem, tim dense + bm25 cho CA HAI cau roi RRF mot lan tren
# moi danh sach truoc khi rerank. Them mot lan goi Groq cho moi cau hoi RAG (chia doi ngan sach GROQ_RPM).
# Luon bat ----
QUERY_REWRITE_MAX_CHARS = 300  # cau viet lai dai hon bi cat
QUERY_REWRITE_TIMEOUT_S = _env("QUERY_REWRITE_TIMEOUT_S", 6.0, float)  # cau hoi cho buoc viet lai TOI DA chung nay (ket noi + cho gioi han toc do + thu lai), cung la timeout moi request Groq cua no. Do thuc te trung vi 0,6 s, p90 2,6 s

# ---- OCR (app/ocr): consumer file-service.ocr.jobs. PDF co chu hien thi va DOCX doc thang (khong the sai chinh ta); chi trang scan / anh moi qua PaddleOCR-VL-1.6 ----
# Pipeline `paddleocr[doc-parser]` (PP-DocLayoutV3 + VLM 0.9B) chay o container chinh thuc `paddleocr-vl-api` (docker-compose.yml)
OCR_VL_URL = os.environ.get("OCR_VL_URL", "http://localhost:8080").rstrip("/")
OCR_VL_TIMEOUT_S = _env("OCR_VL_TIMEOUT_S", 1800.0, float)  # toi da cho MOT lo trang; service doc cham theo so trang va GPU
OCR_PAGES_PER_REQUEST = _env("OCR_PAGES_PER_REQUEST", 8, int)  # so trang scan gui trong mot request (giu request / thoi gian cho vua phai)
OCR_SCAN_IMAGE_COVER = _env("OCR_SCAN_IMAGE_COVER", 0.5, float)  # trang co anh phu tu ti le nay tro len coi la scan, bo qua lop chu
# Nhan khoi layout khong dua vao Markdown (so trang, dau / chan trang lap lai o moi trang); mac dinh cua PaddleOCR-VL bo them `footnote`
OCR_IGNORE_LABELS = _env("OCR_IGNORE_LABELS", ["number", "header", "header_image", "footer", "footer_image", "aside_text"],
                          lambda raw: [x.strip() for x in raw.split(",") if x.strip()])

# Thêm vào ngay dưới dòng MAX_AUDIO_BYTES trong ai-service/app/config.py:
CALL_GREETING_TEXT = _env(
    "CALL_GREETING_TEXT",
    "Xin chào! Mình là trợ lý học vụ Đại học Hoa Sen. Bạn có câu hỏi nào cần giải đáp không?",
)
CALL_NO_SPEECH_TEXT = _env(
    "CALL_NO_SPEECH_TEXT",
    "Mình chưa nghe rõ câu hỏi, bạn vui lòng nói lại giúp mình nhé.",
)
