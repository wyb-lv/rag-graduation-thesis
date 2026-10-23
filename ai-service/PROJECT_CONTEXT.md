# AI Service – Project Context

Ngữ cảnh để người mới (hoặc AI agent) vào việc ngay với `ai-service`. Chi tiết từng endpoint xem [API.md](API.md).
Mô tả này khớp với code tại thời điểm viết; khi code đổi thì sửa file này.

## 1. Vai trò

`ai-service` là service Python (FastAPI) làm hai việc cho chatbot quy chế **Đại học Hoa Sen** (đồ án tốt nghiệp):

1. **Hỏi đáp RAG**: nhận câu hỏi, tìm đoạn quy định liên quan trong Qdrant, sinh câu trả lời có trích dẫn bằng LLM (Groq Qwen).
2. **Lập chỉ mục (embed)**: nhận văn bản đã OCR của file-service, chia đoạn, tạo vector, ghi vào Qdrant.

Không có database riêng: dữ liệu vector nằm ở Qdrant, văn bản gốc nằm ở file-service.

## 2. Vị trí trong hệ thống

```
FE (React, :5173) ──► api-gateway (:9090) ──┬─► auth-service  (:8081, gRPC :9091)
                      JWT + CORS + rate     ├─► file-service  (:8080)  ── OCR ──► RabbitMQ ──┐
                                            └─► ai-service    (:8000 trong container,        │
                                                              :8001 trên máy host) ◄─────────┘
                                                   │  embed job (queue ai-service.embed.jobs)
                                                   ├─► Qdrant (:6333)   collection quy_dinh_hybrid
                                                   └─► Groq API (LLM)
```

- FE **chỉ gọi gateway**. Route sang ai-service: `/api/ai/ask`, `/api/ai/ask/stream`, `/api/ai/health`, `/api/ai/embed[/{id}]` (gateway bỏ tiền tố `/api/ai`).
- **CORS do gateway lo.** ai-service không được thêm `CORSMiddleware` (xem mục 9).
- Cả hệ thống chạy bằng `docker-compose up -d` ở thư mục gốc repo (trên máy Windows này gõ `docker-compose`, có dấu gạch).

## 3. Luồng hỏi đáp (`POST /ask`, `POST /ask/stream`)

Code: [app/api.py](app/api.py) → [app/rag/query_model.py](app/rag/query_model.py).

**Lớp guard chạy trước và sau các bước dưới đây** (code: [app/guard.py](app/guard.py), [app/small_talk.py](app/small_talk.py); chỉ regex/chuỗi, CPU, mỗi lần gọi < 5 ms, không model, không gọi LLM). Hai endpoint không có hàm pipeline chung (`/ask` đi qua `qm.ask`, `/ask/stream` đi qua `qm.find_hits` + `qm.generate_answer_stream`), nên các điểm móc nằm ở những chỗ cả hai cùng đi qua:

- **Đầu vào** – `api._question` (chuẩn hóa) rồi `api._gate` (detector + small talk). Thứ tự: `guard.normalize_question` (NFKC, bỏ zero width/ký tự điều khiển, gộp khoảng trắng nhưng **giữ xuống dòng** để bắt `system:` đầu dòng, cắt quá `GUARD_MAX_QUESTION_CHARS`) → `guard.inspect_question` (`allow`/`suspicious`/`block` theo 6 nhóm: override, roleplay, extraction, exfiltration, delimiter, obfuscation) → `small_talk.route` (`greeting`/`thanks`/`goodbye`/`capability`/`meta`/`rag`).
  - `block` (override, extraction, exfiltration, hoặc từ 2 nhóm trở lên) → `GUARD_REFUSAL_ANSWER`, `in_scope=false`, `sources=[]`, `intent=blocked`.
  - `suspicious` (một tín hiệu yếu) → vẫn chạy RAG, nhưng `guard.strip_delimiters` bỏ role token / thẻ phân cách khỏi câu hỏi (bỏ xong mà rỗng thì coi như `block`).
  - Small talk → câu trả lời cố định từ config, `in_scope=true`, `sources=[]`, `intent=<tên>`. Chỉ khớp khi cả tin nhắn ≤ 8 từ và mỗi mệnh đề khớp trọn một cụm đã biết; "Chào thầy, cho em hỏi…" bỏ lời chào rồi đi tiếp là `rag`; không chắc thì `rag`.
  - Cả ba nhánh trên **không** gọi encoder, Qdrant, reranker hay Groq.
- **Đoạn trích** – cuối `qm.find_hits` (sau rerank + ngưỡng): `guard.drop_flagged` bỏ đoạn mà `scan_chunk` thấy override/extraction/exfiltration/delimiter (không xét roleplay, obfuscation để khỏi loại nhầm văn bản thật) và log `document_id` + nhóm để admin dọn tài liệu. Bỏ hết thì rơi vào đường `in_scope=false` bên dưới.
- **Dựng prompt** – `qm.build_messages`: mỗi đoạn là `<chunk n="i">` trong `<context>`, câu hỏi trong `<question>`; mọi thẻ `context/question/chunk` xuất hiện trong dữ liệu (đoạn, tên tài liệu, câu hỏi) bị đổi thành `[...]`. `SYSTEM_PROMPT` = `BASE_SYSTEM_PROMPT` (luật cũ, giữ nguyên) + 3 luật chống injection + canary (`guard.harden_prompt`).
- **Đầu ra** – `qm.ask` (`/ask`) và `qm.generate_answer_stream` (`/ask/stream`), sau ThinkFilter, chỉ xét text hiển thị: `guard.check_output` thất bại nếu lộ canary, ≥ 12 từ liên tiếp của system prompt (trừ câu từ chối được phép nói nguyên văn), khóa `gsk_…`, `NAME=value` của biến bí mật, hoặc đúng giá trị `GROQ_API_KEY`. `/ask` trả `GUARD_REFUSAL_ANSWER` (`in_scope=false`, `sources=[]`, `intent=blocked`); `/ask/stream` kiểm tra trên cửa sổ trượt 600 ký tự cuối + delta mới *trước khi* gửi delta, thất bại thì phát `{"event":"error"}` (không có `[DONE]`).
- `GUARD_ENABLED=false` → hành vi y hệt trước đây (prompt cũ, câu hỏi cũ, chào hỏi đi qua RAG), chỉ có thêm trường `intent="rag"`. `GUARD_MODE=log_only` → chỉ ghi log mọi quyết định chặn/loại/sửa (small talk vẫn trả lời cố định). Mỗi quyết định ghi một dòng `guard intent=… level=… categories=… mode=… q=<8 ký tự hash>`; không bao giờ log nội dung câu hỏi hay canary.

1. `_query_legs()` tạo vector truy vấn cho từng nhánh trong `HYBRID_LEGS` (mặc định `dense,sparse`; thêm `bm25` khi collection có vector đó):
   - `dense` (1024 chiều) và `sparse` (trọng số học được của BGE-M3) lấy từ **một** lần `encoders.encode_texts([question])` (GPU);
   - `bm25` do `bm25.bm25_query_vector(question)` (CPU, không dính GPU lock; truy vấn chỉ-bm25 thì không nạp BGE-M3).
   Nhánh nào không có token (câu hỏi toàn dấu câu) bị bỏ; không còn nhánh nào thì dùng dense.
2. `retrieve()`: **một** request `query_points` tới Qdrant, mỗi nhánh là một `Prefetch` (limit `TOP_K_RETRIEVAL`=20); Qdrant hợp nhất các nhánh bằng **RRF** ở phía server và trả `TOP_K_RETRIEVAL` ứng viên. **Cùng một filter** (`kho`, văn bản còn hiệu lực tại `reference_date` mặc định hôm nay, `khoa` của sinh viên, `loai_van_ban` nếu có) được gắn vào từng Prefetch, vì nhánh nào thiếu filter sẽ trả cả đoạn ngoài phạm vi. Chỉ còn một nhánh thì truy vấn thẳng nhánh đó (điểm là điểm của nhánh, không phải RRF). Mode `dense` luôn chỉ dùng dense, bỏ qua `HYBRID_LEGS`.
3. `rerank()`: jina-reranker-v3.5 chấm lại các ứng viên, giữ `TOP_K_RERANK` (5) đoạn tốt nhất.
4. `_relevant()`: lọc theo ngưỡng. Mode `hybrid_rerank` giữ đoạn có `rerank_score >= RERANK_MIN_SCORE` (0.1).
5. Không còn đoạn nào → trả `in_scope=false`, `sources=[]`, câu cố định `NO_RESULT_ANSWER`, **không gọi LLM**.
6. Còn đoạn → `build_messages()` ghép các đoạn thành context (`[Đoạn i] Tài liệu: ...`, đúng thứ tự rerank) và `groq_client` gọi Qwen với `SYSTEM_PROMPT` (trả lời tiếng Việt, mỗi ý kèm trích dẫn `[Tên văn bản, Điều X, Khoản Y]`, không bịa số Điều).

Ba mode (`mode` trong body hoặc `RAG_MODE`): `dense`, `hybrid` (các nhánh của `HYBRID_LEGS`, RRF, không rerank), `hybrid_rerank` (mặc định: như `hybrid` rồi rerank). Request có thêm `loai_van_ban` (danh sách, ví dụ `["thong_bao"]`) để chỉ tìm trong các loại văn bản đó; bỏ trống thì không lọc.

Lúc khởi động (`lifespan`), `qdrant_setup.validate_legs()` kiểm tra mọi nhánh trong `HYBRID_LEGS` có vector trong collection. Thiếu thì `RuntimeError` nêu tên nhánh và `scripts/migrate_add_bm25.py`, server không lên. Qdrant chưa sẵn sàng hoặc collection chưa tồn tại thì chỉ ghi log và vẫn khởi động.

`/ask/stream` trả SSE: `{"event":"sources"}` rồi nhiều `{"event":"delta","text":...}` rồi `data: [DONE]`. Lỗi lúc truy vấn trả mã HTTP (400/500/502); lỗi khi đang stream gửi `{"event":"error"}` (không có `[DONE]`). Câu trả lời cố định (small talk, bị chặn) dùng cùng hợp đồng: `sources` (rỗng, kèm `intent`), đúng một `delta`, `[DONE]`. Sự kiện `sources` của câu hỏi RAG **không** có khóa `intent` (giữ nguyên từng byte như trước); `/ask` luôn có `intent`.

## 4. Luồng lập chỉ mục (embed)

Code: [app/ingest/embed_consumer.py](app/ingest/embed_consumer.py) → [app/ingest/embed_service.py](app/ingest/embed_service.py) → [app/ingest/chunker.py](app/ingest/chunker.py).

1. file-service OCR xong → publish job vào exchange `file-service.embed` (routing key `embed.jobs`).
2. `embed_consumer` chạy trong **thread nền** (khởi động ở `lifespan`, chỉ khi có `RABBITMQ_URL`). Queue `ai-service.embed.jobs` là quorum queue, có dead-letter queue `ai-service.embed.jobs.dead`; lỗi xử lý thì `nack` không requeue.
3. Message: `documentId`, `source` (tên tài liệu), `kho`, `hieuLucTu`, `hieuLucDen`, `text` (tùy chọn), `action` (`UPSERT` hoặc `DELETE`). Nếu không có `text`, consumer gọi `GET {FILE_SERVICE_URL}/api/documents/{id}/text`.
4. `embed_document()`:
   - `chunker.chunk_document()` chia theo cấu trúc **Chương → Điều → khoản → điểm**; văn bản không có Điều (thông báo, biểu mẫu) thì gom theo khoản/đoạn.
   - **Xóa vector cũ theo `document_id` rồi mới ghi vector mới** (re-OCR/re-embed không bị trùng).
   - Encode theo batch (`UPSERT_BATCH_SIZE`) và upsert vào Qdrant.
   - Collection chưa có thì tạo bằng `qdrant_setup.create_collection()` (đủ `dense`, `sparse`, `bm25` và payload index).
   - Nếu collection có vector `bm25`, ghi thêm `bm25` tính từ chính `text` của chunk **bất kể `HYBRID_LEGS`**, để bật/tắt nhánh sau này không phải index lại. Collection cũ không có `bm25` thì hành xử như trước. Chunk không có token nào thì bỏ riêng vector `bm25`, các vector khác vẫn ghi.
5. Cũng gọi tay được: `POST /embed`, `DELETE /embed/{document_id}`.

### Chunker (đáng biết)
- Văn bản đầu vào là OCR nên nhiễu. `clean_lines()` bỏ số trang, header lặp, dòng mục lục, dòng toàn ký hiệu (con dấu/chữ ký).
- Regex tiêu đề cố tình **mờ** để chịu lỗi OCR ("Piéu", "Chuong"...).
- Mỗi chunk có tiền tố `[Tên tài liệu – Chương – Điều]`. Chunk > 1500 ký tự thì tách theo khoản; chunk < 150 ký tự thì gộp vào chunk kề.
- Dòng bảng Markdown (chứa `|`) được giữ nguyên là một đơn vị, không nối với dòng trước.
- `infer_metadata()` suy ra `loai_van_ban` (`quy_che`, `quy_dinh`, `thong_bao`, `bieu_mau`, `huong_dan`, `khac`) và khóa áp dụng `khoa_tu`/`khoa_den` **từ tên tài liệu** ("khóa 2024 trở về sau" → `khoa_tu=2024`).

## 5. Dữ liệu trong Qdrant

Collection `quy_dinh_hybrid` (tên đặt bằng `QDRANT_COLLECTION`) có các named vector:

| Vector | Loại | Ghi chú |
|---|---|---|
| `dense` | 1024, cosine, HNSW m=32, ef_construct=200 | BGE-M3 |
| `sparse` | sparse, **không** IDF | trọng số học được của BGE-M3 |
| `bm25` | sparse, `modifier=IDF` | BM25 do [app/bm25.py](app/bm25.py) tạo; Qdrant nhân IDF lúc truy vấn. Collection tạo trước khi có BM25 **không có** vector này |

Chỉ số sparse của `bm25` là hash 32 bit ổn định của token (blake2b, không dùng `hash()` của Python); trọng số tài liệu là `tf*(k1+1) / (tf + k1*(1 - b + b*len/avg_len))` với `len` là số token của chunk và `avg_len = BM25_AVG_LEN`.

Payload mỗi điểm: `document_id`, `source`, `text`, `paragraph_index`, `loai_van_ban`, `dieu`, `chuong`, và tùy chọn `kho`, `hieu_luc_tu`, `hieu_luc_den`, `khoa`, `khoa_tu`, `khoa_den`. Có payload index cho `kho`, `doi_tuong`, `loai_van_ban` (keyword), `hieu_luc_*` (datetime), `khoa_*` (integer) khi collection được tạo bởi `qdrant_setup.create_collection()`. Tên các field đổi ở các hằng số `FIELD_*` trong [app/config.py](app/config.py).

**Thêm BM25 vào dữ liệu đã có**: không sửa collection cũ tại chỗ. `python -m scripts.migrate_add_bm25 [--source quy_dinh_hybrid] [--target quy_dinh_hybrid_v2]` chép sang collection mới (cùng id, dense/sparse giữ nguyên, thêm `bm25`), đo `BM25_AVG_LEN` thật, so số điểm và in các biến cần đổi (`QDRANT_COLLECTION`, `HYBRID_LEGS`, `BM25_AVG_LEN`, `BM25_TOKENIZER`). Không bao giờ ghi vào collection nguồn, chạy lại nhiều lần cho kết quả giống nhau. Lần chạy trên 1297 chunk với `syllable_bigram` đo được `BM25_AVG_LEN=381.45`.

Volume dữ liệu thật trên máy dev: `model_qdrant_storage` (đặt qua `QDRANT_VOLUME` trong `.env` gốc). Đổi tên volume là mất index hiện có.

## 6. Cấu trúc thư mục

```
ai-service/
├── app/
│   ├── api.py                    # Điểm vào FastAPI (app.api:app): /health /ask /ask/stream /embed
│   ├── config.py                 # Toàn bộ cấu hình: hằng số + 4 biến đọc từ env / .env (xem mục 7)
│   ├── guard.py                  # Chống prompt injection: chuẩn hóa, detector đầu vào, quét đoạn trích, dựng prompt có rào, guard đầu ra (CPU, regex)
│   ├── small_talk.py             # Router ý định (chào/cảm ơn/tạm biệt/hỏi về trợ lý) + tra reply cố định từ config
│   ├── bm25.py                   # Nhánh BM25: token hóa tiếng Việt, SparseVector tài liệu/câu hỏi (CPU, không dính GPU)
│   ├── qdrant_setup.py           # Tạo collection (dense+sparse+bm25+payload index), kiểm tra nhánh lúc khởi động
│   ├── rag/
│   │   └── query_model.py        # Pipeline RAG: filter, retrieve (RRF), rerank, prompt, ngưỡng
│   ├── models/
│   │   ├── encoders.py           # Singleton BGE-M3 + Jina reranker; lock suy luận; xử lý CUDA OOM
│   │   └── jina_reranker_modeling.py  # Bản vendored của modeling Jina (khỏi trust_remote_code)
│   ├── llm/
│   │   └── groq_client.py        # Groq client: rate limit RPM/TPM, backoff, cache đĩa, ThinkFilter
│   └── ingest/
│       ├── chunker.py            # Làm sạch OCR + chia chunk theo Chương/Điều/khoản
│       ├── embed_service.py      # Chunk → encode → upsert Qdrant; xóa vector theo document_id
│       └── embed_consumer.py     # RabbitMQ consumer (thread nền, tự reconnect)
├── scripts/
│   ├── calibrate_rerank.py       # Hiệu chỉnh ngưỡng RERANK_MIN_SCORE
│   └── migrate_add_bm25.py       # Chép collection sang collection mới có thêm vector bm25
├── tests/                        # pytest (api stream, chunker, encoders, groq, khoa filter, OOM, qdrant, pipeline, reranker, bm25, hybrid legs, embed, migrate, guard, small talk, guard pipeline)
│   └── data/                     # guard_attacks.txt (câu tấn công, phải bị flag) và guard_legit.txt (câu quy chế thật, phải `allow` + không là small talk)
├── conftest.py                   # Gom .pyc của pytest vào __pycache__ duy nhất ở gốc (xem mục 8)
├── __pycache__/                  # Cache bytecode duy nhất của dự án (gitignore, dockerignore), sinh khi chạy trên host
├── Dockerfile                    # python:3.13-slim; torch cu126 (GPU) hoặc cpu qua build-arg TORCH_INDEX_URL
├── requirements.txt / constraints.txt
├── API.md                        # Đặc tả endpoint
└── .env                          # Bí mật + cấu hình (KHÔNG commit, KHÔNG copy vào image)
```

## 7. Cấu hình quan trọng

Chỉ **bốn** biến được đọc từ môi trường (`ai-service/.env`, `environment` của compose, `ENV` của Dockerfile): `GROQ_API_KEY` và `GROQ_QWEN_MODEL` (trong `.env`), `QDRANT_URL` (compose đặt `http://qdrant:6333`), `GROQ_CACHE_DIR` (Dockerfile đặt `/data/groq_cache`). Mọi giá trị còn lại là hằng số trong [app/config.py](app/config.py); tên trong bảng dưới là tên hằng số, sửa ở đó rồi `docker-compose up -d --build ai-service` (đặt biến môi trường hay `-e` **không** đổi được chúng). `RABBITMQ_URL`, `FILE_SERVICE_URL` do `embed_consumer.py` tự đọc từ môi trường. Hay dùng:

| Biến | Ý nghĩa |
|---|---|
| `GROQ_API_KEY` | Khóa Groq (bắt buộc; tên cũ `API_KEY` không còn được đọc) |
| `GROQ_QWEN_MODEL` | ID model Qwen trên Groq (bắt buộc) |
| `QDRANT_URL`, `QDRANT_COLLECTION` | Địa chỉ Qdrant / tên collection |
| `RABBITMQ_URL`, `FILE_SERVICE_URL` | Bật embed consumer / nơi lấy văn bản. docker-compose đã đặt sẵn |
| `RAG_MODE`, `TOP_K_RETRIEVAL`, `TOP_K_RERANK` | Mode và số đoạn |
| `HYBRID_LEGS` | Các nhánh của mode `hybrid`/`hybrid_rerank`, lấy từ `dense`, `sparse`, `bm25`, phân cách bằng dấu phẩy. Mặc định `dense,sparse`. Nhánh `bm25` chỉ dùng được với collection có vector `bm25` (xem mục 5); thiếu thì server không khởi động. Mode `dense` không bị ảnh hưởng |
| `BM25_TOKENIZER` | `syllable`, `syllable_bigram` (mặc định) hoặc `pyvi`. Phải **giống lúc index** |
| `BM25_K1`, `BM25_B` | Tham số BM25 (1.2, 0.75). Phải giống lúc index |
| `BM25_AVG_LEN` | Số token BM25 trung bình mỗi chunk (mặc định 120.0 chỉ là giá trị tạm); dùng giá trị `scripts/migrate_add_bm25.py` đo và in ra. Phải giống lúc index |
| `PAYLOAD_FIELD_LOAI_VAN_BAN` | Tên field payload cho `loai_van_ban` (mặc định `loai_van_ban`; chunker ghi sẵn field này) |
| `RERANK_MIN_SCORE` | Ngưỡng liên quan của reranker (hiệu chỉnh trên corpus cũ, đổi model/corpus/nhánh truy vấn thì đo lại bằng `scripts/calibrate_rerank.py`) |
| `EMBED_DEVICE`, `RERANK_DEVICE`, `RERANK_CPU_ON_OOM` | Thiết bị; GPU nhỏ (4GB) có thể đẩy reranker sang CPU |
| `WARMUP` | Nạp model ngay khi khởi động (mặc định bật; `/health` báo `models: pending/ready/error`) |
| `GROQ_RPM`, `GROQ_TPM`, `GROQ_CACHE` | Giới hạn gọi LLM và cache trả lời |
| `GUARD_ENABLED` | `true`. `false` tắt cả guard lẫn small talk: hành vi y hệt trước đây (prompt cũ, chào hỏi đi qua RAG) |
| `GUARD_MODE` | `enforce` (mặc định) hoặc `log_only` (chỉ log: không chặn câu hỏi, không loại đoạn, không chặn đầu ra, không sửa câu hỏi; small talk vẫn trả lời cố định). Dùng `log_only` để đo false positive trước khi bật thật; giá trị khác làm `import app.config` báo `ValueError` |
| `GUARD_MAX_QUESTION_CHARS` | `1000`. Câu hỏi dài hơn bị cắt (và log) sau khi chuẩn hóa |
| `GUARD_CANARY` | Để trống = `CANARY-` + 12 ký tự đầu của sha256(`BASE_SYSTEM_PROMPT`): ổn định giữa các lần chạy, nên cache đĩa Groq vẫn trúng. Đặt tay chỉ khi muốn đổi canary mà không sửa prompt (đổi canary = đổi prompt = mất cache cũ) |
| `GUARD_REFUSAL_ANSWER` | Câu từ chối cho câu hỏi bị chặn / đầu ra bị chặn (không nêu luật nào đã khớp) |
| `GUARD_REPLY_GREETING`, `_CAPABILITY`, `_META`, `_THANKS`, `_GOODBYE` | Câu trả lời cố định của small talk, mặc định trong [app/config.py](app/config.py) (một chỗ duy nhất) |

`config.py` gọi `load_dotenv(override=True)` nên **`.env` ghi đè biến môi trường**. Vì vậy `.env` bị loại khỏi image (`.dockerignore`) để không đè `QDRANT_URL` của compose. Hằng số `HYBRID_LEGS` ghi sai tên nhánh thì server vẫn khởi động (chỉ cảnh báo trong log) và mọi truy vấn hybrid lỗi `KeyError`; `BM25_TOKENIZER` sai tên thì `bm25.py` báo `ValueError` khi dùng. Chỉ `GUARD_MODE` được kiểm tra ngay lúc import.

Chuyển sang collection có BM25 (sửa trong `app/config.py` rồi `docker-compose up -d --build ai-service`): `QDRANT_COLLECTION`, `HYBRID_LEGS = ("dense", "sparse", "bm25")`, `BM25_AVG_LEN`, `BM25_TOKENIZER` đúng như `scripts/migrate_add_bm25.py` in ra. Đổi `HYBRID_LEGS` không cần index lại (vector `bm25` luôn được ghi); đổi tokenizer/K1/B/`BM25_AVG_LEN` thì phải chạy lại migrate hoặc embed lại.

## 8. Chạy và kiểm thử

- **Toàn hệ thống**: ở thư mục gốc repo, `docker-compose up -d --build`. Swagger của ai-service: http://localhost:8001/docs. Qua gateway: `http://localhost:9090/api/ai/health`.
- **Cần NVIDIA GPU** (compose đặt `deploy.resources.reservations.devices`). Máy không có GPU: xóa khối đó và build với `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`.
- Model BGE-M3 + reranker (~3GB) lưu ở volume `kltn_hf_cache`; lần đầu tải rất lâu.
- **Chạy tay**: `uvicorn app.api:app --port 8000` (cần Qdrant và `.env`).
- **Test** (trong thư mục `ai-service`, cấu hình ở `pytest.ini`): `python -m pytest -m "not integration"` là unit test, không cần mạng. Test `integration` cần Qdrant ở `QDRANT_URL` và chỉ dùng collection tạm `test_*` (tự xóa, không đụng collection thật): `tests/test_qdrant_hybrid.py`, `tests/test_embed_bm25.py`, `tests/test_migrate_bm25.py`. `tests/test_encoders.py` cần thêm model BGE-M3 thật nên chạy trong container (dừng ai-service trước để có VRAM). Test `pyvi` tự bỏ qua khi máy chưa cài `pyvi` (image Docker có sẵn). Xem mục 10 về test còn hỏng.
- **Bytecode (`__pycache__`)**: `app/` không có `__init__.py` (namespace package, như `scripts/` và `tests/`) và dự án chỉ có **một** `__pycache__` ở gốc `ai-service`. `pytest` tự làm vậy nhờ [conftest.py](conftest.py). Với `uvicorn` hay script chạy tay trên máy host, đặt biến môi trường trước khi chạy, nếu không Python sẽ tạo lại `__pycache__` cạnh từng thư mục: PowerShell `$env:PYTHONPYCACHEPREFIX = "$PWD\__pycache__"`, bash `export PYTHONPYCACHEPREFIX="$PWD/__pycache__"`. Python gom cache vào một thư mục thì bỏ qua cache sẵn có của thư viện, nên lần chạy đầu chậm thêm khoảng 25 giây và cache nặng ~100MB (phần lớn là torch, transformers). Muốn không có cache nào thì đặt `PYTHONDONTWRITEBYTECODE=1`; Docker đã đặt sẵn nên container không có `__pycache__`.
- **Đo lại ngưỡng rerank / chạy migrate trong container**: `docker-compose run --rm --no-deps ai-service python -m scripts.calibrate_rerank` (muốn đo trên collection/nhánh khác thì sửa `QDRANT_COLLECTION`/`HYBRID_LEGS` trong `app/config.py` và build lại; `-e` không còn đổi được chúng) (GPU 4GB không đủ chạy song song với ai-service đang chạy, hãy `docker-compose stop ai-service` trước).
- Tài khoản demo để gọi qua gateway: `demo_hsu` / `Password@123` (seed của auth-service). Token JWT sống 15 phút.

## 9. Bẫy đã gặp (đọc trước khi sửa)

- **Không thêm CORS ở đây.** Gateway đã gắn `Access-Control-Allow-Origin`; thêm một lần nữa ở ai-service làm response có 2 header, trình duyệt chặn `fetch`, FE rơi vào nhánh "mô phỏng" và báo "chưa kết nối được". `curl` không kiểm tra CORS nên không thấy lỗi này.
- **Re-embed phải xóa vector cũ** theo `document_id` (đã làm trong `embed_document`). `delete_document_vectors` chỉ ghi log khi lỗi chứ không ném ngoại lệ, nên nếu xóa lỗi thì vector cũ vẫn còn.
- Log `WARNING: Unsupported upgrade request` khi gateway gọi sang là **vô hại** (gateway thử nâng cấp HTTP/2 cleartext, uvicorn bỏ qua).
- Chất lượng trả lời phụ thuộc chất lượng text OCR của file-service (Tesseract hoặc ONNX). Sửa OCR xong phải **chạy lại OCR và embed** tài liệu thì Qdrant mới cập nhật.
- Collection cũ `paragraphs` (`QDRANT_LEGACY_COLLECTION`) chỉ được đọc để lấy cấu hình HNSW, không ghi/xóa.
- **Không có `__init__.py`**: Python 3.3+ nhận thư mục là package (namespace package). Cần thêm lại `__init__.py` rỗng nếu đóng gói bằng setuptools/PyInstaller, hoặc nếu có thư mục khác tên `app` nằm trên `sys.path` (namespace package sẽ gộp lẫn hai thư mục).
- **Guard là regex theo cụm từ, không hiểu nghĩa**: bắt tốt câu tấn công theo mẫu (có/không dấu, tách chữ, zero width, fullwidth, chữ Cyrillic giống Latin, leetspeak) nhưng diễn đạt lại tự do thì lọt (lần thử đầu trên 28 câu diễn đạt khác chưa từng đưa vào pattern chỉ bắt được 10/28; sau khi bổ sung pattern cho các câu lọt thì 25/28, nhưng con số sau đã bị "học" từ chính các câu đó nên không còn là số đo độc lập). Lớp bảo vệ thật nằm ở prompt có rào + guard đầu ra + loại đoạn trích. Khi sửa pattern phải chạy lại `tests/test_guard.py`: `guard_legit.txt` phải giữ **0 false positive** — các từ "quên", "bỏ qua", "hướng dẫn", "tiết lộ", "mật khẩu", "đóng vai trò", "không có giới hạn" xuất hiện trong câu hỏi quy chế thật, nên pattern luôn đòi cả cụm (động từ + danh từ chỉ dẫn + từ tổng thể/"trước đó"), không khớp từ đơn.
- **Đổi `BASE_SYSTEM_PROMPT` (hoặc `_SECURITY_RULES` trong guard.py) là đổi canary và làm mất cache Groq cũ.** Đổi tay `GUARD_CANARY` cũng vậy. Canary phải suy từ sha256, không dùng `hash()` của Python (ngẫu nhiên theo từng tiến trình).
- `/ask/stream` gửi delta đi *sau khi* kiểm tra cửa sổ trượt, nên canary bị cắt đôi qua hai delta bị chặn ở delta thứ hai, còn nửa đầu (≤ vài ký tự) đã tới client; system prompt lộ tối đa ~11 từ trước khi bị cắt. Giữ lại đuôi chưa gửi sẽ đổi cách chia delta mà test stream hiện có kiểm tra, nên chưa làm.
- `qm.main()` (CLI) và gọi thẳng `qm.ask()` không đi qua `api._gate`, nghĩa là không có chuẩn hóa/router/detector đầu vào; chỉ có phần chunk scan, prompt có rào và guard đầu ra (nằm trong `qm`).
- **Token hóa BM25**: dùng chung một bộ hàm ở [app/bm25.py](app/bm25.py) cho cả index và truy vấn (NFC, chữ thường, **giữ dấu**, giữ chữ số). Tách theo khoảng trắng làm vỡ từ ghép, nên `syllable_bigram` thêm cặp âm tiết liền nhau (`học_bổng`, `điều_15`): câu hỏi "Điều 15" khớp đúng chunk có tiền tố `Điều 15`, không khớp lẫn chunk có "điều" và "15" ở chỗ khác. Cặp được ghép cả qua dấu câu (`15_khoản`), chỉ thêm token thừa chứ không làm sai kết quả. Chỉ số sparse là hash blake2b 32 bit; **đừng** đổi sang `hash()` của Python (ngẫu nhiên theo từng lần chạy) hay đổi cách băm, vì mọi vector `bm25` đã index sẽ không còn khớp (có test khóa giá trị). BM25 của Qdrant có sẵn mặc định stemming/stopwords tiếng Anh nên không dùng.
- **Filter phải nằm trong từng Prefetch.** Đã thử trên Qdrant 1.19.1 + client 1.19.1: filter cấp cao nhất đứng một mình cũng lọc được các prefetch, nhưng không nên dựa vào đó (phiên bản khác có thể khác); và gắn filter cho chỉ một nhánh thì nhánh kia vẫn trả đoạn ngoài phạm vi. `retrieve()` gắn cùng một filter vào mọi Prefetch.
- Tokenizer `pyvi` chậm hơn và tách từ phụ thuộc ngữ cảnh câu nên câu hỏi và chunk có thể tách khác nhau; mặc định `syllable_bigram` ổn định hơn. `pyvi` được nạp lười (kéo scikit-learn) và có khóa vì CRF tagger không an toàn khi gọi song song.

## 10. Tài liệu/code đã lỗi thời (đừng tin)

- [STRUCTURE.md](STRUCTURE.md): nhắc `scripts/export_paragraphs.py`, `scripts/index_qdrant.py`, `data/docs/`, `docker-compose.yml` riêng của ai-service. Các file này **không còn**; việc lập chỉ mục giờ qua RabbitMQ/`/embed`.
- Docstring ở đầu [app/api.py](app/api.py) ghi `uvicorn api:app`; lệnh đúng là `uvicorn app.api:app`. Mục "Chạy local" của [API.md](API.md) cũng còn nhắc `index_qdrant.py`, `uvicorn api:app` và CORS `*`.
- `PARAGRAPHS_PATH` trong config và `pymupdf` trong requirements là dấu vết của script cũ. Script `index_qdrant.py` từng tạo collection bằng cách chép cấu hình HNSW từ collection cũ `paragraphs`; nó đã bị xóa, nơi tạo collection duy nhất hiện nay là `qdrant_setup.create_collection()` (dùng `HNSW_M`/`HNSW_EF_CONSTRUCT` trong config).
- `test_reranker.py::test_vendored_modeling_is_the_pinned_upstream_plus_the_documented_change` còn lỗi vì file vendored không còn dòng header mà test tìm (có từ trước, không liên quan BM25).
- `.env` đang chứa API key Groq thật; nên đổi key và đảm bảo file này không bị commit.
