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

1. `encoders.encode_texts([question])` → BGE-M3 cho vector **dense** (1024 chiều) và **sparse** (lexical weights).
2. `retrieve()`: Qdrant `query_points` với 2 nhánh prefetch (dense + sparse), hợp nhất bằng **RRF**, lấy `TOP_K_RETRIEVAL` (20) ứng viên. Có filter payload: `kho`, văn bản còn hiệu lực tại `reference_date` (mặc định hôm nay), và `khoa` của sinh viên.
3. `rerank()`: jina-reranker-v3.5 chấm lại, giữ `TOP_K_RERANK` (5) đoạn tốt nhất.
4. `_relevant()`: lọc theo ngưỡng. Mode `hybrid_rerank` giữ đoạn có `rerank_score >= RERANK_MIN_SCORE` (0.1).
5. Không còn đoạn nào → trả `in_scope=false`, `sources=[]`, câu cố định `NO_RESULT_ANSWER`, **không gọi LLM**.
6. Còn đoạn → `groq_client` gọi Qwen với `SYSTEM_PROMPT` (trả lời tiếng Việt, mỗi ý kèm trích dẫn `[Tên văn bản, Điều X, Khoản Y]`, không bịa số Điều).

Ba mode (`mode` trong body hoặc `RAG_MODE`): `dense`, `hybrid`, `hybrid_rerank` (mặc định).

`/ask/stream` trả SSE: `{"event":"sources"}` rồi nhiều `{"event":"delta","text":...}` rồi `data: [DONE]`. Lỗi lúc truy vấn trả mã HTTP (400/500/502); lỗi khi đang stream gửi `{"event":"error"}` (không có `[DONE]`).

## 4. Luồng lập chỉ mục (embed)

Code: [app/ingest/embed_consumer.py](app/ingest/embed_consumer.py) → [app/ingest/embed_service.py](app/ingest/embed_service.py) → [app/ingest/chunker.py](app/ingest/chunker.py).

1. file-service OCR xong → publish job vào exchange `file-service.embed` (routing key `embed.jobs`).
2. `embed_consumer` chạy trong **thread nền** (khởi động ở `lifespan`, chỉ khi có `RABBITMQ_URL`). Queue `ai-service.embed.jobs` là quorum queue, có dead-letter queue `ai-service.embed.jobs.dead`; lỗi xử lý thì `nack` không requeue.
3. Message: `documentId`, `source` (tên tài liệu), `kho`, `hieuLucTu`, `hieuLucDen`, `text` (tùy chọn), `action` (`UPSERT` hoặc `DELETE`). Nếu không có `text`, consumer gọi `GET {FILE_SERVICE_URL}/api/documents/{id}/text`.
4. `embed_document()`:
   - `chunker.chunk_document()` chia theo cấu trúc **Chương → Điều → khoản → điểm**; văn bản không có Điều (thông báo, biểu mẫu) thì gom theo khoản/đoạn.
   - **Xóa vector cũ theo `document_id` rồi mới ghi vector mới** (re-OCR/re-embed không bị trùng).
   - Encode theo batch (`UPSERT_BATCH_SIZE`) và upsert vào Qdrant.
5. Cũng gọi tay được: `POST /embed`, `DELETE /embed/{document_id}`.

### Chunker (đáng biết)
- Văn bản đầu vào là OCR nên nhiễu. `clean_lines()` bỏ số trang, header lặp, dòng mục lục, dòng toàn ký hiệu (con dấu/chữ ký).
- Regex tiêu đề cố tình **mờ** để chịu lỗi OCR ("Piéu", "Chuong"...).
- Mỗi chunk có tiền tố `[Tên tài liệu – Chương – Điều]`. Chunk > 1500 ký tự thì tách theo khoản; chunk < 150 ký tự thì gộp vào chunk kề.
- Dòng bảng Markdown (chứa `|`) được giữ nguyên là một đơn vị, không nối với dòng trước.
- `infer_metadata()` suy ra `loai_van_ban` (`quy_che`, `quy_dinh`, `thong_bao`, `bieu_mau`, `huong_dan`, `khac`) và khóa áp dụng `khoa_tu`/`khoa_den` **từ tên tài liệu** ("khóa 2024 trở về sau" → `khoa_tu=2024`).

## 5. Dữ liệu trong Qdrant

Collection `quy_dinh_hybrid`, hai named vector: `dense` (1024, cosine, HNSW m=32, ef_construct=200) và `sparse`.

Payload mỗi điểm: `document_id`, `source`, `text`, `paragraph_index`, `loai_van_ban`, `dieu`, `chuong`, và tùy chọn `kho`, `hieu_luc_tu`, `hieu_luc_den`, `khoa`, `khoa_tu`, `khoa_den`. Có payload index cho `kho` (keyword), `hieu_luc_*` (datetime), `khoa_*` (integer). Tên các field đổi được bằng biến `PAYLOAD_FIELD_*` trong [app/config.py](app/config.py).

Volume dữ liệu thật trên máy dev: `model_qdrant_storage` (đặt qua `QDRANT_VOLUME` trong `.env` gốc). Đổi tên volume là mất index hiện có.

## 6. Cấu trúc thư mục

```
ai-service/
├── app/
│   ├── api.py                    # Điểm vào FastAPI (app.api:app): /health /ask /ask/stream /embed
│   ├── config.py                 # Toàn bộ cấu hình, đọc từ env / .env
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
├── scripts/calibrate_rerank.py   # Hiệu chỉnh ngưỡng RERANK_MIN_SCORE
├── tests/                        # pytest (api stream, chunker, encoders, groq, khoa filter, OOM, qdrant, pipeline, reranker)
├── Dockerfile                    # python:3.13-slim; torch cu126 (GPU) hoặc cpu qua build-arg TORCH_INDEX_URL
├── requirements.txt / constraints.txt
├── API.md                        # Đặc tả endpoint
└── .env                          # Bí mật + cấu hình (KHÔNG commit, KHÔNG copy vào image)
```

## 7. Cấu hình quan trọng

Mọi giá trị đổi bằng biến môi trường (xem [app/config.py](app/config.py)). Hay dùng:

| Biến | Ý nghĩa |
|---|---|
| `API_KEY` hoặc `GROQ_API_KEY` | Khóa Groq (bắt buộc) |
| `GROQ_QWEN_MODEL` | ID model Qwen trên Groq (bắt buộc) |
| `QDRANT_URL`, `QDRANT_COLLECTION` | Địa chỉ Qdrant / tên collection |
| `RABBITMQ_URL`, `FILE_SERVICE_URL` | Bật embed consumer / nơi lấy văn bản. docker-compose đã đặt sẵn |
| `RAG_MODE`, `TOP_K_RETRIEVAL`, `TOP_K_RERANK` | Mode và số đoạn |
| `RERANK_MIN_SCORE` | Ngưỡng liên quan của reranker (hiệu chỉnh trên corpus cũ, đổi model/corpus thì đo lại) |
| `EMBED_DEVICE`, `RERANK_DEVICE`, `RERANK_CPU_ON_OOM` | Thiết bị; GPU nhỏ (4GB) có thể đẩy reranker sang CPU |
| `WARMUP` | Nạp model ngay khi khởi động (mặc định bật; `/health` báo `models: pending/ready/error`) |
| `GROQ_RPM`, `GROQ_TPM`, `GROQ_CACHE` | Giới hạn gọi LLM và cache trả lời |

`config.py` gọi `load_dotenv(override=True)` nên **`.env` ghi đè biến môi trường**. Vì vậy `.env` bị loại khỏi image (`.dockerignore`) để không đè `QDRANT_URL` của compose.

## 8. Chạy và kiểm thử

- **Toàn hệ thống**: ở thư mục gốc repo, `docker-compose up -d --build`. Swagger của ai-service: http://localhost:8001/docs. Qua gateway: `http://localhost:9090/api/ai/health`.
- **Cần NVIDIA GPU** (compose đặt `deploy.resources.reservations.devices`). Máy không có GPU: xóa khối đó và build với `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`.
- Model BGE-M3 + reranker (~3GB) lưu ở volume `kltn_hf_cache`; lần đầu tải rất lâu.
- **Chạy tay**: `uvicorn app.api:app --port 8000` (cần Qdrant và `.env`).
- **Test**: `python -m pytest -m "not integration" --ignore=tests/test_qdrant_hybrid.py` trong thư mục `ai-service` (cấu hình ở `pytest.ini`); xem mục 10 về hai test đang hỏng.
- Tài khoản demo để gọi qua gateway: `demo_hsu` / `Password@123` (seed của auth-service). Token JWT sống 15 phút.

## 9. Bẫy đã gặp (đọc trước khi sửa)

- **Không thêm CORS ở đây.** Gateway đã gắn `Access-Control-Allow-Origin`; thêm một lần nữa ở ai-service làm response có 2 header, trình duyệt chặn `fetch`, FE rơi vào nhánh "mô phỏng" và báo "chưa kết nối được". `curl` không kiểm tra CORS nên không thấy lỗi này.
- **Re-embed phải xóa vector cũ** theo `document_id` (đã làm trong `embed_document`). `delete_document_vectors` chỉ ghi log khi lỗi chứ không ném ngoại lệ, nên nếu xóa lỗi thì vector cũ vẫn còn.
- Log `WARNING: Unsupported upgrade request` khi gateway gọi sang là **vô hại** (gateway thử nâng cấp HTTP/2 cleartext, uvicorn bỏ qua).
- Chất lượng trả lời phụ thuộc chất lượng text OCR của file-service (Tesseract hoặc ONNX). Sửa OCR xong phải **chạy lại OCR và embed** tài liệu thì Qdrant mới cập nhật.
- Collection cũ `paragraphs` (`QDRANT_LEGACY_COLLECTION`) chỉ được đọc để lấy cấu hình HNSW, không ghi/xóa.

## 10. Tài liệu/code đã lỗi thời (đừng tin)

- [STRUCTURE.md](STRUCTURE.md): nhắc `scripts/export_paragraphs.py`, `scripts/index_qdrant.py`, `data/docs/`, `docker-compose.yml` riêng của ai-service. Các file này **không còn**; việc lập chỉ mục giờ qua RabbitMQ/`/embed`.
- Docstring ở đầu [app/rag/query_model.py](app/rag/query_model.py) nhắc `index_qdrant.py`, và ở đầu [app/api.py](app/api.py) ghi `uvicorn api:app`; lệnh đúng là `uvicorn app.api:app`.
- `PARAGRAPHS_PATH` trong config và `pymupdf` trong requirements là dấu vết của script cũ.
- [tests/test_qdrant_hybrid.py](tests/test_qdrant_hybrid.py) import `index_qdrant` (đã bị xóa) nên không chạy được; `test_reranker.py::test_vendored_modeling_is_the_pinned_upstream_plus_the_documented_change` lỗi vì file vendored không còn dòng header mà test tìm.
- `.env` đang chứa API key Groq thật; nên đổi key và đảm bảo file này không bị commit.
