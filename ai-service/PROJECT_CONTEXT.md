# AI Service – Project Context

Ngữ cảnh để người mới (hoặc AI agent) vào việc ngay với `ai-service`. Chi tiết từng endpoint xem [API.md](API.md).
Mô tả này khớp với code tại thời điểm viết; khi code đổi thì sửa file này.

## 1. Vai trò

`ai-service` là service Python (FastAPI) làm hai việc cho chatbot quy chế **Đại học Hoa Sen** (đồ án tốt nghiệp):

1. **Hỏi đáp RAG**: nhận câu hỏi, tìm đoạn quy định liên quan trong Qdrant, sinh câu trả lời có trích dẫn bằng LLM (Groq Qwen).
2. **Lập chỉ mục (embed)**: nhận văn bản đã OCR của file-service, chia đoạn, tạo vector, ghi vào Qdrant.

Không có database riêng: dữ liệu vector nằm ở Qdrant, văn bản gốc nằm ở file-service. Ngoại lệ tùy chọn: lịch sử hội thoại ngắn hạn của từng `session_id` (mục 3, 5), mặc định nằm trong RAM của tiến trình.

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

**Router LLM + session + lớp bảo vệ còn lại** (code: [app/router/router_llm.py](app/router/router_llm.py), [app/session.py](app/session.py), [app/router/guard.py](app/router/guard.py), [app/router/small_talk.py](app/router/small_talk.py)). Hai endpoint là `async def` và chạy chung các hàm `api._begin` → `api._plan`; phần chặn (encode GPU, Qdrant, rerank, Groq) chạy trong threadpool (`run_in_threadpool`), còn router và session chạy trên event loop nên không chặn nó. **Không còn luật từ khóa nào phân loại câu hỏi hay đoạn trích**: Gemini tự suy luận câu hỏi thuộc loại nào. Đã xóa `inspect_question`, `strip_delimiters`, `small_talk.route*`, `scan_chunk`/`drop_flagged` và mọi danh sách regex đi kèm (bản cũ nằm ở commit `428c97c` của repo ai-service). Cờ `ROUTER_ENABLED` và `SESSION_ENABLED` đều tắt mặc định.

Thứ tự trong một request:

1. `api._begin`. Chuẩn hóa câu hỏi (`guard.normalize_question`: NFKC, bỏ zero width/ký tự điều khiển, gộp khoảng trắng nhưng **giữ xuống dòng**, cắt quá `GUARD_MAX_QUESTION_CHARS`; rỗng → 400). Khi `SESSION_ENABLED`: `session_id` phải là UUID (sai → 400), mở session (giữ lock đến hết request), rồi gộp `khoa`/`kho`: giá trị trong request đè giá trị đã lưu (kiểm lại bằng chính validator của `AskRequest`); `reference_date` không bao giờ được lưu.
2. `api._plan` với `ROUTER_ENABLED=true`. Một lần gọi Gemini (REST `generateContent` bằng `httpx`) nhận **chỉ**: câu hỏi đã chuẩn hóa, `ROUTER_HISTORY_TURNS` lượt cuối của lịch sử (mỗi tin bị cắt), `khoa`/`kho` hiệu lực và cờ `previous_was_clarify` (suy ra từ `intent` của câu trả lời cuối đã lưu, không bao giờ từ request). Không gửi đoạn trích, payload, system prompt, bí mật, `user_id`/`session_id`. Kết quả `RouteDecision(intent, confidence, standalone_query, missing)` đi qua `router_llm.apply_policy` theo thứ tự:
   - `injection` ≥ `ROUTER_BLOCK_CONFIDENCE` (0.7) → `GUARD_REFUSAL_ANSWER`, `in_scope=false`, `intent=blocked`, **không lưu**.
   - `off_topic` ≥ `ROUTER_OFFTOPIC_CONFIDENCE` (0.8) → `OFF_TOPIC_ANSWER`, `in_scope=false`, `intent=off_topic`, **không lưu**.
   - `clarify` ≥ `ROUTER_CLARIFY_CONFIDENCE` (0.7), khi `ROUTER_CLARIFY_ENABLED`, lượt trước **không** phải clarify (không hỏi hai lần liên tiếp) và không phải (`missing=audience` mà `khoa` đã biết) → câu hỏi lại cố định `CLARIFY_ANSWER_TOPIC/AUDIENCE/TIME` theo `missing`, `in_scope=true`, `intent=clarify`, **có lưu** để lượt sau hoàn tất ngữ cảnh.
   - `greeting`/`thanks`/`goodbye`/`capability`/`meta` ≥ `ROUTER_MIN_CONFIDENCE` (0.5) → câu cố định `GUARD_REPLY_*`, có lưu.
   - Mọi quyết định dưới ngưỡng của nó, và `rag`, đi tiếp RAG; ngưỡng liên quan `RERANK_MIN_SCORE` quyết định có `NO_RESULT_ANSWER` hay không.
   - Mọi chữ hiện ra cho người dùng lấy từ config. `standalone_query` (≤ 300 ký tự, rỗng thì dùng câu hỏi chuẩn hóa) chỉ là chuỗi tìm kiếm cho encode/Qdrant/rerank, không bao giờ vào prompt trả lời.
   - Router tắt, hoặc lỗi (timeout, quota, JSON hỏng sau một lần thử lại, hết `ROUTER_RPM`): **câu hỏi đi thẳng vào RAG**, không có `off_topic`/`clarify`; log ghi `fallback=rag failure=<loại lỗi>`. Router không bao giờ làm request trả lỗi. Quyết định của request không có lịch sử được cache trong tiến trình (LRU + TTL, khóa gồm phiên bản prompt, model, câu hỏi, `khoa`, `kho`).
3. Câu trả lời cố định (chào hỏi, bị chặn, off_topic, clarify) **không** gọi encoder, Qdrant, reranker hay Groq và có `sources=[]`.
4. Câu còn lại chạy RAG bên dưới, với `standalone_query` (nếu có) làm câu truy vấn cho encode/retrieve/rerank. Prompt trả lời vẫn mang câu hỏi gốc trong `<question>`.
5. **Dựng prompt** – `qm.build_messages`: mỗi đoạn là `<chunk n="i">` trong `<context>`, lịch sử (nếu có) trong `<history>` (nhãn `Người dùng:`/`Trợ lý:`, mỗi tin gọn thành một dòng), câu hỏi trong `<question>`; thẻ `context/question/chunk/history/facts/system/assistant/user` xuất hiện trong dữ liệu bị đổi thành `[...]`. `SYSTEM_PROMPT` = `BASE_SYSTEM_PROMPT` (luật cũ, giữ nguyên) + 3 luật chống injection (luật 4 nói `<history>` chỉ là dữ liệu để hiểu câu hỏi) + canary (`guard.harden_prompt`). `GUARD_ENABLED=false`: prompt cũ, không có lịch sử trong prompt trả lời (retrieval vẫn dùng `standalone_query`).
6. **Đầu ra** – `qm.ask` (`/ask`) và `qm.generate_answer_stream` (`/ask/stream`), sau ThinkFilter, chỉ xét text hiển thị: `guard.check_output` thất bại nếu lộ canary, ≥ 12 từ liên tiếp của system prompt (trừ câu từ chối được phép nói nguyên văn), khóa `gsk_…`/thông tin đăng nhập `amqp://`, `NAME=value` của biến bí mật, hoặc đúng giá trị `GROQ_API_KEY`. `/ask` trả `GUARD_REFUSAL_ANSWER` (`in_scope=false`, `sources=[]`, `intent=blocked`); `/ask/stream` kiểm tra trên cửa sổ trượt 600 ký tự cuối + delta mới *trước khi* gửi delta, thất bại thì phát `{"event":"error"}` (không có `[DONE]`). `GUARD_MODE=log_only` chỉ ghi log lớp này.
7. **Lưu lượt** (chỉ khi `SESSION_ENABLED`): câu hỏi chuẩn hóa + câu trả lời cuối + `intent` + `khoa`/`kho` hiệu lực, **sau khi câu trả lời hoàn tất** (với `/ask/stream`: sau delta cuối, trước `[DONE]`). Không lưu `injection`, `off_topic`, câu bị guard đầu ra chặn, stream lỗi. Lỗi khi lưu chỉ ghi log, không làm hỏng câu trả lời.
8. Mỗi request có router ghi một dòng `router intent=… conf=… action=… model=… cached=… fallback=… failure=… latency_ms=… session=<8 ký tự hash> history_turns=… note=…`. Không bao giờ log nội dung câu hỏi, lịch sử, khóa Gemini hay canary.

**Session** ([app/session.py](app/session.py)): mỗi `session_id` (UUID, server tạo UUID v4 nếu thiếu) là một LlamaIndex `Memory` chỉ giữ lịch sử chat ngắn hạn (không có memory block gọi LLM), tất cả dùng chung **một** engine SQLite (trong RAM với `SESSION_STORE=memory`, file với `sqlite`). Server không bao giờ nhận lịch sử từ client. Cửa sổ đưa vào prompt: tối đa `SESSION_MAX_TURNS` lượt **và** `SESSION_MAX_HISTORY_TOKENS` token (đếm bảo thủ 2 ký tự/token cho tiếng Việt), mỗi tin bị cắt `SESSION_MESSAGE_MAX_CHARS`. Mỗi session có một `asyncio.Lock`: request thứ hai cùng session chờ request đầu xong (kể cả stream) nên cả hai được trả lời và lưu đúng thứ tự. Hết hạn `SESSION_TTL_S` (tính từ lượt lưu cuối) hoặc quá `SESSION_MAX_SESSIONS` (LRU) thì lịch sử bị xóa; session đang có request giữ/chờ không bị dọn. `reset=true` xóa lịch sử trước khi xử lý. Khóa lưu trữ là `session_id`, hoặc `X-User-Id:session_id` khi gateway gửi header đó (xem mục 9). **Không mở được session store thì request vẫn được trả lời, chỉ không có lịch sử** (log ghi loại lỗi, không ghi nội dung).

1. `_query_legs()` tạo vector truy vấn cho từng nhánh trong `HYBRID_LEGS` (mặc định `dense,sparse`):
   - `dense` (1024 chiều) và `sparse` (trọng số học được của BGE-M3) lấy từ **một** lần `encoders.encode_texts([question])` (GPU);
   Nhánh nào không có token (câu hỏi toàn dấu câu) bị bỏ; không còn nhánh nào thì dùng dense.
2. `retrieve()`: **một** request `query_points` tới Qdrant, mỗi nhánh là một `Prefetch` (limit `TOP_K_RETRIEVAL`=20); Qdrant hợp nhất các nhánh bằng **RRF** ở phía server và trả `TOP_K_RETRIEVAL` ứng viên (mỗi chunk chỉ ra một lần dù nằm trong nhiều danh sách). **Lọc metadata trước khi tìm vector**: cùng một filter (`kho`, văn bản còn hiệu lực tại `reference_date` mặc định hôm nay, `khoa` của sinh viên, `loai_van_ban` nếu có) được gắn vào từng Prefetch; Qdrant áp nó ngay trong lúc tìm (có payload index) chứ không lọc sau khi đã lấy top-k, và nhánh nào thiếu filter sẽ trả cả đoạn ngoài phạm vi. Chỉ còn một nhánh thì truy vấn thẳng nhánh đó (điểm là điểm của nhánh, không phải RRF). Mode `dense` luôn chỉ dùng dense, bỏ qua `HYBRID_LEGS`.
   **Viết lại câu hỏi** (`QUERY_REWRITE_ENABLED=true`, chỉ `hybrid`/`hybrid_rerank`): `find_hits` gọi [app/rag/query_rewrite.py](app/rag/query_rewrite.py) cho Qwen viết lại câu tìm kiếm (sau router thì là `standalone_query` của router). `retrieve(..., alt_question=...)` tìm `dense` + `sparse` cho **cả hai câu** (4 danh sách) rồi RRF **một lần** trên tất cả, trong **một** request `query_points` (các Prefetch nằm phẳng, không lồng nhau; filter gắn ở từng Prefetch như trên). Chunk nằm trong nhiều danh sách chỉ ra một lần (Qdrant gộp theo id; kho đo 2026-10 gồm 1311 chunk không có cặp nào trùng nội dung nên không cần dedup thêm). Danh sách hợp nhất vẫn `TOP_K_RETRIEVAL` ứng viên và rerank chấm theo câu gốc. Qwen lỗi / trả rỗng / trùng câu gốc → tìm một câu như cũ, không báo lỗi.
3. `rerank()`: jina-reranker-v3.5 chấm lại các ứng viên, giữ `TOP_K_RERANK` (5) đoạn tốt nhất.
4. `_relevant()`: lọc theo ngưỡng. Mode `hybrid_rerank` giữ đoạn có `rerank_score >= RERANK_MIN_SCORE` (0.0).
5. Không còn đoạn nào → trả `in_scope=false`, `sources=[]`, câu cố định `NO_RESULT_ANSWER`, **không gọi LLM**.
6. Còn đoạn → `build_messages()` ghép các đoạn thành context (`[Đoạn i] Tài liệu: ...`, đúng thứ tự rerank) và `groq_client` gọi Qwen với `SYSTEM_PROMPT` (trả lời tiếng Việt, mỗi ý kèm trích dẫn `[Tên văn bản, Điều X, Khoản Y]`, không bịa số Điều).

Ba mode (`mode` trong body hoặc `RAG_MODE`): `dense`, `hybrid` (các nhánh của `HYBRID_LEGS`, RRF, không rerank), `hybrid_rerank` (mặc định: như `hybrid` rồi rerank). Request có thêm `loai_van_ban` (danh sách, ví dụ `["thong_bao"]`) để chỉ tìm trong các loại văn bản đó; bỏ trống thì không lọc.

Lúc khởi động (`lifespan`), `qdrant_setup.validate_legs()` kiểm tra mọi nhánh trong `HYBRID_LEGS` có vector trong collection. Thiếu thì `RuntimeError` nêu tên nhánh, server không lên. Qdrant chưa sẵn sàng hoặc collection chưa tồn tại thì chỉ ghi log và vẫn khởi động.

`/ask/stream` trả SSE: `{"event":"sources"}` rồi nhiều `{"event":"delta","text":...}` rồi `data: [DONE]`. Lỗi lúc truy vấn trả mã HTTP (400/500/502); lỗi khi đang stream gửi `{"event":"error"}` (không có `[DONE]`). Câu trả lời cố định (small talk, bị chặn, `off_topic`, `clarify`) dùng cùng hợp đồng: `sources` (rỗng, kèm `intent`), đúng một `delta`, `[DONE]`. Sự kiện `sources` của câu hỏi RAG **không** có khóa `intent`; `/ask` luôn có `intent`. Khi `SESSION_ENABLED`, sự kiện `sources` và response `/ask` có thêm `session_id`; tắt session thì không có khóa này (hình dạng y hệt trước).

## 4. Luồng lập chỉ mục (embed)

Code: [app/ingest/embed_consumer.py](app/ingest/embed_consumer.py) → [app/ingest/embed_service.py](app/ingest/embed_service.py) → [app/ingest/chunker.py](app/ingest/chunker.py).

1. file-service OCR xong → publish job vào exchange `file-service.embed` (routing key `embed.jobs`).
2. `embed_consumer` chạy trong **thread nền** (khởi động ở `lifespan`, chỉ khi có `RABBITMQ_URL`). Queue `ai-service.embed.jobs` là quorum queue, có dead-letter queue `ai-service.embed.jobs.dead`; lỗi xử lý thì `nack` không requeue.
3. Message: `documentId`, `source` (tên tài liệu), `kho`, `hieuLucTu`, `hieuLucDen`, `text` (tùy chọn), `action` (`UPSERT` hoặc `DELETE`). Nếu không có `text`, consumer gọi `GET {FILE_SERVICE_URL}/api/documents/{id}/text`.
4. `embed_document()`:
   - `chunker.chunk_document()` chia theo cấu trúc **Chương → Điều → khoản → điểm**; văn bản không có Điều (thông báo, biểu mẫu) thì gom theo khoản/đoạn.
   - **Xóa vector cũ theo `document_id` rồi mới ghi vector mới** (re-OCR/re-embed không bị trùng).
   - Encode theo batch (`UPSERT_BATCH_SIZE`) và upsert vào Qdrant.
   - Collection chưa có thì tạo bằng `qdrant_setup.create_collection()` (đủ `dense`, `sparse` và payload index).
5. Cũng gọi tay được: `POST /embed`, `DELETE /embed/{document_id}`.

### Chunker (đáng biết)
- Văn bản đầu vào là OCR nên nhiễu. `clean_lines()` bỏ số trang, header lặp, dòng mục lục, dòng toàn ký hiệu (con dấu/chữ ký).
- Regex tiêu đề cố tình **mờ** để chịu lỗi OCR: "Piéu", "Ei6u", "Chuong"; ký tự rác đứng trước (`! Điều 9.`, `. Điều 5.`); dấu chấm bị đọc thành `,`/`;` (`Điều 9, Xử lý ...`, chỉ nhận khi sau đó là chữ hoa và không phải dẫn chiếu như `Điều 9, 10 và 11`). Tiêu đề bị OCR dựng thành hàng bảng (`| Điều 3. Học phí | / |  |`) được cứu trước bộ lọc "dòng toàn ký hiệu" (`_unpipe`), nếu không dòng này bị xóa như con dấu và nội dung Điều đó bị gắn vào Điều trước.
- Chọn tiêu đề thân bài (`_find_sections`): chuỗi số Điều tăng dần có điểm cao nhất, điểm = số tiêu đề trong chuỗi trừ số tiêu đề bị bỏ qua nằm giữa chuỗi. Nhờ vậy bìa Quyết định (Điều 1-3), mục lục không có dấu chấm và dẫn chiếu lạc dòng không chen vào thân bài. Cách cũ ("lần xuất hiện cuối của mỗi số" + lọc tham lam) làm 26/45 văn bản có cấu trúc bị mất hoặc gắn nhầm tiêu đề. Sửa chunker thì phải **nạp lại tài liệu** (dữ liệu trong Qdrant là kết quả của chunker lúc nạp): `POST {file-service}/api/documents/{id}/embed` phát lại job embed, không cần OCR lại. Còn 4 văn bản (44, 60, 63, 77) có tiêu đề thật sự không còn trong OCR nên vẫn nhảy số.
- Mỗi chunk có tiền tố `[Tên tài liệu – Chương – Điều]`. Chunk > 1500 ký tự thì tách theo khoản; chunk < 150 ký tự thì gộp vào chunk kề.
- Dòng bảng Markdown (chứa `|`) được giữ nguyên là một đơn vị, không nối với dòng trước.
- `infer_metadata()` suy ra `loai_van_ban` (`quy_che`, `quy_dinh`, `thong_bao`, `bieu_mau`, `huong_dan`, `khac`) và khóa áp dụng `khoa_tu`/`khoa_den` **từ tên tài liệu** ("khóa 2024 trở về sau" → `khoa_tu=2024`).

## 5. Dữ liệu trong Qdrant

Collection `quy_dinh_hybrid` (tên đặt bằng `QDRANT_COLLECTION`) có các named vector:

| Vector | Loại | Ghi chú |
|---|---|---|
| `dense` | 1024, cosine, HNSW m=32, ef_construct=200 | BGE-M3 |
| `sparse` | sparse, **không** IDF | trọng số học được của BGE-M3 |

Payload mỗi điểm: `document_id`, `source`, `text`, `paragraph_index`, `loai_van_ban`, `dieu`, `chuong`, và tùy chọn `kho`, `hieu_luc_tu`, `hieu_luc_den`, `khoa`, `khoa_tu`, `khoa_den`. Có payload index cho `kho`, `doi_tuong`, `loai_van_ban` (keyword), `hieu_luc_*` (datetime), `khoa_*` (integer) khi collection được tạo bởi `qdrant_setup.create_collection()`. Tên các field đổi ở các hằng số `FIELD_*` trong [app/config.py](app/config.py).

Volume dữ liệu thật trên máy dev: `model_qdrant_storage` (đặt qua `QDRANT_VOLUME` trong `.env` gốc). Đổi tên volume là mất index hiện có.

### Lịch sử session (tùy chọn, `SESSION_ENABLED=true`)

Bảng `chat_turns` (LlamaIndex `SQLAlchemyChatStore`; các cột `key`, `timestamp`, `role`, `status`, `data` JSON). `key` là khóa session, `data` là tin nhắn LlamaIndex kèm `additional_kwargs` do server ghi: `ts`, `intent` (của câu trả lời), `khoa`/`kho` (của câu hỏi). Chỉ lưu câu hỏi chuẩn hóa và câu trả lời cuối; không lưu đoạn trích, `sources`, system prompt hay canary. Mặc định (`SESSION_STORE=memory`) là SQLite trong RAM: mất khi restart. `SESSION_STORE=sqlite` ghi file tại `SESSION_DB_URI` (`sqlite+aiosqlite:///…`) và **cần volume** (compose gốc chưa sửa, xem mục 9). Dòng của session hết hạn/bị LRU loại được xóa.

## 6. Cấu trúc thư mục

```
ai-service/
├── app/
│   ├── api.py                    # Điểm vào FastAPI (app.api:app): /health /ask /ask/stream /embed
│   ├── config.py                 # Toàn bộ cấu hình: hằng số + 4 biến đọc từ env / .env (xem mục 7)
│   ├── session.py                # Lịch sử hội thoại ngắn hạn theo session_id (LlamaIndex Memory trên SQLite), lock từng session, TTL/LRU
│   ├── router/
│   │   ├── router_llm.py         # Router Gemini (httpx, REST): prompt + ví dụ, RouteDecision, limiter RPM, cache, chính sách (off_topic/clarify/...)
│   │   ├── guard.py              # Chuẩn hóa câu hỏi, vô hiệu hóa thẻ phân cách, system prompt có rào + canary, guard đầu ra (không còn luật từ khóa)
│   │   └── small_talk.py         # enum Intent + câu trả lời cố định (reply_for); không còn bộ phân loại từ khóa
│   ├── qdrant_setup.py           # Tạo collection (dense+sparse+payload index), kiểm tra nhánh lúc khởi động
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
│   └── eval_router.py            # Đánh giá router trên Gemini thật (tốn quota): recall tấn công, off_topic/clarify sai, độ trễ, số lần gọi
├── tests/                        # pytest (api stream, chunker, encoders, groq, khoa filter, OOM, qdrant, pipeline, reranker, hybrid legs, query rewrite, embed, guard, small talk, guard pipeline, router_llm, session, router+session qua API, corpus router, live)
│   └── data/                     # guard_attacks.txt / guard_legit.txt (corpus tấn công và câu quy chế thật, dùng cho eval_router), router_offtopic.txt, router_clarify.txt, router_smalltalk.txt, router_dialogs.json (hội thoại nhiều lượt)
├── conftest.py                   # Gom .pyc của pytest vào __pycache__ duy nhất ở gốc (xem mục 8)
├── __pycache__/                  # Cache bytecode duy nhất của dự án (gitignore, dockerignore), sinh khi chạy trên host
├── Dockerfile                    # python:3.13-slim; torch cu126 (GPU) hoặc cpu qua build-arg TORCH_INDEX_URL
├── requirements.txt / constraints.txt
├── API.md                        # Đặc tả endpoint
└── .env                          # Bí mật + cấu hình (KHÔNG commit, KHÔNG copy vào image)
```

## 7. Cấu hình quan trọng

**Bốn** biến cũ được đọc từ môi trường (`ai-service/.env`, `environment` của compose, `ENV` của Dockerfile): `GROQ_API_KEY` và `GROQ_QWEN_MODEL` (trong `.env`), `QDRANT_URL` (compose đặt `http://qdrant:6333`), `GROQ_CACHE_DIR` (Dockerfile đặt `/data/groq_cache`); **cộng nhóm router/session** (`ROUTER_*`, `GEMINI_API_KEY`, `GEMINI_MODEL`, `SESSION_*`) đọc cùng cách, đổi không cần build lại. Mọi giá trị còn lại là hằng số trong [app/config.py](app/config.py); tên trong bảng dưới là tên hằng số, sửa ở đó rồi `docker-compose up -d --build ai-service` (đặt biến môi trường hay `-e` **không** đổi được chúng). `RABBITMQ_URL`, `FILE_SERVICE_URL` do `embed_consumer.py` tự đọc từ môi trường. Hay dùng:

| Biến | Ý nghĩa |
|---|---|
| `GROQ_API_KEY` | Khóa Groq (bắt buộc; tên cũ `API_KEY` không còn được đọc) |
| `GROQ_QWEN_MODEL` | ID model Qwen trên Groq (bắt buộc) |
| `QDRANT_URL`, `QDRANT_COLLECTION` | Địa chỉ Qdrant / tên collection |
| `RABBITMQ_URL`, `FILE_SERVICE_URL` | Bật embed consumer / nơi lấy văn bản. docker-compose đã đặt sẵn |
| `RAG_MODE`, `TOP_K_RETRIEVAL`, `TOP_K_RERANK` | Mode và số đoạn |
| `HYBRID_LEGS` | Các nhánh của mode `hybrid`/`hybrid_rerank`, lấy từ `dense`, `sparse`, phân cách bằng dấu phẩy. Mặc định `dense,sparse`. Tên khác thì truy vấn báo `ValueError`. Mode `dense` không bị ảnh hưởng |
| `PAYLOAD_FIELD_LOAI_VAN_BAN` | Tên field payload cho `loai_van_ban` (mặc định `loai_van_ban`; chunker ghi sẵn field này) |
| `RERANK_MIN_SCORE` | Ngưỡng liên quan của reranker, `0.0` (đo lại 2026-10 trên 1297 chunk: câu lạc đề cao nhất −0,032, đoạn tốt nhất của câu hợp lệ ≥ 0,2; `0.1` cũ cắt hết đoạn đúng của câu hỏi ngắn). Đổi model/corpus/nhánh truy vấn thì đo lại bằng `scripts/calibrate_rerank.py` |
| `EMBED_DEVICE`, `RERANK_DEVICE`, `RERANK_CPU_ON_OOM` | Thiết bị; GPU nhỏ (4GB) có thể đẩy reranker sang CPU |
| `WARMUP` | Nạp model ngay khi khởi động (mặc định bật; `/health` báo `models: pending/ready/error`) |
| `GROQ_RPM`, `GROQ_TPM`, `GROQ_CACHE` | Giới hạn gọi LLM và cache trả lời |
| `QUERY_REWRITE_ENABLED` | Env, mặc định `true` (luôn dùng; `false` để tắt, vd khi hết quota Groq). Bước viết lại bằng Qwen, tìm `dense` + `sparse` cho cả hai câu rồi RRF một lần (mục luồng RAG, bước 2). Mỗi câu hỏi RAG tốn thêm **một** lần gọi Groq (chia đôi ngân sách `GROQ_RPM`, thêm ~0,2–0,8 s đo được 2026-10); gọi với `retries=0` nên lỗi Groq không bắt request chờ backoff. Rerank vẫn chấm theo câu gốc, câu viết lại không bao giờ vào prompt trả lời. `QUERY_REWRITE_MAX_CHARS`=300 là hằng số |
| `GUARD_ENABLED` | `true`. `false` tắt chuẩn hóa câu hỏi, prompt có rào + canary và guard đầu ra: prompt cũ, không có lịch sử trong prompt trả lời |
| `GUARD_MODE` | `enforce` (mặc định) hoặc `log_only` (guard đầu ra chỉ ghi log, không chặn). Giá trị khác làm `import app.config` báo `ValueError` |
| `GUARD_MAX_QUESTION_CHARS` | `1000`. Câu hỏi dài hơn bị cắt (và log) sau khi chuẩn hóa; router cũng cắt theo giá trị này |
| `GUARD_CANARY` | Để trống = `CANARY-` + 12 ký tự đầu của sha256(`BASE_SYSTEM_PROMPT`): ổn định giữa các lần chạy, nên cache đĩa Groq vẫn trúng. Đặt tay chỉ khi muốn đổi canary mà không sửa prompt (đổi canary = đổi prompt = mất cache cũ) |
| `GUARD_REFUSAL_ANSWER`, `GUARD_REPLY_GREETING`/`_CAPABILITY`/`_META`/`_THANKS`/`_GOODBYE`, `OFF_TOPIC_ANSWER`, `CLARIFY_ANSWER_TOPIC`/`_AUDIENCE`/`_TIME` | Mọi câu cố định hiện cho người dùng (một chỗ duy nhất trong [app/config.py](app/config.py)); không nêu tên model hay nhà cung cấp |
| `ROUTER_ENABLED` | Env, mặc định `false`. Bật router Gemini (thay cho mọi luật từ khóa). Tắt: mọi câu hỏi vào thẳng RAG |
| `GEMINI_API_KEY` | Env. Gửi ở header `x-goog-api-key` (không nằm trong URL); không bao giờ log, không nằm trong image (`.env` bị `.dockerignore` loại, compose truyền qua `env_file`) |
| `GEMINI_MODEL` → hằng số `ROUTER_MODEL` | Env. Id model Gemini, vd `gemini-3.5-flash-lite` (stable, không dùng bản `*-preview`); tiền tố `gemini/` kiểu cũ vẫn được chấp nhận và bị bỏ. Bắt buộc khi `ROUTER_ENABLED` |
| `ROUTER_TIMEOUT_S`, `ROUTER_MAX_RETRIES` | Env, `10` / `1`. Timeout mỗi lần gọi và số lần thử lại (timeout, mất kết nối, 5xx; 429 và 4xx khác không thử lại). Độ trễ thật đo được 1–9 s (p95 ≈ 8,8 s); xấu nhất trước khi rơi vào RAG là `(1 + ROUTER_MAX_RETRIES) × ROUTER_TIMEOUT_S` = 20 s: xem mục 9 |
| `ROUTER_RPM` | Env, `15`. Giới hạn trong tiến trình; hết ngân sách → request đi thẳng RAG. Google không công bố số free tier, 15 là số LiteLLM từng ghi cho model này |
| `ROUTER_MIN_CONFIDENCE`, `ROUTER_BLOCK_CONFIDENCE`, `ROUTER_OFFTOPIC_CONFIDENCE`, `ROUTER_CLARIFY_CONFIDENCE` | Env, `0.5` / `0.7` / `0.8` / `0.7`. Quyết định dưới ngưỡng của nó thì coi là `rag` |
| `ROUTER_CLARIFY_ENABLED` | Env, `true`. `false`: `clarify` luôn thành `rag` (để đo ablation) |
| `ROUTER_HISTORY_TURNS` | Env, `3`. Số lượt lịch sử gửi cho router (mỗi tin cắt `ROUTER_HISTORY_MSG_CHARS`=1500) |
| `ROUTER_REASONING_EFFORT` | Env, `low`. **Đừng đặt `none`/`minimal` với Gemini 3.5 flash-lite**: `standalone_query` lặp vô hạn (xem mục 9) |
| `SESSION_ENABLED` | Env, mặc định `false`. Tắt: bỏ qua `session_id`/`reset`, hành vi y hệt trước (không có `session_id` trong response) |
| `SESSION_STORE`, `SESSION_DB_URI` | Env, `memory` / rỗng. `sqlite` cần `SESSION_DB_URI=sqlite+aiosqlite:///đường/dẫn.db` và volume |
| `SESSION_TTL_S`, `SESSION_MAX_SESSIONS` | Env, `1800` / `200`. Hết hạn (từ lượt lưu cuối) hoặc vượt số session thì xóa (LRU) |
| `SESSION_MAX_TURNS`, `SESSION_MAX_HISTORY_TOKENS` | Env, `6` / `1500` (vẫn nhỏ vì Groq chỉ ~8K TPM). Lấy cái nhỏ hơn; mỗi tin cắt `SESSION_MESSAGE_MAX_CHARS`=1500 |

`config.py` gọi `load_dotenv(override=True)` nên **`.env` ghi đè biến môi trường**. Vì vậy `.env` bị loại khỏi image (`.dockerignore`) để không đè `QDRANT_URL` của compose. Hằng số `HYBRID_LEGS` ghi sai tên nhánh thì server vẫn khởi động (chỉ cảnh báo trong log) và mọi truy vấn hybrid lỗi `ValueError`. Kiểm tra ngay lúc import (sai thì `ValueError`, không in giá trị bí mật): `GUARD_MODE`, `SESSION_STORE`, `ROUTER_MODEL` khi `ROUTER_ENABLED`, `SESSION_DB_URI` khi `sqlite`.

## 8. Chạy và kiểm thử

- **Toàn hệ thống**: ở thư mục gốc repo, `docker-compose up -d --build`. Swagger của ai-service: http://localhost:8001/docs. Qua gateway: `http://localhost:9090/api/ai/health`.
- **Cần NVIDIA GPU** (compose đặt `deploy.resources.reservations.devices`). Máy không có GPU: xóa khối đó và build với `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`.
- Model BGE-M3 + reranker (~3GB) lưu ở volume `kltn_hf_cache`; lần đầu tải rất lâu.
- **Chạy tay**: `uvicorn app.api:app --port 8000` (cần Qdrant và `.env`).
- **Test** (trong thư mục `ai-service`, cấu hình ở `pytest.ini`): `python -m pytest -m "not integration"` là unit test, không cần mạng (router được thay tại `router_llm._complete`; encoder, Qdrant, reranker và Groq là bản giả). Test `live` gọi Gemini thật (tốn quota) bị bỏ qua mọi lúc trừ khi chọn rõ `python -m pytest -m live`. Đánh giá chất lượng router trên corpus: `python -m scripts.eval_router [--limit N] [--rpm 15] [--timeout 15]` (cần `GEMINI_API_KEY` + `GEMINI_MODEL`; chạy đủ ~370 lần gọi, `--limit 2` để chạy thử). Test `integration` cần Qdrant ở `QDRANT_URL` và chỉ dùng collection tạm `test_*` (tự xóa, không đụng collection thật): `tests/test_qdrant_hybrid.py`. `tests/test_encoders.py` cần thêm model BGE-M3 thật nên chạy trong container (dừng ai-service trước để có VRAM). Xem mục 10 về test còn hỏng.
- **Bytecode (`__pycache__`)**: `app/` không có `__init__.py` (namespace package, như `scripts/` và `tests/`) và dự án chỉ có **một** `__pycache__` ở gốc `ai-service`. `pytest` tự làm vậy nhờ [conftest.py](conftest.py). Với `uvicorn` hay script chạy tay trên máy host, đặt biến môi trường trước khi chạy, nếu không Python sẽ tạo lại `__pycache__` cạnh từng thư mục: PowerShell `$env:PYTHONPYCACHEPREFIX = "$PWD\__pycache__"`, bash `export PYTHONPYCACHEPREFIX="$PWD/__pycache__"`. Python gom cache vào một thư mục thì bỏ qua cache sẵn có của thư viện, nên lần chạy đầu chậm thêm khoảng 25 giây và cache nặng ~100MB (phần lớn là torch, transformers). Muốn không có cache nào thì đặt `PYTHONDONTWRITEBYTECODE=1`; Docker đã đặt sẵn nên container không có `__pycache__`.
- **Đo lại ngưỡng rerank trong container**: `docker-compose run --rm --no-deps ai-service python -m scripts.calibrate_rerank` (muốn đo trên collection/nhánh khác thì sửa `QDRANT_COLLECTION`/`HYBRID_LEGS` trong `app/config.py` và build lại; `-e` không còn đổi được chúng) (GPU 4GB không đủ chạy song song với ai-service đang chạy, hãy `docker-compose stop ai-service` trước).
- Tài khoản demo để gọi qua gateway: `demo_hsu` / `Password@123` (seed của auth-service). Token JWT sống 15 phút.

## 9. Bẫy đã gặp (đọc trước khi sửa)

- **Không thêm CORS ở đây.** Gateway đã gắn `Access-Control-Allow-Origin`; thêm một lần nữa ở ai-service làm response có 2 header, trình duyệt chặn `fetch`, FE rơi vào nhánh "mô phỏng" và báo "chưa kết nối được". `curl` không kiểm tra CORS nên không thấy lỗi này.
- **Re-embed phải xóa vector cũ** theo `document_id` (đã làm trong `embed_document`). `delete_document_vectors` chỉ ghi log khi lỗi chứ không ném ngoại lệ, nên nếu xóa lỗi thì vector cũ vẫn còn.
- Log `WARNING: Unsupported upgrade request` khi gateway gọi sang là **vô hại** (gateway thử nâng cấp HTTP/2 cleartext, uvicorn bỏ qua).
- Chất lượng trả lời phụ thuộc chất lượng text OCR của file-service (Tesseract hoặc ONNX). Sửa OCR xong phải **chạy lại OCR và embed** tài liệu thì Qdrant mới cập nhật.
- Collection cũ `paragraphs` (`QDRANT_LEGACY_COLLECTION`) chỉ được đọc để lấy cấu hình HNSW, không ghi/xóa.
- **Không có `__init__.py`**: Python 3.3+ nhận thư mục là package (namespace package). Cần thêm lại `__init__.py` rỗng nếu đóng gói bằng setuptools/PyInstaller, hoặc nếu có thư mục khác tên `app` nằm trên `sys.path` (namespace package sẽ gộp lẫn hai thư mục).
- **Không còn luật từ khóa nào; Gemini là tuyến phòng thủ đầu tiên và nó có thể sai.** Tấn công diễn đạt lạ vẫn có thể lọt. Lớp bảo vệ thật nằm ở prompt có rào + canary + guard đầu ra + chuẩn hóa/vô hiệu thẻ phân cách. Router tắt hoặc lỗi thì **mọi** câu hỏi (kể cả chào hỏi, tấn công) vào thẳng RAG. Không còn quét đoạn trích (indirect injection): một đoạn tài liệu chứa chỉ dẫn chỉ còn bị chặn bởi prompt rào và guard đầu ra. Đo bằng `scripts/eval_router.py`, không bằng cảm giác.
- **Cấu hình Gemini đã đo thật (2026-10, `gemini-3.5-flash-lite`).** `reasoning_effort="none"` (thinkingLevel minimal) làm `standalone_query` lặp vô hạn đến hết `max_tokens` (7/8 lần JSON bị cắt, ~5 s) nên mặc định là `low`. Mọi trường của `RouteDecision` phải **bắt buộc** trong schema (không đặt `= None`): trường tùy chọn bị Gemini bỏ qua, `clarify` mất `missing` và bị coi là rag. LiteLLM từng cảnh báo `temperature` của Gemini 3 sẽ bị gỡ; ta vẫn gửi `0` (live test qua). Độ trễ một lần gọi đo được 1,1–8,8 s (p50 ≈ 2,4 s, p95 ≈ 8,8 s, ~4.200 token prompt) nên `ROUTER_TIMEOUT_S` mặc định là 10 (với 4 một phần request rơi vào fallback RAG). Xấu nhất người dùng chờ `(1 + ROUTER_MAX_RETRIES) × ROUTER_TIMEOUT_S` = 20 s trước khi vào RAG; hạ `ROUTER_MAX_RETRIES` xuống 0 nếu muốn rút còn 10 s.
- **Gói Gemini miễn phí cho Google dùng nội dung để cải thiện sản phẩm** (theo trang pricing, đọc 2026-10), mà router gửi câu hỏi và vài lượt lịch sử của người dùng thật; gói trả phí thì không. Trang rate-limits của Google không công bố RPM free tier ("xem trong AI Studio"): `ROUTER_RPM=15` là số LiteLLM từng ghi, hãy đối chiếu AI Studio.
- **Hạn mức Gemini miễn phí đo thật 2026-10-24** (lỗi `RESOURCE_EXHAUSTED`, router rơi về RAG với câu hỏi gốc nên câu hỏi nối tiếp mất ngữ cảnh): `gemini-3.5-flash-lite` 500 yêu cầu/**ngày**; `gemini-3.5-flash` 20 yêu cầu/ngày và 5 yêu cầu/phút (nhỏ hơn `ROUTER_RPM`=15, đặt `ROUTER_RPM=5` nếu dùng); `gemini-2.5-*` không còn cấp cho tài khoản mới. Hạn mức tính theo project + model và reset lúc nửa đêm giờ Thái Bình Dương (khoảng 14h giờ VN). Mỗi câu hỏi tốn một yêu cầu, chạy `scripts/eval_router` và test `live` ăn vào cùng hạn mức với người dùng thật. Thoát: bật billing cho project, dùng key của project khác, hoặc chờ reset.
- **Câu hỏi trỏ vào mục của câu trả lời trước ("giải thích rõ điều 2").** Hai nguyên nhân từng làm nó hiểu thành Điều 2 của mọi văn bản: (1) lịch sử bị cắt 300 ký tự cho router và 600 cho mô hình trả lời nên mục 2, 3 của câu trả lời dài nằm ngoài tầm nhìn (nay cả hai 1500); (2) không có quy tắc phân biệt "điều N" là mục trong danh sách hay Điều N của một văn bản (nay có ở `ROUTER_PROMPT` quy tắc 7, kèm hai ví dụ, và ở quy tắc 4 của prompt trả lời trong `guard._SECURITY_RULES`). Câu có nêu tên văn bản ("Điều 19 của Quy chế đào tạo") vẫn là tham chiếu Điều. `ROUTER_PROMPT_VERSION` = 2. Chưa chạy lại `eval_router` sau thay đổi này vì hết hạn mức.
- **Gọi Gemini bằng `httpx`, không qua LiteLLM.** `router_llm._complete` POST `generateContent` (`systemInstruction`, `responseMimeType=application/json`, `responseJsonSchema` = schema pydantic của `RouteDecision`, `thinkingConfig.thinkingLevel` = `ROUTER_REASONING_EFFORT`, `none` đổi thành `minimal`). Timeout tổng mỗi lần do `asyncio.timeout` lo (httpx đặt `timeout=None`). Thử lại `ROUTER_MAX_RETRIES` lần khi timeout / mất kết nối / 5xx; 429 và 4xx khác lỗi ngay (thử lại 429 ngay lập tức vô ích, `ROUTER_RPM` đã chặn phía ta). Mỗi lần route mở một `AsyncClient` mới (thêm một lần bắt tay TLS, ~100–200 ms): đổi sang client dùng chung nếu đo thấy đáng kể (cần gắn với event loop, vì `eval_router.py` và live test gọi `asyncio.run` nhiều lần). Không còn model dự phòng (`ROUTER_FALLBACK_MODEL` của LiteLLM chưa từng được khai báo). Env `GEMINI_MODEL` → hằng số `ROUTER_MODEL`, bắt buộc khi `ROUTER_ENABLED`.
- **`session_id` là bí mật mang theo người dùng (bearer) nếu không có danh tính tin cậy.** Gateway gắn `X-User-Id` sau khi xác thực JWT (`JwtAuthFilter`; đã đọc code, chưa kiểm bằng request thật) và ai-service dùng nó làm khóa `user:session`. Nhưng cổng `8001` của ai-service được compose mở ra máy host: gọi thẳng vào đó thì client tự đặt được `X-User-Id`. FE hiện **chưa** gửi lại `session_id` nên mỗi câu hỏi là một session mới cho đến khi FE lưu và gửi lại giá trị server trả về.
- **Session trong RAM mất khi restart**, mỗi worker một kho riêng (Dockerfile chạy 1 worker nên ổn). `SESSION_STORE=sqlite` cần sửa `docker-compose.yml` gốc (chưa sửa): thêm cho `ai-service` `volumes: - session_data:/data/sessions`, khai báo `session_data:` ở cuối file, đặt `SESSION_DB_URI=sqlite+aiosqlite:////data/sessions/sessions.db`; thư mục `/data/sessions` phải ghi được bởi user `app` (Dockerfile cần `mkdir`/`chown` như `/data/groq_cache`). Session **không** lưu vĩnh viễn: hết hạn `SESSION_TTL_S`, bị LRU loại khi vượt `SESSION_MAX_SESSIONS` (xóa luôn dòng trong DB) hoặc `reset=true` đều xóa; chỉ giữ `SESSION_MAX_TURNS` lượt gần nhất.
- **Lock session và stream.** Lock giữ từ lúc mở session đến hết stream; nhả ở `finally` của generator và bằng `BackgroundTask` của response (client ngắt sớm). Đường mới nào giữ session phải nhả trên mọi nhánh lỗi. `SESSION_ENABLED=false`: `clarify` vẫn chạy nhưng không thấy lượt trước nên câu hoàn tất phải là câu hỏi đầy đủ, và không có chặn "hỏi hai lần liên tiếp".
- **LlamaIndex Memory**: hạn mức token đếm bằng `tokenizer_fn` của ta (2 ký tự/token) chứ không phải tiktoken (tránh tải từ vựng qua mạng); Memory luôn giữ ít nhất một lượt gần nhất kể cả khi vượt hạn mức. Cắt theo số lượt/token khi đưa vào prompt do `session._window` làm.
- `qm.main()` (CLI) và gọi thẳng `qm.ask()` không đi qua router/session; chỉ có prompt có rào và guard đầu ra (nằm trong `qm`).
- **Không dùng BM25.** Từng có nhánh BM25 (token hóa âm tiết + bigram, IDF do Qdrant nhân) và script chép collection sang bản có vector `bm25`; đã gỡ hẳn vì dự án chỉ dùng `dense` + `sparse` của BGE-M3. Collection `quy_dinh_hybrid` chưa bao giờ có vector này, bản thử `quy_dinh_hybrid_v2` đã xóa. Code cũ còn trong lịch sử git; đừng thêm lại nếu không có lý do mới.
- **Filter phải nằm trong từng Prefetch.** Đã thử trên Qdrant 1.19.1 + client 1.19.1: filter cấp cao nhất đứng một mình cũng lọc được các prefetch, nhưng không nên dựa vào đó (phiên bản khác có thể khác); và gắn filter cho chỉ một nhánh thì nhánh kia vẫn trả đoạn ngoài phạm vi. `retrieve()` gắn cùng một filter vào mọi Prefetch.

## 10. Tài liệu/code đã lỗi thời (đừng tin)

- [STRUCTURE.md](STRUCTURE.md): nhắc `scripts/export_paragraphs.py`, `scripts/index_qdrant.py`, `data/docs/`, `docker-compose.yml` riêng của ai-service. Các file này **không còn**; việc lập chỉ mục giờ qua RabbitMQ/`/embed`.
- Docstring ở đầu [app/api.py](app/api.py) ghi `uvicorn api:app`; lệnh đúng là `uvicorn app.api:app`. Mục "Chạy local" của [API.md](API.md) cũng còn nhắc `index_qdrant.py`, `uvicorn api:app` và CORS `*`.
- `PARAGRAPHS_PATH` trong config và `pymupdf` trong requirements là dấu vết của script cũ. Script `index_qdrant.py` từng tạo collection bằng cách chép cấu hình HNSW từ collection cũ `paragraphs`; nó đã bị xóa, nơi tạo collection duy nhất hiện nay là `qdrant_setup.create_collection()` (dùng `HNSW_M`/`HNSW_EF_CONSTRUCT` trong config).
- `test_reranker.py::test_vendored_modeling_is_the_pinned_upstream_plus_the_documented_change` còn lỗi vì file vendored không còn dòng header mà test tìm (có từ trước).
- `.env` đang chứa API key Groq thật; nên đổi key và đảm bảo file này không bị commit.
