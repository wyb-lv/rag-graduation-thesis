# Cấu trúc Dự án RAG Hybrid (rag-graduation-thesis)

Tài liệu này mô tả chi tiết kiến trúc thư mục hiện tại, vai trò của từng tệp mã nguồn, các thay đổi sau khi tái cấu trúc và hướng dẫn khởi chạy nhanh cho người mới.

---

## 1. Cây thư mục tổng quan

.
├── app/                          # Mã nguồn ứng dụng chính (Runtime & Serving)
│   ├── __init__.py               # Định nghĩa app là một Python package
│   ├── config.py                 # Cấu hình tập trung, đọc biến môi trường từ .env
│   ├── api.py                    # FastAPI app, REST API (/ask) & SSE Stream (/ask/stream)
│   ├── encoders.py               # Quản lý Singleton BGE-M3 (dense/sparse) & Jina Reranker v3.5
│   ├── jina_reranker_modeling.py # Mã nguồn vendored của Jina Reranker (không cần remote code)
│   ├── groq_client.py            # Groq LLM client (Rate limiting, Exponential Backoff, Disk Cache)
│   └── query_model.py            # RAG pipeline: Hybrid search (RRF) -> Rerank -> Prompt -> LLM
│
├── scripts/                      # Các công cụ CLI chạy offline / một lần (Data Ingestion)
│   ├── export_paragraphs.py      # Tách PDF trong data/docs/ thành các đoạn văn bản (paragraphs)
│   └── index_qdrant.py           # Mã hóa BGE-M3 và nạp dữ liệu vào vector database Qdrant
│
├── data/                         # Dữ liệu tài liệu và kết quả xử lý
│   ├── docs/                     # Chứa các văn bản PDF gốc (quy_che.pdf, QuyDinhHocBong.pdf)
│   └── paragraphs.txt            # Dữ liệu đoạn văn trích xuất từ PDF (tạo bởi export_paragraphs.py)
│
├── Dockerfile                    # Kịch bản đóng gói ứng dụng (hỗ trợ cả CPU và GPU NVIDIA)
├── docker-compose.yml             # Điều phối đồng thời service Qdrant và AI FastAPI Service
├── requirements.txt              # Danh sách thư viện phụ thuộc
├── constraints.txt               # Ghim cứng phiên bản chính xác để tránh xung đột
├── API.md                        # Đặc tả chi tiết các API HTTP/SSE
├── structure.md                  # Hướng dẫn cấu trúc thư mục (file này)
└── .env                          # Biến môi trường bí mật (API key, cấu hình model...)

---

## 2. Chi tiết trách nhiệm từng module

### 2.1. Thư mục app/ (Runtime Package)
Chứa toàn bộ logic phục vụ cho việc vận hành server và pipeline hỏi đáp:

* config.py:
  * Tự động xác định BASE_DIR ở thư mục gốc để nạp cấu hình từ .env.
  * Quản trị tập trung các tham số phần cứng (EMBED_DEVICE, RERANK_DEVICE), batch size, kích thước vector (EMBED_DIM=1024).
  * Quản lý tham số HNSW của Qdrant và các ngưỡng lọc độ tương đồng (SCORE_THRESHOLD, RERANK_MIN_SCORE).

* encoders.py:
  * Nạp và quản lý 2 mô hình AI: BGE-M3 (dense + sparse lexical weights) và Jina Reranker v3.5.
  * Quản lý khóa luồng _infer_lock đảm bảo suy luận GPU an toàn khi có nhiều request đồng thời.
  * Tự động xử lý phục hồi khi gặp lỗi tràn bộ nhớ (CUDA OOM): tự giảm batch size hoặc chuyển sang chạy CPU.

* jina_reranker_modeling.py:
  * Bản sao mã nguồn nội bộ của mô hình Jina Reranker v3.5 đã qua kiểm duyệt bảo mật.
  * Giúp hệ thống không cần bật cờ nguy hiểm trust_remote_code=True.

* groq_client.py:
  * Giao tiếp với API Groq (Qwen) qua cơ chế RateLimiter kiểm soát số lượt (RPM) và token (TPM).
  * Bộ đệm đĩa (.groq_cache/) ghi nguyên tử (atomic write) lưu câu trả lời theo mã băm SHA-256 của prompt để tiết kiệm chi phí và tăng tốc phản hồi.
  * Tích hợp ThinkFilter tự động loại bỏ các thẻ suy nghĩ của reasoning model khi streaming token.

* query_model.py:
  * Điều phối pipeline RAG: Xây dựng bộ lọc Qdrant (kho, ngày hiệu lực reference_date), truy vấn song song Dense + Sparse, kết hợp thứ hạng bằng thuật toán RRF (Reciprocal Rank Fusion), chấm lại điểm bằng Reranker.

* api.py:
  * Điểm vào của server FastAPI.
  * Cơ chế Warmup nền: Nạp trước mô hình BGE-M3 và Reranker ngay khi khởi động thông qua lifespan trên luồng phụ để không làm nghẽn server.
  * Cung cấp các endpoint: GET /health, POST /ask (trả lời JSON) và POST /ask/stream (trả lời stream qua Server-Sent Events).

### 2.2. Thư mục scripts/ (Công cụ CLI Ingestion)
Chạy độc lập khi cần chuẩn bị hoặc cập nhật dữ liệu vào cơ sở dữ liệu:

* export_paragraphs.py: Sử dụng PyMuPDF quét toàn bộ PDF trong data/docs/, nhận diện tiêu đề, gom cụm các khối chữ thành từng đoạn văn hợp lý (tối đa 700 ký tự) và lưu ra data/paragraphs.txt.
* index_qdrant.py: Tạo bộ sưu tập (Collection) trên Qdrant với 2 vector named (dense và sparse), thiết lập payload index cho các trường lọc (kho, doi_tuong, hieu_luc_tu, hieu_luc_den), mã hóa nội dung bằng BGE-M3 và nạp vào Qdrant.

### 2.3. Thư mục data/
* data/docs/: Chứa các văn bản quy định gốc dạng file PDF.
* data/paragraphs.txt: Tệp văn bản chứa các đoạn đã trích xuất, làm đầu vào để nạp vector vào Qdrant.

---

## 3. Lịch sử tái cấu trúc (Changelog Refactoring)

Dự án đã được tổ chức lại từ cấu trúc phẳng (Flat - tất cả file nằm ở root) sang cấu trúc phân lớp theo chức năng:

| Vị trí cũ ở Root | Vị trí mới | Nội dung thay đổi |
|---|---|---|
| api.py | app/api.py | Đưa vào package app. Import nội bộ dạng relative: from . import config, encoders, query_model. |
| config.py | app/config.py | Cập nhật BASE_DIR = dirname(dirname(...)) để trỏ đúng ra root chứa file .env. |
| encoders.py | app/encoders.py | Đưa vào package app. |
| groq_client.py | app/groq_client.py | Đưa vào package app. |
| query_model.py | app/query_model.py | Đưa vào package app. |
| jina_reranker_modeling.py | app/jina_reranker_modeling.py | Đặt chung trong package app. |
| export_model.py | scripts/export_paragraphs.py | Đổi tên cho rõ chức năng xuất đoạn văn từ PDF, đặt trong scripts/. |
| index_qdrant.py | scripts/index_qdrant.py | Đặt trong scripts/, bổ sung sys.path để import module từ app. |
| *.pdf | data/docs/*.pdf | Tách dữ liệu tài liệu gốc vào data/docs/. |
| paragraphs.txt | data/paragraphs.txt | Đưa file dữ liệu trung gian vào data/. |
| paragraph_embeddings.txt | (Đã xóa) | File nhúng tĩnh cũ không còn dùng (Qdrant tự sinh embedding khi nạp). |

---

## 4. Hướng dẫn khởi chạy nhanh

### Cách 1: Chạy trực tiếp ở máy cục bộ (Local Development)

1. Cài đặt các thư viện cần thiết:
   uv pip install -r requirements.txt -c constraints.txt

2. Xử lý tài liệu PDF và nạp vào Qdrant (yêu cầu Qdrant đang bật tại localhost:6333):
   python scripts/export_paragraphs.py
   python scripts/index_qdrant.py --recreate

3. Khởi chạy API Server:
   uv run uvicorn app.api:app --reload --port 8000

### Cách 2: Chạy đồng bộ qua Docker Compose (Khuyên dùng)

1. Khởi động đồng thời Qdrant và AI Service:
   docker compose up -d

2. Theo dõi log nạp model và khởi động:
   docker compose logs -f ai-service

Swagger UI tài liệu API kiểm thử tại: http://localhost:8000/docs