# ChatHSU – Hoa Sen University regulations chatbot

Graduation thesis project: a chatbot that answers students' questions from the university's official regulations (RAG). An administrator uploads PDF / DOCX / scanned images, the system OCRs, chunks and indexes them automatically, and users ask questions by text or voice and get answers with source citations.

- **RAG question answering**: dense vector search (Vietnamese_Embedding_v2) and BM25 keyword search run together, are merged with RRF inside Qdrant, re-scored by a reranker, and an LLM (Groq Qwen) writes the answer with citations. Answers stream over SSE.
- **LLM router** (Gemini): recognises greetings, off-topic questions and follow-ups, and rewrites the question before retrieval. Prompt-injection guards on input and output.
- **OCR**: PDFs with a text layer and DOCX files are read directly (no recognition errors possible); scanned pages and images go through PaddleOCR-VL-1.6. Tables are kept as Markdown.
- **Voice**: speech-to-text (STT) and text-to-speech (TTS), plus a voice-call mode.
- **Accounts and roles**: JWT login; account management for ADMIN.

## Architecture

```
React (Vite, :5173)
      │
      ▼
api-gateway :9090 ── JWT, CORS, rate limit, aggregated Swagger UI
      │
      ├─► auth-service    :8081 (gRPC :9091) ──gRPC──► account-service :8082 (internal)
      ├─► file-service    :8080 ── MinIO (files) + Postgres
      │        │  OCR / embed jobs
      │        ▼
      │     RabbitMQ ───────────────┐
      │                             ▼
      └─► ai-service :8000 (host :8001) ──► Qdrant (vectors)
               │                      ├──► MongoDB (conversation history)
               │                      ├──► Gemini (router), Groq (Qwen, Whisper)
               └── OCR of scanned pages ──► PaddleOCR-VL (layout API + vLLM, needs a GPU)
```

| Component | Stack | Host port |
|---|---|---|
| `ui-graduation-thesis` | React 19, Vite 6, TypeScript, Tailwind 4 | 5173 (dev) |
| `api-gateway` | Spring Boot (Java 21), Spring Cloud Gateway | **9090** |
| `auth-service` | Spring Boot, Spring Security, JWT, Postgres | 8081, gRPC 9091 |
| `account-service` | Spring Boot, Postgres (own DB; auth-service calls it over gRPC) | not published |
| `file-service` | Spring Boot, MinIO, RabbitMQ, Postgres | 8080 |
| `ai-service` | FastAPI (Python), sentence-transformers, Qdrant | 8001 |
| Infrastructure | Qdrant 6333, RabbitMQ 5672 (UI 15672), MinIO 9000 (UI 9001), MongoDB 27018, Postgres 5433 / 5436 / 5437 | |

Spring Security lives only in `auth-service`. Services talk to each other over gRPC, not each other's REST APIs. The frontend **only calls the gateway**.

Main gateway routes (the full list is in the Swagger UI):

| Frontend calls (port 9090) | Forwarded to |
|---|---|
| `/api/accounts/**` | account-service (ADMIN only, except `/me`) |
| `/api/documents/**` | file-service |
| `/api/ai/ask`, `/api/ai/ask/stream`, `/api/ai/embed`, `/api/ai/health` | ai-service (`/api/ai` prefix stripped) |

### Document ingestion flow

1. An admin uploads a file with `POST /api/documents/upload`. file-service stores it in MinIO, writes a DB row (`PENDING`) and queues an OCR job in RabbitMQ.
2. ai-service takes the OCR job, downloads the file, extracts text and tables, and reports the text back to file-service (`COMPLETED`).
3. file-service queues an embed job. ai-service splits the text by Chapter / Article / clause, builds dense (Vietnamese_Embedding_v2) vectors and BM25 weights, and writes them to the `quy_dinh_hybrid_v3` collection in Qdrant. Re-embedding a document deletes its old vectors first.

Re-run OCR for one document: `POST /api/documents/{id}/ocr`. Re-embed without OCR: `POST /api/documents/{id}/embed`.

### Question answering flow

`POST /api/ai/ask` (or `/api/ai/ask/stream`) → normalise the question → Gemini router → rewrite the search query → hybrid search (dense + BM25, RRF) → rerank → score threshold → LLM answer with citations. If no passage passes the threshold, a fixed "no information" answer is returned without calling the LLM.

## Requirements

- Docker Desktop (WSL2 on Windows) and an **NVIDIA GPU** with a CUDA 12.6+ driver. The GPU is used by the embedding model, the reranker and PaddleOCR-VL.
- API keys for **Gemini** (router) and **Groq** (answer LLM, STT).
- To run services outside Docker: JDK 21, Python 3.13, Node.js.

## Quick start with Docker

```bash
# 1. Shared environment
cp .env.example .env
#    Edit .env: set JWT_SECRET and INTERNAL_TOKEN (each >= 32 chars, e.g. `openssl rand -hex 32`) and the passwords.

# 2. ai-service API keys (docker-compose reads this file; it is not committed)
cat > ai-service/.env <<'EOF'
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.5-flash-lite
GROQ_API_KEY=...
GROQ_QWEN_MODEL=qwen/qwen3.8-27b
STT_MODEL=whisper-large-v3
EOF

# 3. Build and start everything
docker compose up -d --build        # Compose v1: docker-compose
```

The first start is slow: PaddleOCR images are several GB, vLLM compiles for a few minutes, and ai-service downloads the embedding model + the reranker (~3 GB) into the `hf_cache` volume on the first request.

Then:

| URL | What |
|---|---|
| http://localhost:9090/swagger-ui.html | Swagger UI aggregating the auth, account and file services |
| http://localhost:8001/docs | ai-service Swagger |
| http://localhost:15672 / http://localhost:9001 | RabbitMQ / MinIO consoles (credentials in `.env`) |
| http://localhost:6333/dashboard | Qdrant |

With `ACCOUNT_SEED_ENABLED=true` (the default in `.env.example`), account-service creates the sample accounts `admin`, `employee1`, `employee2` and `demo_hsu` (passwords in [DataSeeder.java](account-service/src/main/java/com/example/accountservice/config/DataSeeder.java)). This is the only way to create the first Admin; set it to `false` outside development.

Sample regulation documents are in [docs/](docs/). Log in as Admin and upload them to build the index.

### Frontend

```bash
cd ui-graduation-thesis
cp .env.example .env                # VITE_API_BASE_URL=http://localhost:9090
npm install
npm run dev                         # http://localhost:5173
```

### 4 GiB GPUs

By default the PaddleOCR-VL vLLM server takes 60% of VRAM and profiles the vision encoder with `max-num-batched-tokens=131072`, which crashes on a 4 GiB GPU ("No available memory for the cache blocks"). Set in `.env`:

```
OCR_VLM_GPU_UTIL=0.8
OCR_VLM_MAX_SEQS=4
OCR_VLM_MAX_BATCHED_TOKENS=16384
```

This runs on an RTX 3050 Laptop 4 GiB, but very little VRAM is left for ai-service while OCR is loaded.

## Running services individually for development

```bash
docker compose up -d postgres-auth postgres-account postgres-file rabbitmq minio qdrant mongodb

cd ai-service && uvicorn app.api:app --reload --port 8000
cd file-service && ./mvnw spring-boot:run          # same for auth-service and api-gateway (account-service: mvn)
```

ai-service tests (no outbound network): `cd ai-service && pytest -m "not integration and not live"`. `integration` tests need the real models and Qdrant; `live` tests call the real Gemini API and use quota, so they only run with an explicit `-m live`.

## Configuration

| Variable | Meaning |
|---|---|
| `JWT_SECRET` | **Required.** JWT signing key (>= 32 chars) |
| `INTERNAL_TOKEN` | **Required.** Shared secret for file-service `/internal/**`; ai-service sends it as `X-Internal-Token` |
| `ACCOUNT_SEED_ENABLED` | Create the sample accounts at startup |
| `GATEWAY_PORT`, `CORS_ORIGINS`, `AI_RPS`, `FILE_RPS`, `AUTH_RPS` | Gateway port, allowed frontend origins, requests-per-second limits |
| `POSTGRES_*`, `AUTH_DB_*`, `ACCOUNT_DB_*`, `RABBITMQ_*`, `MINIO_*`, `MONGO_*` | Infrastructure credentials |
| `OCR_VLM_GPU_UTIL`, `OCR_VLM_MAX_SEQS`, `OCR_VLM_MAX_BATCHED_TOKENS`, `PADDLEOCR_IMAGE_TAG` | Tune PaddleOCR-VL to your GPU |
| `ai-service/.env` | `GEMINI_*`, `GROQ_*`, `STT_MODEL`; thresholds, models and devices are described in [ai-service/API.md](ai-service/API.md) |

## Further reading

The ai-service documents are written in Vietnamese.

- [ai-service/PROJECT_CONTEXT.md](ai-service/PROJECT_CONTEXT.md): OCR, embedding, question answering, router and guard; kept in sync with the code.
- [ai-service/API.md](ai-service/API.md): every endpoint, configuration parameters, frontend examples.
- [account-service/openapi.yaml](account-service/openapi.yaml): account management API.

## Notes

- file-service has no user authentication of its own (JWT is checked by the gateway), so it must only be reachable through the gateway. Its port 8080 is bound to `127.0.0.1` in `docker-compose.yml`; do not publish it on other interfaces.
- `/internal/**` on file-service (the OCR result callback from ai-service) requires the `X-Internal-Token` header to match `INTERNAL_TOKEN`; with no token configured it rejects every request. The gateway does not route `/internal/**` either.
- CORS is handled by the gateway; ai-service must not add `CORSMiddleware`.
- Before a real deployment, change `JWT_SECRET` and the infrastructure passwords and turn off `ACCOUNT_SEED_ENABLED`.
- After changing the OCR code you must re-run OCR **and** embedding for a document before Qdrant reflects the change.
