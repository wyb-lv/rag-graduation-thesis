# RAG QA API

API tra loi cau hoi dua tren tai lieu (RAG, phuong phap paragraph). Backend: FastAPI (`api.py`).

## Chay local

```bash
docker compose up -d           # Qdrant (HNSW) tai http://localhost:6333, du lieu luu trong docker volume
uv run python index_qdrant.py # nap paragraphs.txt + paragraph_embeddings.txt vao Qdrant
uv run uvicorn api:app --reload --port 8000
```

Bien moi truong tuy chon: `QDRANT_URL` (mac dinh `http://localhost:6333`), `QDRANT_COLLECTION` (mac dinh `paragraphs`), `HNSW_EF` (efSearch, mac dinh `128`).

Base URL mac dinh: `http://localhost:8000`

## Endpoints

### GET /health

Kiem tra server con song.

**Response `200`**
```json
{ "status": "ok" }
```

### POST /ask

Gui cau hoi, nhan cau tra loi cua LLM kem nguon trich dan.

**Request body**
```json
{
  "question": "Thi sinh duoc dang ky toi da bao nhieu nguyen vong?",
  "top_k": 1
}
```

| Field | Type | Bat buoc | Mac dinh | Ghi chu |
|---|---|---|---|---|
| `question` | string | co | - | Khong duoc rong |
| `top_k` | int | khong | `1` | So doan van lay lam ngu canh (1-10) |

**Response `200`**
```json
{
  "answer": "Thi sinh duoc dang ky toi da 15 nguyen vong.",
  "in_scope": true,
  "sources": [
    {
      "source": "quy_che.pdf",
      "text": "Thi sinh duoc dang ky toi da 15 nguyen vong ...",
      "score": 0.8662
    }
  ]
}
```

**Response `200` khi cau hoi nam ngoai pham vi tai lieu** (khong goi LLM, `sources` rong)
```json
{
  "answer": "Xin lỗi, câu hỏi này nằm ngoài phạm vi văn bản tôi được cung cấp nên tôi không thể trả lời.",
  "in_scope": false,
  "sources": []
}
```

| Field | Type | Ghi chu |
|---|---|---|
| `in_scope` | bool | `false` khi khong doan van nao dat nguong tuong dong (cosine < `SCORE_THRESHOLD`) |

Nguong mac dinh `0.57`, doi bang bien moi truong `SCORE_THRESHOLD` trong `.env`. Chi nhung doan van dat nguong moi duoc dua vao ngu canh cho LLM va tra ve trong `sources`.

**Loi**

| Status | Khi nao | Body |
|---|---|---|
| `400` | `question` rong | `{ "detail": "question khong duoc de trong" }` |
| `500` | Server thieu bien moi truong (`.env`) | `{ "detail": "Thieu bien moi truong: ..." }` |
| `502` | Goi LLM that bai (timeout, 429 lien tuc, ...) | `{ "detail": "Loi khi goi LLM: ..." }` |

### POST /ask/stream (SSE Streaming)

Gui cau hoi va nhan cau tra loi dang dong du lieu thoi gian thuc (**Server-Sent Events** - `text/event-stream`), giup giam thoi gian phan hoi ban dau (TTFT) xuong duoi 0.5s.

**Request body** *(Giong `POST /ask`)*
```json
{
  "question": "Thi sinh duoc dang ky toi da bao nhieu nguyen vong?",
  "top_k": 1
}
```

**Response `200`** (`Content-Type: text/event-stream`)

Du lieu tuon ve theo 3 giai doan:

1. **Su kien `sources`**: Ban nguon trich dan ngay khi tim kiem xong:
```text
data: {"event": "sources", "in_scope": true, "sources": [{"source": "quy_che.pdf", "text": "...", "score": 0.8662}]}

```

2. **Su kien `delta`**: Ban tung tu cau tra loi theo thoi gian thuc:
```text
data: {"event": "delta", "text": "Thi "}

data: {"event": "delta", "text": "sinh "}

data: {"event": "delta", "text": "duoc dang ky toi da 15 nguyen vong."}

```

3. **Tin hieu ket thuc**:
```text
data: [DONE]

```

*Neu cau hoi ngoai pham vi tai lieu (`in_scope: false`), server se gui `sources: []` va mot `delta` chua cau tu choi tra loi, roi ket thuc bang `[DONE]`.*

## Vi du goi tu FE

### 1. Goi REST thuong (POST /ask)


```js
async function askQuestion(question) {
  const res = await fetch("http://localhost:8000/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Loi khong xac dinh");
  }
  return res.json(); // { answer, sources }
}
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Hoc bong Tinh hoa HSU giam bao nhieu hoc phi?"}'
```

### 2. Goi SSE Streaming (POST /ask/stream)

```js
async function askQuestionStream(question, onSources, onDelta) {
  const res = await fetch("http://localhost:8000/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop();

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const raw = line.slice(6).trim();
        if (raw === "[DONE]") return;

        const payload = JSON.parse(raw);
        if (payload.event === "sources") {
          onSources(payload.sources, payload.in_scope);
        } else if (payload.event === "delta") {
          onDelta(payload.text);
        }
      }
    }
  }
}
```

```bash
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Hoc bong Tinh hoa HSU giam bao nhieu hoc phi?"}'
```

## Ghi chu

- Cau tra loi co the mat vai giay (goi LLM that + throttle ~2.2s/request).
- CORS dang mo cho moi origin (`*`) de FE goi truc tiep khi dev; can gioi han domain cu the khi deploy.
- Lan goi `/ask` dau tien se cham hon vi phai nap model embedding (`BAAI/bge-m3`) vao bo nho.
