# RAG QA API

API tra loi cau hoi dua tren tai lieu (RAG, phuong phap paragraph). Backend: FastAPI (`api.py`).

## Chay local

```bash
uv run uvicorn api:app --reload --port 8000
```

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
  "sources": [
    {
      "source": "quy_che.pdf",
      "text": "Thi sinh duoc dang ky toi da 15 nguyen vong ...",
      "score": 0.8662
    }
  ]
}
```

**Loi**

| Status | Khi nao | Body |
|---|---|---|
| `400` | `question` rong | `{ "detail": "question khong duoc de trong" }` |
| `500` | Server thieu bien moi truong (`.env`) | `{ "detail": "Thieu bien moi truong: ..." }` |
| `502` | Goi LLM that bai (timeout, 429 lien tuc, ...) | `{ "detail": "Loi khi goi LLM: ..." }` |

## Vi du goi tu FE

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

## Ghi chu

- Cau tra loi co the mat vai giay (goi LLM that + throttle ~2.2s/request).
- CORS dang mo cho moi origin (`*`) de FE goi truc tiep khi dev; can gioi han domain cu the khi deploy.
- Lan goi `/ask` dau tien se cham hon vi phai nap model embedding (`BAAI/bge-m3`) vao bo nho.
