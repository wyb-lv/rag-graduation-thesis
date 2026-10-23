# RAG QA API

API tra loi cau hoi dua tren tai lieu (RAG hybrid). Backend: FastAPI (`api.py`).

Luong xu ly: cau hoi -> chuan hoa -> guard dau vao -> small talk router (chao hoi tra loi co dinh, khong di tiep) -> BGE-M3 (`FlagEmbedding`, dense + sparse trong 1 lan encode) -> Qdrant hybrid search (prefetch dense + sparse, hop nhat RRF, loc payload) -> top 20 -> `jina-reranker-v3.5` (listwise, nap qua transformers) -> top 5 doan -> quet doan trich (bo doan chua chi dan chen vao) -> prompt co rao -> Groq Qwen -> guard dau ra -> cau tra loi co trich dan.

## Chay local

```bash
cd Model
pip install -r requirements.txt   # torch cai rieng ban CUDA, xem ghi chu dau requirements.txt
docker compose up -d              # Qdrant tai http://localhost:6333, du lieu luu trong docker volume
python index_qdrant.py            # tao collection MOI quy_dinh_hybrid + payload index, ma hoa va nap paragraphs.txt
uvicorn api:app --reload --port 8000
```

`index_qdrant.py` khong bao gio ghi/xoa collection cu `paragraphs`. Chay lai khi du lieu doi: `python index_qdrant.py --recreate` (chi xoa va tao lai collection moi).

CLI: `python query_model.py [--mode dense|hybrid|hybrid_rerank] [--kho KHO] [--date YYYY-MM-DD] "cau hoi"`.

Cau hinh tap trung o `config.py`. Chi **bon** bien doc tu moi truong (`.env` cua ai-service, `environment` cua docker-compose, `ENV` cua Dockerfile): `GROQ_API_KEY`, `GROQ_QWEN_MODEL`, `QDRANT_URL`, `GROQ_CACHE_DIR`. Moi gia tri khac la hang so trong `config.py` (ten trong bang la ten hang so; bool ghi `1`/`0` la `True`/`False`): sua o do roi `docker-compose up -d --build ai-service`.

| Bien | Mac dinh | Y nghia |
|---|---|---|
| `GROQ_QWEN_MODEL` | (bat buoc) | ID model Qwen tren Groq, vi du `qwen/qwen3.8-27b` |
| `GROQ_API_KEY` | (bat buoc) | Groq API key (ten cu `API_KEY` khong con duoc doc) |
| `GROQ_REASONING_EFFORT` | `none` | Tat reasoning cua Qwen; de trong = khong gui tham so |
| `GROQ_RPM` / `GROQ_TPM` | `30` / `8000` | Gioi han toc do phia client |
| `GROQ_MAX_RETRIES` | `5` | So lan thu lai khi 429/timeout (backoff mu, ton trong `Retry-After`) |
| `GROQ_CACHE` / `GROQ_CACHE_DIR` | `1` / `.groq_cache` | Cache dia theo hash (model, prompt) |
| `WARMUP` | `1` | Nap BGE-M3 + reranker ngay khi server khoi dong (thread nen) thay vi o cau hoi dau tien. `0` de tat (vd khi dev can `--reload` nhanh) |
| `RAG_MODE` | `hybrid_rerank` | `dense` / `hybrid` / `hybrid_rerank` |
| `TOP_K_RETRIEVAL` / `TOP_K_RERANK` | `20` / `5` | So ung vien lay tu Qdrant / so doan dua vao LLM |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://localhost:6333` / `quy_dinh_hybrid` | |
| `EMBED_DEVICE` / `RERANK_DEVICE` | tu chon | `cuda:0` hoac `cpu`. Dat `RERANK_DEVICE=cpu` de nhuong ~1.1GB VRAM |
| `RERANK_MODEL` / `RERANK_REVISION` | `jinaai/jina-reranker-v3.5` / commit da ghim | Weights, config, tokenizer lay dung revision nay. Doi revision thi phai duyet va vendor lai `jina_reranker_modeling.py` |
| `RERANK_DTYPE` | `bfloat16` | dtype tren GPU (tren CPU luon `float32`) |
| `RERANK_CPU_ON_OOM` | `0` | CUDA OOM ngay ca khi cham 1 doan/lan thi chuyen reranker sang CPU |
| `EMBED_BATCH_SIZE` / `EMBED_MAX_LENGTH` | `16` / `512` | Tu dong giam batch khi OOM |
| `RERANK_BATCH_SIZE` / `RERANK_MAX_LENGTH` | `64` / `512` | So doan cham chung trong 1 prompt (chia doi khi OOM) / so token toi da moi doan |
| `RERANK_MIN_SCORE` | `0.1` | Diem rerank (cosine, -1..1) toi thieu cua mot doan (mode `hybrid_rerank`); `-1` de tat |
| `SCORE_THRESHOLD` | `0.5` | Cosine toi thieu (chi mode `dense`) |
| `HNSW_EF` | `128` | efSearch cua nhanh dense |
| `HYBRID_LEGS` | `dense,sparse` | Cac nhanh cua mode `hybrid` / `hybrid_rerank`, lay tu `dense`, `sparse`, `bm25` (phan cach bang dau phay). `bm25` can collection co vector `bm25` (chay `scripts/migrate_add_bm25.py`), thieu thi server khong khoi dong. Mode `dense` khong doi |
| `BM25_TOKENIZER` | `syllable_bigram` | `syllable`, `syllable_bigram` hoac `pyvi`. Phai giong luc index |
| `BM25_K1` / `BM25_B` | `1.2` / `0.75` | Tham so BM25. Phai giong luc index |
| `BM25_AVG_LEN` | `120.0` (gia tri tam) | So token BM25 trung binh moi chunk; dung gia tri `scripts/migrate_add_bm25.py` do va in ra. Phai giong luc index |
| `PAYLOAD_FIELD_LOAI_VAN_BAN` | `loai_van_ban` | Ten field payload dung de loc theo `loai_van_ban` |
| `PAYLOAD_FIELD_KHO`, `..._DOI_TUONG`, `..._HIEU_LUC_TU`, `..._HIEU_LUC_DEN` | `kho`, `doi_tuong`, `hieu_luc_tu`, `hieu_luc_den` | Ten field payload dung de index/loc |
| `GUARD_ENABLED` | `1` | `0` tat guard + small talk: hanh vi y het truoc khi co guard (chi them truong `intent` = `rag`) |
| `GUARD_MODE` | `enforce` | `log_only` chi ghi log (khong chan cau hoi, khong bo doan, khong chan dau ra, khong sua cau hoi) de do false positive; small talk van tra loi co dinh |
| `GUARD_MAX_QUESTION_CHARS` | `1000` | Cau hoi dai hon bi cat (sau khi chuan hoa) |
| `GUARD_CANARY` | (de trong) | De trong = suy ra tu noi dung system prompt, on dinh giua cac lan chay nen cache dia Groq van trung |
| `GUARD_REFUSAL_ANSWER`, `GUARD_REPLY_GREETING` / `_CAPABILITY` / `_META` / `_THANKS` / `_GOODBYE` | xem `config.py` | Cau tra loi co dinh |

Base URL mac dinh: `http://localhost:8000`

## Ba mode pipeline

| `mode` | Truy van Qdrant | Rerank |
|---|---|---|
| `dense` | chi vector `dense` (cosine) | khong |
| `hybrid` | prefetch cac nhanh trong `HYBRID_LEGS` (mac dinh `dense` + `sparse`, them `bm25` neu bat; moi nhanh 20), hop nhat RRF o phia Qdrant | khong |
| `hybrid_rerank` | nhu `hybrid` | `jina-reranker-v3.5`, giu 5 doan |

Moi nhanh la mot `Prefetch` trong cung mot request toi Qdrant va **cung mang mot bo loc** (`kho`, hieu luc, `khoa`, `loai_van_ban`). Nhanh `dense`/`sparse` la vector BGE-M3 (GPU); `bm25` la nhanh tu khoa tieng Viet (CPU, giu dau, them cap am tiet nhu `hoc_bong`, `dieu_15`), IDF do Qdrant tinh.

Nhanh nao khong co token (vd cau hoi chi co dau cau) thi bi bo; khong con nhanh nao thi dung `dense`. Chi con mot nhanh thi truy van thang nhanh do, khi do `score` la diem cua nhanh chu khong phai diem RRF.

## Endpoints

### GET /health

Kiem tra server con song va trang thai nap model.

**Response `200`**
```json
{ "status": "ok", "models": "ready" }
```

| `models` | Y nghia |
|---|---|
| `pending` | Dang nap model (~15s sau khi server len). Request toi luc nay van duoc xu ly, chi cho nap xong |
| `ready` | Model da nap va chay thu xong, cau hoi tra loi ngay |
| `error` | Warmup loi (xem log server); model se duoc nap lai o request dau tien |
| `disabled` | `WARMUP=0`: model nap o request dau tien |

### POST /ask

Gui cau hoi, nhan cau tra loi cua LLM kem nguon trich dan.

**Request body**
```json
{
  "question": "Thi sinh duoc dang ky toi da bao nhieu nguyen vong?",
  "mode": "hybrid_rerank",
  "top_k": 5,
  "kho": "Khoa A",
  "reference_date": "2026-01-31",
  "loai_van_ban": ["thong_bao", "quy_che"]
}
```

| Field | Type | Bat buoc | Mac dinh | Ghi chu |
|---|---|---|---|---|
| `question` | string | co | - | Khong duoc rong |
| `mode` | string | khong | `RAG_MODE` | `dense`, `hybrid` hoac `hybrid_rerank` |
| `top_k` | int | khong | `5` | So doan van lay lam ngu canh (1-10) |
| `kho` | string | khong | - | Chi tim trong kho nay |
| `reference_date` | date | khong | hom nay | Chi lay van ban con hieu luc tai ngay nay |
| `loai_van_ban` | string[] | khong | - | Chi tim trong cac loai van ban nay: `quy_che`, `quy_dinh`, `thong_bao`, `bieu_mau`, `huong_dan`, `khac` (chunker suy ra tu ten tai lieu). Bo trong hoac `[]` = khong loc; gia tri khong ton tai thi khong co doan nao (`in_scope=false`) |

Loc theo ngay: van ban hop le khi `hieu_luc_tu <= reference_date <= hieu_luc_den`; thieu `hieu_luc_tu` / `hieu_luc_den` (hoac `null`) coi nhu khong gioi han o phia do. Payload hien tai chi co `source` va `text` nen cac bo loc nay chua loai doan nao cho den khi bo chunking ghi them `kho`, `doi_tuong`, `hieu_luc_tu`, `hieu_luc_den`.

**Response `200`**
```json
{
  "answer": "Thi sinh duoc dang ky toi da 15 nguyen vong [quy_che.pdf].",
  "in_scope": true,
  "sources": [
    {
      "source": "quy_che.pdf",
      "text": "Thi sinh duoc dang ky toi da 15 nguyen vong ...",
      "score": 0.6004,
      "rerank_score": 0.6004,
      "payload": { "source": "quy_che.pdf", "text": "Thi sinh duoc dang ky toi da 15 nguyen vong ..." }
    }
  ],
  "intent": "rag"
}
```

Prompt yeu cau chi tra loi tu cac doan duoc cung cap, trich dan ten van ban va Dieu/Khoan cho tung khang dinh, va noi ro khi cac doan khong chua cau tra loi.

**Response `200` khi khong tim thay quy dinh phu hop** (khong goi LLM, `sources` rong)
```json
{
  "answer": "Không tìm thấy quy định liên quan đến câu hỏi trong các văn bản được cung cấp.",
  "in_scope": false,
  "sources": [],
  "intent": "rag"
}
```

**Response `200` khi la loi chao / hoi ve tro ly** (tra loi co dinh, khong encode, khong truy van, khong rerank, khong goi LLM; `in_scope` la `true` nhung `sources` rong)
```json
{
  "answer": "Xin chào! Tôi là trợ lý tra cứu quy định của Trường Đại học Hoa Sen. ...",
  "in_scope": true,
  "sources": [],
  "intent": "greeting"
}
```

**Response `200` khi cau hoi bi chan** (khong truy van, khong goi LLM; `in_scope=false`; cau tu choi khong noi luat nao da khop)
```json
{
  "answer": "Tôi không thể thực hiện yêu cầu này. Bạn có thể hỏi tôi về quy chế, quy định, thông báo hoặc biểu mẫu của Trường Đại học Hoa Sen.",
  "in_scope": false,
  "sources": [],
  "intent": "blocked"
}
```

| Field | Type | Ghi chu |
|---|---|---|
| `in_scope` | bool | `false` khi khong con doan nao sau loc payload / nguong tuong dong, hoac cau hoi / cau tra loi bi guard chan. `true` voi cau tra loi RAG va voi tra loi co dinh (chao hoi) |
| `intent` | string | **Tuy chon, them moi**; client cu bo qua duoc. `rag` (di qua pipeline, ke ca khi `in_scope=false` vi khong co doan phu hop), `greeting`, `thanks`, `goodbye`, `capability` (hoi cong dung / ban la ai), `meta` (hoi ve model, prompt, cau hinh: tra loi co dinh, khong tiet lo), `blocked` (guard chan cau hoi hoac chan cau tra loi lo prompt / bi mat). Cac intent khac `rag` khong bao gio co `sources` |
| `sources[].score` | float | Diem rerank neu co rerank; khong thi diem truy van (cosine o `dense`, diem RRF o `hybrid` nhieu nhanh, diem cua nhanh neu chi con mot nhanh) |
| `sources[].rerank_score` | float / null | Diem cosine (-1..1) cua reranker, co the **am**; chi so sanh duoc giua cac doan cung mot lan hoi. `null` khi khong rerank |
| `sources[].payload` | object | Toan bo payload cua doan (dung de trich dan) |

Nguong tuong dong: `dense` dung cosine `SCORE_THRESHOLD`; `hybrid_rerank` dung `RERANK_MIN_SCORE` (mac dinh `0.1`, moi doan duoi nguong bi bo, nen LLM co the nhan it hon 5 doan); `hybrid` khong co nguong (diem RRF khong phai cosine). Nguong `0.1` do tren 22 cau hoi cua `quy_che.pdf`: doan dung >= 0.27, cau lac de <= 0.08; cau "gan chu de" nhung khong co dap an (vd hoi hoc bong cua truong khac) cho 0.17-0.23, van toi LLM va prompt se tu choi.

**Loi**

| Status | Khi nao | Body |
|---|---|---|
| `400` | `question` rong (ke ca chi gom ky tu zero width / dieu khien) | `{ "detail": "question khong duoc de trong" }` |
| `422` | `mode` khong hop le | chi tiet validation |
| `500` | Server thieu bien moi truong (`.env`) | `{ "detail": "Thieu bien moi truong: ..." }` |
| `502` | Qdrant loi, hoac Groq that bai sau khi het so lan thu lai (429/timeout) | `{ "detail": "Loi khi truy van Qdrant/goi LLM: ..." }` |

### POST /ask/stream

Nhu `/ask` (cung request body) nhung tra cau tra loi dang **Server-Sent Events** (`Content-Type: text/event-stream`), tung doan token ngay khi Groq sinh ra. Dung `fetch` + `ReadableStream` (khong dung `EventSource` vi `EventSource` chi ho tro GET).

Moi su kien la mot dong `data: <JSON>` ket thuc bang dong trong, theo thu tu:

```
data: {"event": "sources", "in_scope": true, "sources": [{"source": "quy_che.pdf", "text": "...", "score": 0.6004, "rerank_score": 0.6004, "payload": {...}}]}

data: {"event": "delta", "text": "Thí sinh "}

data: {"event": "delta", "text": "được đăng ký tối đa 15 nguyện vọng [quy_che.pdf]."}

data: [DONE]
```

| Su kien | So lan | Noi dung |
|---|---|---|
| `sources` | 1, dau tien | `in_scope` + `sources` (cung dinh dang voi `/ask`) |
| `delta` | >= 1 | `text`: doan tiep theo cua cau tra loi, noi lai theo thu tu. Ngoai pham vi (`in_scope=false`): 1 `delta` duy nhat la cau "Không tìm thấy quy định..." va khong goi LLM. Chao hoi / bi chan: 1 `delta` duy nhat la cau tra loi co dinh |
| `error` | 0-1 | `detail`: loi khi dang sinh cau tra loi. Stream dong ngay sau do, **khong** co `[DONE]`. Guard dau ra phat hien canary / lo system prompt / bi mat giua chung thi `detail` la cau tu choi (`GUARD_REFUSAL_ANSWER`); cac `delta` da gui truoc do (khong chua du mau lo) khong thu hoi duoc |
| `[DONE]` | 1, cuoi cung | Cau tra loi da xong |

- Tra loi co dinh (chao hoi) va cau hoi bi chan van theo dung hop dong tren: `sources` voi danh sach rong (`in_scope=true` cho chao hoi, `false` cho cau bi chan), dung 1 `delta`, roi `[DONE]`. Su kien `sources` cua chung co them khoa `intent` (`greeting`, `thanks`, `goodbye`, `capability`, `meta`, `blocked`); su kien `sources` cua cau hoi RAG **khong** co khoa nay (giu nguyen nhu truoc).
- Truy van Qdrant + rerank chay **truoc** khi gui header, nen loi o buoc nay van tra ma HTTP `400/422/500/502` voi body JSON nhu `/ask`. Sau khi da nhan `200`, loi chi bao bang su kien `error`.
- Stream ket thuc ma khong co `[DONE]` (va khong co `error`) = mat ket noi giua chung, FE nen coi la loi.
- Groq 429/timeout truoc token dau tien van duoc thu lai nhu `/ask` (client cho lau hon, chua nhan `delta` nao). Loi sau khi da gui token thi khong thu lai, gui `error`.
- Cau tra loi day du duoc ghi vao cache dia nhu `/ask`; cau hoi lap lai tra ve trong 1 `delta` duy nhat.
- Khoi `<think>...</think>` (neu bat reasoning) duoc loc ngay trong stream.

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
  return res.json(); // { answer, in_scope, sources }
}
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Hoc bong Tinh hoa HSU giam bao nhieu hoc phi?"}'
```

Stream (FE day du: `ui-graduation-thesis/src/services/conversationService.ts`, ham `askQuestionStream`):

```js
async function askQuestionStream(question, { onSources, onDelta }) {
  const res = await fetch("http://localhost:8000/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || "Loi khong xac dinh");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) throw new Error("Mat ket noi truoc khi tra loi xong");
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop();
    for (const block of blocks) {
      const data = block.replace(/^data: /, "");
      if (data === "[DONE]") return;
      const ev = JSON.parse(data);
      if (ev.event === "sources") onSources(ev.sources, ev.in_scope);
      else if (ev.event === "delta") onDelta(ev.text);
      else if (ev.event === "error") throw new Error(ev.detail);
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

- BGE-M3 (~1.1GB VRAM) va reranker (~1.1GB) nap ngay khi server khoi dong (`WARMUP`, ~15s tren RTX 3050; lan dau con tai model tu Hugging Face) roi nam trong bo nho cho cac request sau. Reranker chi duoc nap truoc khi `RAG_MODE=hybrid_rerank`.
- Windows: goi `http://localhost:8000` tu Python/curl co the cham them ~2s moi ket noi (thu IPv6 `::1` truoc, uvicorn chi nghe IPv4). Dung `http://127.0.0.1:8000` de tranh. Truong hop xau nhat (20 doan x 512 token) dinh khoang 3.3GB tren GPU 4GB; sat gioi han thi dat `RERANK_DEVICE=cpu` hoac `RERANK_CPU_ON_OOM=1`.
- `jina-reranker-v3.5` dung giay phep **CC BY-NC 4.0** (chi phi thuong mai). Model duoc nap bang transformers qua `jina_reranker_modeling.py`: ban sao modeling.py cua model o revision da ghim (xem header file de biet nguon, giay phep, va dong duy nhat da sua), nen khong can `trust_remote_code` va khong chay code tai ve luc chay.
- Groq bi gioi han client-side 30 RPM / 8K TPM, nen request co the cho; cau hoi lap lai tra ngay tu cache dia.
- CORS dang mo cho moi origin (`*`) de FE goi truc tiep khi dev; can gioi han domain cu the khi deploy.
- Chay test: `pytest` (can model that + Qdrant dang chay); chi unit test: `pytest -m "not integration"`.
