# RAG QA API

API tra loi cau hoi dua tren tai lieu (RAG hybrid). Backend: FastAPI (`api.py`).

Luong xu ly: cau hoi -> chuan hoa -> mo session: lich su + khoa/kho -> router Gemini tu suy luan y dinh: chao hoi / tan cong / ngoai pham vi / thieu thong tin tra loi co dinh va dung o day, con lai duoc viet lai thanh cau tim kiem doc lap -> BGE-M3 (`sentence-transformers`, chi dense; nhanh lexical la BM25 tu viet) -> Qdrant hybrid search (loc payload ngay trong tung nhanh tim vector; prefetch dense + bm25, hop nhat RRF; Qwen viet lai cau tim kiem, dense + bm25 tim cho ca hai cau va RRF mot lan tren moi danh sach) -> top 20 -> `jina-reranker-v3.5` (listwise, nap qua transformers) -> top 5 doan -> prompt co rao (doan trich + lich su + cau hoi goc) -> Groq Qwen -> guard dau ra -> luu luot vao session -> cau tra loi co trich dan. Khong con luat tu khoa nao phan loai cau hoi: router loi thi cau hoi di thang vao RAG. Moi buoc tren luon chay, khong co cong tac bat / tat nao.

## Chay local

```bash
cd Model
pip install -r requirements.txt   # torch cai rieng ban CUDA, xem ghi chu dau requirements.txt
docker compose up -d              # Qdrant tai http://localhost:6333, du lieu luu trong docker volume
python index_qdrant.py            # tao collection MOI quy_dinh_hybrid + payload index, ma hoa va nap paragraphs.txt
uvicorn api:app --reload --port 8000
```

`index_qdrant.py` khong bao gio ghi/xoa collection cu `paragraphs`. Chay lai khi du lieu doi: `python index_qdrant.py --recreate` (chi xoa va tao lai collection moi).

CLI: `python query_model.py [--kho KHO] [--date YYYY-MM-DD] "cau hoi"`.

Cau hinh tap trung o `config.py`. Bon bien cu doc tu moi truong (`.env` cua ai-service, `environment` cua docker-compose, `ENV` cua Dockerfile): `GROQ_API_KEY`, `GROQ_QWEN_MODEL`, `QDRANT_URL`, `GROQ_CACHE_DIR`; cong nhom router / session doc cung cach (`GEMINI_API_KEY`, `GEMINI_MODEL`, `ROUTER_*`, `SESSION_*`, bool ghi `true`/`false`). Moi gia tri khac la hang so trong `config.py` (ten trong bang la ten hang so; bool ghi `1`/`0` la `True`/`False`): sua o do roi `docker-compose up -d --build ai-service`.

| Bien | Mac dinh | Y nghia |
|---|---|---|
| `GROQ_QWEN_MODEL` | (bat buoc) | ID model Qwen tren Groq, vi du `qwen/qwen3.8-27b` |
| `GROQ_API_KEY` | (bat buoc) | Groq API key (ten cu `API_KEY` khong con duoc doc) |
| `GROQ_REASONING_EFFORT` | `none` | Tat reasoning cua Qwen; de trong = khong gui tham so |
| `GROQ_RPM` / `GROQ_TPM` | `30` / `8000` | Gioi han toc do phia client |
| `GROQ_MAX_RETRIES` | `5` | So lan thu lai khi 429/timeout (backoff mu, ton trong `Retry-After`) |
| (viet lai cau hoi) | luon chay | Qwen viet lai cau tim kiem; dense + bm25 tim cho ca hai cau, RRF mot lan tren moi danh sach (gop trung theo id) truoc khi rerank. Them mot lan goi Groq moi cau hoi RAG (cau hoi cho buoc nay toi da `QUERY_REWRITE_TIMEOUT_S`, mac dinh 6 s, tinh ca ket noi va thu lai; 429 / mat ket noi duoc thu lai mot lan sau 1 s, timeout thi khong); loi thi tim mot cau nhu cu |
| `GROQ_CACHE_DIR` | `.groq_cache` | Cache dia theo hash (model, prompt), luon bat |
| `WARMUP_WAIT_S` | `900` | BGE-M3 + reranker luon duoc nap ngay khi server khoi dong (thread nen). Request toi luc dang warm-up doi toi da chung nay (giay) roi moi chay router / viet lai / truy van; warm-up xong som thi di ngay |
| `OCR_VL_URL` | `http://localhost:8080` | Container `paddleocr-vl-api` (compose: `http://paddleocr-vl-api:8080`) nhan trang scan, tra Markdown |
| `OCR_VL_TIMEOUT_S` | `1800` | Cho toi da cho MOT lo trang scan |
| `OCR_PAGES_PER_REQUEST` | `8` | So trang scan gui trong mot request sang PaddleOCR-VL |
| `OCR_SCAN_IMAGE_COVER` | `0.5` | Trang PDF co anh phu tu ti le nay tro len (hoac khong co chu hien thi) coi la scan; con lai doc thang lop chu |
| `OCR_IGNORE_LABELS` | `number,header,header_image,footer,footer_image,aside_text` | Nhan khoi layout khong dua vao van ban (so trang, dau / chan trang) |
| `TOP_K_RETRIEVAL` / `TOP_K_RERANK` | `20` / `5` | So ung vien lay tu Qdrant / so doan dua vao LLM |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://localhost:6333` / `quy_dinh_hybrid_v2` | Collection `quy_dinh_hybrid` cu khong con dung: can embed lai tai lieu vao v2 (script chep da xoa, con trong git history) |
| `EMBED_DEVICE` / `RERANK_DEVICE` | tu chon | `cuda:0` hoac `cpu`. Dat `RERANK_DEVICE=cpu` de nhuong ~1.1GB VRAM |
| `RERANK_MODEL` / `RERANK_REVISION` | `jinaai/jina-reranker-v3.5` / commit da ghim | Weights, config, tokenizer lay dung revision nay. Doi revision thi phai duyet va vendor lai `jina_reranker_modeling.py` |
| `RERANK_DTYPE` | `bfloat16` | dtype tren GPU (tren CPU luon `float32`) |
| `RERANK_CPU_ON_OOM` | `0` | CUDA OOM ngay ca khi cham 1 doan/lan thi chuyen reranker sang CPU |
| `EMBED_BATCH_SIZE` / `EMBED_MAX_LENGTH` | `16` / `512` | Tu dong giam batch khi OOM |
| `RERANK_BATCH_SIZE` / `RERANK_MAX_LENGTH` | `64` / `512` | So doan cham chung trong 1 prompt (chia doi khi OOM) / so token toi da moi doan |
| `RERANK_MIN_SCORE` | `0.0` | Diem rerank (cosine, -1..1) toi thieu cua mot doan; `-1` de tat |
| `HNSW_EF` | `128` | efSearch cua nhanh dense |
| `HYBRID_LEGS` | `dense,bm25` | Cac nhanh cua pipeline hybrid, lay tu `dense`, `bm25` (phan cach bang dau phay). Ten khac thi truy van bao `ValueError` |
| `PAYLOAD_FIELD_LOAI_VAN_BAN` | `loai_van_ban` | Ten field payload dung de loc theo `loai_van_ban` |
| `PAYLOAD_FIELD_KHO`, `..._DOI_TUONG`, `..._HIEU_LUC_TU`, `..._HIEU_LUC_DEN` | `kho`, `doi_tuong`, `hieu_luc_tu`, `hieu_luc_den` | Ten field payload dung de index/loc |
| `GUARD_MAX_QUESTION_CHARS` | `1000` | Cau hoi dai hon bi cat (sau khi chuan hoa) |
| `GUARD_CANARY` | (de trong) | De trong = suy ra tu noi dung system prompt, on dinh giua cac lan chay nen cache dia Groq van trung |
| `GUARD_REFUSAL_ANSWER`, `GUARD_REPLY_GREETING` / `_CAPABILITY` / `_META` / `_THANKS` / `_GOODBYE`, `OFF_TOPIC_ANSWER`, `CLARIFY_ANSWER_TOPIC` / `_AUDIENCE` / `_TIME` | xem `config.py` | Cau tra loi co dinh (moi chu hien ra cho nguoi dung deu lay tu day, khong bao gio tu chuoi do router sinh ra) |
| `GEMINI_API_KEY` | (nen co) | Router Gemini (goi thang REST `generateContent` bang httpx) luon chay; thieu khoa thi moi request di thang vao RAG va server canh bao luc khoi dong. Gui o header `x-goog-api-key`; khong bao gio log, khong nam trong image |
| `GEMINI_MODEL` (hang so `ROUTER_MODEL`) | (nen co) | Id model Gemini, vi du `gemini-3.5-flash-lite` (stable; khong dung ban `*-preview`). Tien to `gemini/` kieu cu van duoc chap nhan va bi bo |
| `ROUTER_TIMEOUT_S` / `ROUTER_MAX_RETRIES` | `10` / `1` | Timeout moi lan goi / so lan thu lai (timeout, mat ket noi, 5xx; 429 va 4xx khac khong thu lai). Do thuc te 1-9 s moi lan goi (p95 ~8.8 s); xau nhat truoc khi roi vao RAG la (1 + retry) x timeout = 20 s |
| `ROUTER_RPM` | `15` | Gioi han trong tien trinh; het ngan sach thi request vao thang RAG. Google khong cong bo RPM free tier (xem trong AI Studio) |
| `ROUTER_MIN_CONFIDENCE` / `ROUTER_BLOCK_CONFIDENCE` / `ROUTER_OFFTOPIC_CONFIDENCE` / `ROUTER_CLARIFY_CONFIDENCE` | `0.5` / `0.7` / `0.8` / `0.7` | Quyet dinh duoi nguong thi coi la `rag` |
| `ROUTER_HISTORY_TURNS` | `3` | So luot lich su gui cho router |
| `ROUTER_REASONING_EFFORT` | `low` | Khong dat `none` voi gemini-3.5-flash-lite: `standalone_query` lap vo han |
| `SESSION_STORE` / `SESSION_DB_URI` | `memory` / (rong) | `memory`: SQLite trong RAM (mat khi restart). `sqlite`: file `sqlite+aiosqlite:///...`, can volume |
| `SESSION_TTL_S` / `SESSION_MAX_SESSIONS` | `1800` / `200` | Het han (tu luot luu cuoi) hoac vuot so session thi xoa (LRU) |
| `SESSION_MAX_TURNS` / `SESSION_MAX_HISTORY_TOKENS` | `6` / `1500` | Cua so lich su dua vao prompt: lay cai nho hon |
| `MONGODB_URI` | `mongodb://localhost:27017` | Luu tru hoi thoai lau dai (luon bat), vd `mongodb://user:pass@mongodb:27017/?authSource=admin` (`session_id` la khoa cuoc hoi thoai) |
| `MONGODB_DB` / `MONGODB_TIMEOUT_MS` | `hsu_chatbot` / `2000` | Ten database / thoi gian cho chon server. MongoDB chet thi moi luot hoi dap cham them toi chung nay (cau tra loi van duoc tra) |

Base URL mac dinh: `http://localhost:8000`

## Pipeline: hybrid roi rerank

Luon chay hybrid roi rerank (khong con tham so `mode`; client cu van gui `mode` thi bi bo qua):

1. Qdrant: prefetch cac nhanh trong `HYBRID_LEGS` (`dense` + `bm25`; moi nhanh 20), hop nhat RRF o phia Qdrant, lay 20 ung vien.
2. `jina-reranker-v3.5` cham lai va giu 5 doan (cham theo cau goc, ke ca khi co cau viet lai), bo doan co diem duoi `RERANK_MIN_SCORE`.

Moi nhanh la mot `Prefetch` trong cung mot request toi Qdrant va **cung mang mot bo loc** (`kho`, hieu luc, `khoa`, `loai_van_ban`). Bo loc nam o tung nhanh tim vector nen Qdrant ap no ngay trong luc tim, khong loc sau khi da lay top-k. Nhanh `dense` la vector BGE-M3 (sentence-transformers, GPU); nhanh `bm25` la BM25 tu viet (`app/bm25.py`, CPU; token am tiet + bigram, Qdrant nhan IDF luc truy van). Cau viet lai them hai nhanh nua (cung bo loc) va tat ca cac danh sach vao MOT lan RRF; chunk nam trong nhieu danh sach chi xuat hien mot lan.

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

### POST /ask

Gui cau hoi, nhan cau tra loi cua LLM kem nguon trich dan.

**Request body**
```json
{
  "question": "Thi sinh duoc dang ky toi da bao nhieu nguyen vong?",
  "top_k": 5,
  "kho": "Khoa A",
  "reference_date": "2026-01-31",
  "loai_van_ban": ["thong_bao", "quy_che"],
  "session_id": "6f9619ff-8b86-4011-b42d-00c04fc964ff",
  "reset": false
}
```

| Field | Type | Bat buoc | Mac dinh | Ghi chu |
|---|---|---|---|---|
| `question` | string | co | - | Khong duoc rong |
| `top_k` | int | khong | `5` | So doan van lay lam ngu canh (1-10) |
| `kho` | string | khong | - | Chi tim trong kho nay |
| `reference_date` | date | khong | hom nay | Chi lay van ban con hieu luc tai ngay nay |
| `loai_van_ban` | string[] | khong | - | Chi tim trong cac loai van ban nay: `quy_che`, `quy_dinh`, `thong_bao`, `bieu_mau`, `huong_dan`, `khac` (chunker suy ra tu ten tai lieu). Bo trong hoac `[]` = khong loc; gia tri khong ton tai thi khong co doan nao (`in_scope=false`) |
| `session_id` | string (UUID) | khong | server tao UUID v4 | Khong phai UUID -> `400`. Phai gui lai gia tri server tra ve de cac cau hoi lien tuc dung chung lich su; id chua tung thay (het han, server restart voi `memory`) la session moi |
| `reset` | bool | khong | `false` | Xoa lich su session truoc khi xu ly request nay (khong can route `DELETE` moi) |

Server khong bao gio nhan lich su tu client: moi truong khac ngoai bang tren (ke ca `history`) bi bo qua.

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
| `intent` | string | **Tuy chon, them moi**; client cu bo qua duoc. `rag` (di qua pipeline, ke ca khi `in_scope=false` vi khong co doan phu hop), `greeting`, `thanks`, `goodbye`, `capability` (hoi cong dung / ban la ai), `meta` (hoi ve model, prompt, cau hinh: tra loi co dinh, khong tiet lo), `blocked` (router danh dau tan cong / injection, hoac guard dau ra chan cau tra loi lo prompt / bi mat), `off_topic` (ngoai pham vi quy dinh cua truong; `in_scope=false`), `clarify` (thieu chu de / doi tuong / thoi diem, tra mot cau hoi lai co dinh; `in_scope=true`). Cac intent khac `rag` khong bao gio co `sources` |
| `session_id` | string | UUID cua session (do client gui hoac server tao) |
| `sources[].score` | float | Diem rerank (cosine cua reranker, -1..1) |
| `sources[].rerank_score` | float | Diem cosine (-1..1) cua reranker (bang `score`), co the **am**; chi so sanh duoc giua cac doan cung mot lan hoi |
| `sources[].payload` | object | Toan bo payload cua doan (dung de trich dan) |

Nguong tuong dong: `RERANK_MIN_SCORE` (mac dinh `0.0`, moi doan duoi nguong bi bo, nen LLM co the nhan it hon 5 doan). Nguong `0.0` do lai tren kho hien tai (1297 chunk, 77 van ban): cau hoi ngan nhu "Hoc phi bao nhieu?" bi reranker cham diem thap (doan dung chi 0.01-0.06) nen nguong `0.1` cu cat con 0-1 doan va LLM tra "Cac doan trich duoc cung cap khong chua thong tin..."; cau lac de cao nhat -0.032 nen van bi chan. Cau "gan chu de" nhung khong co dap an van toi LLM va prompt se tu choi.

**Loi**

| Status | Khi nao | Body |
|---|---|---|
| `400` | `question` rong (ke ca chi gom ky tu zero width / dieu khien) | `{ "detail": "question khong duoc de trong" }` |
| `400` | `session_id` khong phai UUID | `{ "detail": "session_id phai la UUID" }` |
| `422` | Tham so khong hop le (vd `top_k` ngoai 1-10) | chi tiet validation |
| `500` | Server thieu bien moi truong (`.env`) | `{ "detail": "Thieu bien moi truong: ..." }` |
| `502` | Qdrant loi, hoac Groq that bai sau khi het so lan thu lai (429/timeout) | `{ "detail": "Loi khi truy van Qdrant/goi LLM: ..." }` |

### POST /ask/stream

Nhu `/ask` (cung request body) nhung tra cau tra loi dang **Server-Sent Events** (`Content-Type: text/event-stream`), tung doan token ngay khi Groq sinh ra. Dung `fetch` + `ReadableStream` (khong dung `EventSource` vi `EventSource` chi ho tro GET).

Moi su kien la mot dong `data: <JSON>` ket thuc bang dong trong, theo thu tu:

```
data: {"event": "sources", "in_scope": true, "sources": [{"source": "quy_che.pdf", "text": "...", "score": 0.6004, "rerank_score": 0.6004, "payload": {...}}], "session_id": "6f9619ff-8b86-4011-b42d-00c04fc964ff"}

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

- Tra loi co dinh (chao hoi, cau bi chan, `off_topic`, `clarify`) van theo dung hop dong tren: `sources` voi danh sach rong (`in_scope=true` cho chao hoi va `clarify`, `false` cho cau bi chan va `off_topic`), dung 1 `delta`, roi `[DONE]`. Su kien `sources` cua chung co them khoa `intent` (`greeting`, `thanks`, `goodbye`, `capability`, `meta`, `blocked`, `off_topic`, `clarify`); su kien `sources` cua cau hoi RAG **khong** co khoa nay.
- Su kien `sources` luon co them khoa `session_id`. Luot hoi dap chi duoc luu sau delta cuoi (truoc `[DONE]`); stream loi (`error`) thi khong luu gi. Khoa session duoc giu den het stream: request thu hai cung session cho request dau xong.
- Truy van Qdrant + rerank chay **truoc** khi gui header, nen loi o buoc nay van tra ma HTTP `400/422/500/502` voi body JSON nhu `/ask`. Sau khi da nhan `200`, loi chi bao bang su kien `error`.
- Stream ket thuc ma khong co `[DONE]` (va khong co `error`) = mat ket noi giua chung, FE nen coi la loi.
- Groq 429/timeout truoc token dau tien van duoc thu lai nhu `/ask` (client cho lau hon, chua nhan `delta` nao). Loi sau khi da gui token thi khong thu lai, gui `error`.
- Cau tra loi day du duoc ghi vao cache dia nhu `/ask`; cau hoi lap lai tra ve trong 1 `delta` duy nhat.
- Khoi `<think>...</think>` (neu bat reasoning) duoc loc ngay trong stream.

## Router Gemini va session

**Router.** Mot lan goi Gemini (REST `generateContent`, structured output) phan loai cau hoi, khong co danh sach tu khoa nao: `rag`, `off_topic`, `clarify`, `greeting`, `thanks`, `goodbye`, `capability`, `meta`, `injection`. Chinh sach (theo thu tu): `injection` >= 0.7 -> tu choi (`intent=blocked`, khong luu); `off_topic` >= 0.8 -> `OFF_TOPIC_ANSWER` (khong luu); `clarify` >= 0.7 -> mot cau hoi lai co dinh theo `missing` (`topic` / `audience` / `time`, co luu), khong hoi hai lan lien tiep va bo qua `audience` khi `khoa` da biet; chao hoi / cam on / tam biet / capability / meta >= 0.5 -> cau co dinh; con lai (ke ca duoi nguong) -> RAG. Voi RAG, `standalone_query` (cau tim kiem doc lap, vd "con hoc ky dau tien thi sao?" -> "Sinh vien duoc dang ky toi da bao nhieu tin chi trong hoc ky dau tien?") thay cau hoi cho encode / Qdrant / rerank; prompt tra loi van mang cau hoi goc. Gemini chi nhan cau hoi da chuan hoa, vai luot lich su cuoi, `khoa` / `kho` va co "luot truoc la clarify" (suy ra tu luot da luu, khong tu request); khong nhan doan trich, system prompt, bi mat hay dinh danh nguoi dung. Router loi (timeout, quota, JSON hong, het `ROUTER_RPM`, thieu khoa / model) thi cau hoi di thang vao RAG, request khong bao gio loi vi router.

Luu y rieng tu: goi Gemini free tier cho phep Google dung noi dung de cai thien san pham; goi tra phi thi khong.

**Session.** Moi `session_id` giu lich su ngan han cua cuoc tro chuyen de hieu cau follow up. Server lay `session_id` tu body (hoac tao moi va tra lai trong response / su kien `sources`); FE can luu va gui lai gia tri nay o cac cau tiep theo (FE hien tai chua lam). `khoa` va `kho` hop le cua cac request truoc duoc nho trong session va dung lam bo loc khi request sau khong gui; gia tri trong request luon de len gia tri da luu; `reference_date` khong bao gio duoc luu. Cac luot `rag`, `clarify` va small talk duoc luu sau khi tra loi xong; `injection`, `off_topic`, cau bi guard dau ra chan va stream loi khong duoc luu. Het han sau `SESSION_TTL_S` (mac dinh 30 phut tu luot cuoi) hoac khi qua `SESSION_MAX_SESSIONS`. Mac dinh lich su nam trong RAM va mat khi restart; `SESSION_STORE=sqlite` luu file va con sau khi restart. Session khong luu vinh vien: het han, bi LRU loai hoac `reset` deu xoa, va chi giu `SESSION_MAX_TURNS` luot gan nhat. Khong mo duoc session store thi request van duoc tra loi, chi khong co lich su.

**Bao mat session.** Neu gateway gui header `X-User-Id` (api-gateway lam viec nay sau khi xac thuc JWT) thi khoa luu tru la `user:session_id`, nen biet UUID cua nguoi khac cung khong doc duoc lich su. Neu khong co header, `session_id` la bi mat mang theo nguoi dung (bearer): ai biet UUID thi doc/ghi duoc session do. Cong `8001` cua ai-service mo ra may host, goi thang vao do thi client tu dat duoc `X-User-Id`; chi tin header khi ai-service khong public.

## Luu tru hoi thoai (MongoDB)

Luon bat (`session_id` la khoa cuoc hoi thoai). Khac session (lich su ngan han cho prompt, het han sau `SESSION_TTL_S`): day la ban ghi lau dai cho nguoi dung xem lai, khong bao gio vao prompt. Moi luot hoi dap da hoan tat duoc them vao cuoc hoi thoai `session_id` (ke ca luot bi chan / lac de, vi nguoi dung da thay chung; stream loi thi khong luu). `reset=true` chi xoa lich su prompt, khong xoa ban luu tru. Loi ghi chi ghi log, khong lam hong cau tra loi.

Khoa luu tru giong session: co header `X-User-Id` thi la `user:session_id` (nguoi khac biet UUID cung khong doc duoc). Khong co header (goi thang, khong qua gateway) thi cuoc hoi thoai chi truy cap duoc bang `session_id` va **khong** liet ke duoc. Qua gateway: `/api/ai/conversations/...`.

| Method | Path | Tra ve |
|---|---|---|
| `GET` | `/conversations?limit=50` | `[{session_id, title, created_at, updated_at}]` cua nguoi dung, moi nhat truoc (`limit` 1-200). Thieu `X-User-Id` -> `401` |
| `GET` | `/conversations/{session_id}` | `{session_id, title, created_at, updated_at, messages: [{role, content, intent?, ts}]}`; `intent` chi co o tin nhan `assistant`. Khong co -> `404` |
| `DELETE` | `/conversations/{session_id}` | Xoa ban luu tru va lich su ngan han cua session. Khong co -> `404` |

Loi chung: `400` `session_id` khong phai UUID, `502` khong truy cap duoc MongoDB. `title` la cau hoi dau tien (toi da 80 ky tu).

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

- BGE-M3 (~1.1GB VRAM) va reranker (~1.1GB) nap ngay khi server khoi dong (~15s tren RTX 3050; lan dau con tai model tu Hugging Face) roi nam trong bo nho cho cac request sau. Reranker cung duoc nap ngay khi khoi dong.
- Windows: goi `http://localhost:8000` tu Python/curl co the cham them ~2s moi ket noi (thu IPv6 `::1` truoc, uvicorn chi nghe IPv4). Dung `http://127.0.0.1:8000` de tranh. Truong hop xau nhat (20 doan x 512 token) dinh khoang 3.3GB tren GPU 4GB; sat gioi han thi dat `RERANK_DEVICE=cpu` hoac `RERANK_CPU_ON_OOM=1`.
- `jina-reranker-v3.5` dung giay phep **CC BY-NC 4.0** (chi phi thuong mai). Model duoc nap bang transformers qua `jina_reranker_modeling.py`: ban sao modeling.py cua model o revision da ghim (xem header file de biet nguon, giay phep, va dong duy nhat da sua), nen khong can `trust_remote_code` va khong chay code tai ve luc chay.
- Groq bi gioi han client-side 30 RPM / 8K TPM, nen request co the cho; cau hoi lap lai tra ngay tu cache dia.
- CORS dang mo cho moi origin (`*`) de FE goi truc tiep khi dev; can gioi han domain cu the khi deploy.
- Chay test: `pytest` (can model that + Qdrant dang chay); chi unit test: `pytest -m "not integration"`.
