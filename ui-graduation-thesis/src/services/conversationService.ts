import type { SourceCitation } from '@/types'
import rawData from '@/data/conversations.json'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

// Khớp với Source trong ai-service/api.py
interface RawApiSource {
  source: string
  text: string
  score: number // điểm rerank nếu có rerank, không thì cosine (dense) / RRF (hybrid)
  rerank_score: number | null
  payload: Record<string, unknown>
}

// Các sự kiện `data:` của POST /ask/stream (xem ai-service/API.md)
type StreamEvent =
  | { event: 'sources'; in_scope: boolean; sources: RawApiSource[] }
  | { event: 'delta'; text: string }
  | { event: 'error'; detail: string }

export interface StreamCallbacks {
  onSources?: (sources: SourceCitation[], inScope: boolean) => void
  onDelta?: (text: string) => void
}

export interface AskOptions {
  top_k?: number
  mode?: 'dense' | 'hybrid' | 'hybrid_rerank'
  kho?: string
  reference_date?: string // YYYY-MM-DD
  signal?: AbortSignal
}

function mapSources(sources: RawApiSource[]): SourceCitation[] {
  return sources.map((s, index) => ({
    id: `src-${Date.now()}-${index}`,
    type: 'document',
    title: s.source,
    description: `Độ tương đồng: ${(s.score * 100).toFixed(1)}%`,
    text: s.text,
    score: s.score,
    fileType: s.source.split('.').pop()?.toUpperCase() || 'DOC',
  }))
}

/**
 * Gọi POST /ask/stream và đọc SSE: `sources` một lần, `delta` nhiều lần, kết thúc bằng `[DONE]`.
 * Không kết nối được backend -> trả lời mô phỏng. Server báo lỗi (HTTP 4xx/5xx, sự kiện `error`,
 * stream đứt trước [DONE]) -> throw để UI hiển thị lỗi.
 */
export async function askQuestionStream(
  question: string,
  callbacks: StreamCallbacks,
  options: AskOptions = {}
): Promise<void> {
  const { onSources, onDelta } = callbacks
  const { signal, ...params } = options

  let res: Response
  try {
    res = await fetch(`${API_BASE_URL}/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ question, ...params }),
      signal,
    })
  } catch (err: unknown) {
    if (signal?.aborted) throw err
    console.warn(`[ChatHSU SSE] Không thể kết nối tới ${API_BASE_URL}/ask/stream:`, err)
    // Fallback mô phỏng nếu chưa bật FastAPI Backend
    onSources?.(rawData.sources as SourceCitation[], true)
    onDelta?.(
      `Mình đã nhận được câu hỏi: "${question}".\n\n(Lưu ý: Chưa thể kết nối tới FastAPI Backend tại ${API_BASE_URL}. Hãy chạy lệnh: \`uvicorn api:app --reload --port 8000\` để nhận câu trả lời dạng SSE stream từ mô hình RAG).`
    )
    return
  }

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const detail = typeof errorData.detail === 'string' ? errorData.detail : undefined
    throw new Error(detail || `Lỗi máy chủ (${res.status})`)
  }
  if (!res.body) {
    throw new Error('Trình duyệt không hỗ trợ ReadableStream.')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  // Trả về true khi gặp [DONE]
  const handleEvent = (block: string): boolean => {
    const data = block
      .split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).replace(/^ /, ''))
      .join('\n')
    if (!data) return false
    if (data === '[DONE]') return true

    let payload: StreamEvent
    try {
      payload = JSON.parse(data)
    } catch (e) {
      console.warn('[SSE] Lỗi parse JSON:', data, e)
      return false
    }
    if (payload.event === 'sources') onSources?.(mapSources(payload.sources || []), payload.in_scope)
    else if (payload.event === 'delta') onDelta?.(payload.text)
    else if (payload.event === 'error') throw new Error(payload.detail || 'Lỗi khi sinh câu trả lời')
    return false
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true }).replace(/\r\n?/g, '\n')
      const blocks = buffer.split('\n\n')
      buffer = blocks.pop() ?? ''

      for (const block of blocks) {
        if (handleEvent(block)) return
      }
    }
    if (buffer && handleEvent(buffer)) return
    throw new Error('Kết nối tới máy chủ bị ngắt trước khi trả lời xong.')
  } finally {
    reader.cancel().catch(() => {})
  }
}

// export interface AskApiResponse {
//   answer: string
//   sources: SourceCitation[]
// }

// interface RawAskResponse {
//   answer: string
//   sources: Array<{ source: string; text: string; score: number }>
// }

// export async function askQuestion(question: string, top_k?: number): Promise<AskApiResponse> {
//   try {
//     const res = await fetch(`${API_BASE_URL}/ask`, {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify({ question, ...(top_k !== undefined ? { top_k } : {}) }),
//     })
//
//     if (!res.ok) throw new Error(`Lỗi máy chủ (${res.status})`)
//
//     const json: RawAskResponse = await res.json()
//
//     return {
//       answer: json.answer,
//       sources: (json.sources || []).map((s, index) => ({
//         id: `src-${Date.now()}-${index}`,
//         type: 'document',
//         title: s.source,
//         description: `Độ tương đồng: ${(s.score * 100).toFixed(1)}%`,
//         text: s.text,
//         score: s.score,
//         fileType: s.source.split('.').pop()?.toUpperCase() || 'DOC',
//       })),
//     }
//   } catch (err: unknown) {
//     console.warn(`[ChatHSU] Fallback mô phỏng (${API_BASE_URL}):`, err)
//     return {
//       answer: `Mình đã nhận được câu hỏi: "${question}".\n\n(Lưu ý: Chưa thể kết nối tới FastAPI Backend tại ${API_BASE_URL}. Hãy chạy: \`uvicorn api:app --reload --port 8000\` để nhận phản hồi từ mô hình RAG).`,
//       sources: rawData.sources as SourceCitation[],
//     }
//   }
// }