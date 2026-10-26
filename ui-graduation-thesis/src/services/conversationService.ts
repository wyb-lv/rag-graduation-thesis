import type { SourceCitation } from '@/types'
import rawData from '@/data/conversations.json'
import { authFetch, getKhoa } from '@/services/authService'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:9090'

interface RawApiSource {
  source: string
  text: string
  score: number
}

export interface StreamCallbacks {
  onSources?: (sources: SourceCitation[], inScope: boolean) => void
  onDelta?: (text: string) => void
  onSession?: (sessionId: string) => void
}

// sessionId: server giữ lịch sử theo id này; gửi lại id của hội thoại để câu hỏi sau hiểu được câu trước
export async function askQuestionStream(
  question: string,
  callbacks: StreamCallbacks,
  top_k?: number,
  sessionId?: string
): Promise<void> {
  const { onSources, onDelta, onSession } = callbacks
  try {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    const res = await authFetch(`${API_BASE_URL}/api/ai/ask/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        question,
        khoa: getKhoa(),
        ...(top_k !== undefined ? { top_k } : {}),
        ...(sessionId ? { session_id: sessionId } : {}),
      }),
    })
    if (res.status === 401) throw new Error('Phiên đăng nhập đã hết hạn, vui lòng đăng nhập lại.')
    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}))
      throw new Error(errorData.detail || errorData.message || `Lỗi máy chủ (${res.status})`)
    }
    if (!res.body) {
      throw new Error('Trình duyệt không hỗ trợ ReadableStream.')
    }
    const reader = res.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()

      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmedLine = line.trim()

        if (trimmedLine.startsWith('data: ')) {
          const raw = trimmedLine.slice(6).trim()

          if (raw === '[DONE]') return
          try {
            const payload = JSON.parse(raw)

            if (payload.event === 'sources' && payload.session_id) onSession?.(payload.session_id)
            if (payload.event === 'sources' && onSources) {
              const mappedSources: SourceCitation[] = (payload.sources || []).map(
                (s: RawApiSource, index: number) => ({
                  id: `src-${Date.now()}-${index}`,
                  type: 'document',
                  title: s.source,
                  description: `Độ tương đồng: ${(s.score * 100).toFixed(1)}%`,
                  text: s.text,
                  score: s.score,
                  fileType: s.source.split('.').pop()?.toUpperCase() || 'DOC',
                })
              )
              onSources(mappedSources, payload.in_scope ?? true)
            } else if (payload.event === 'delta' && onDelta) {
              onDelta(payload.text)
            }
          } catch (e) {
            console.warn('[SSE] Lỗi parse JSON:', raw, e)
          }
        }
      }
    }
  } catch (err: unknown) {
    // Chỉ lỗi mạng (backend chưa chạy) mới dùng bản mô phỏng; lỗi HTTP/401 phải báo cho người dùng
    if (!(err instanceof TypeError)) throw err
    console.warn(`[ChatHSU SSE] Không thể kết nối tới ${API_BASE_URL}/api/ai/ask/stream:`, err)
    // Fallback mô phỏng nếu chưa bật API Gateway
    if (onSources) {
      onSources(rawData.sources as SourceCitation[], true)
    }
    if (onDelta) {
      const mockText = `Mình đã nhận được câu hỏi: "${question}".\n\n(Lưu ý: Chưa thể kết nối tới API Gateway tại ${API_BASE_URL}. Hãy chạy \`docker compose up -d\` rồi tải lại trang).`
      onDelta(mockText)
    }
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