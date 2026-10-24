import type { SourceCitation } from '@/types'
import rawData from '@/data/conversations.json'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '')

/**
 * Chuẩn hóa URL gọi tới AI Service:
 * - Nếu gọi trực tiếp vào ai-service (port 8000) hoặc đã có sẵn `/api/ai` -> nối thẳng path (`/ask/stream`, `/stt`, `/tts`)
 * - Nếu gọi qua API Gateway (port 9090) -> tự động gắn tiền tố `/api/ai` (`/api/ai/ask/stream`, `/api/ai/stt`, `/api/ai/tts`)
 */
export function buildAiUrl(path: string): string {
  const cleanPath = path.startsWith('/') ? path : `/${path}`
  if (API_BASE_URL.endsWith('/api/ai') || API_BASE_URL.includes(':8000')) {
    return `${API_BASE_URL}${cleanPath}`
  }
  return `${API_BASE_URL}/api/ai${cleanPath}`
}

// Khớp với Source trong ai-service/app/api.py
interface RawApiSource {
  source: string
  text: string
  score: number // điểm rerank nếu có rerank, không thì cosine (dense) / RRF (hybrid)
  rerank_score: number | null
  payload: Record<string, unknown>
}

// Các sự kiện `data:` của POST /ask/stream và POST /ask/voice/stream
type StreamEvent =
  | { event: 'transcription'; text: string }
  | { event: 'sources'; in_scope: boolean; sources: RawApiSource[] }
  | { event: 'delta'; text: string }
  | { event: 'error'; detail: string }

export interface StreamCallbacks {
  onTranscription?: (text: string) => void
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

export interface SttResponse {
  is_speech: boolean
  text: string
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

async function consumeSseResponse(res: Response, callbacks: StreamCallbacks): Promise<void> {
  const { onTranscription, onSources, onDelta } = callbacks
  if (!res.body) {
    throw new Error('Trình duyệt không hỗ trợ ReadableStream.')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

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
    if (payload.event === 'transcription') onTranscription?.(payload.text)
    else if (payload.event === 'sources') onSources?.(mapSources(payload.sources || []), payload.in_scope)
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

/**
 * Gọi POST /ask/stream và đọc SSE: `sources` một lần, `delta` nhiều lần, kết thúc bằng `[DONE]`.
 */
export async function askQuestionStream(
  question: string,
  callbacks: StreamCallbacks,
  options: AskOptions = {}
): Promise<void> {
  const { onSources, onDelta } = callbacks
  const { signal, ...params } = options
  const url = buildAiUrl('/ask/stream')

  let res: Response
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ question, ...params }),
      signal,
    })
  } catch (err: unknown) {
    if (signal?.aborted) throw err
    console.warn(`[ChatHSU SSE] Không thể kết nối tới ${url}:`, err)
    onSources?.(rawData.sources as SourceCitation[], true)
    onDelta?.(
      `Mình đã nhận được câu hỏi: "${question}".\n\n(Lưu ý: Chưa thể kết nối tới Backend tại ${url}. Hãy kiểm tra dịch vụ ai-service / api-gateway).`
    )
    return
  }

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const detail = typeof errorData.detail === 'string' ? errorData.detail : undefined
    throw new Error(detail || `Lỗi máy chủ (${res.status})`)
  }

  await consumeSseResponse(res, callbacks)
}

/**
 * Luồng 1: Upload file âm thanh -> Gọi thẳng POST /ask/voice/stream
 * Server nhận file âm thanh, tách chữ (STT) và gọi trực tiếp tới luồng RAG stream rồi trả về SSE.
 */
export async function askVoiceStream(
  audioFile: File | Blob,
  callbacks: StreamCallbacks,
  options: AskOptions = {}
): Promise<void> {
  const { signal, top_k, mode, kho, reference_date } = options
  const url = buildAiUrl('/ask/voice/stream')

  const formData = new FormData()
  const filename = audioFile instanceof File ? audioFile.name : 'voice_message.wav'
  formData.append('file', audioFile, filename)
  if (top_k !== undefined) formData.append('top_k', String(top_k))
  if (mode) formData.append('mode', mode)
  if (kho) formData.append('kho', kho)
  if (reference_date) formData.append('reference_date', reference_date)

  const res = await fetch(url, {
    method: 'POST',
    headers: { Accept: 'text/event-stream' },
    body: formData,
    signal,
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const detail = typeof errorData.detail === 'string' ? errorData.detail : undefined
    throw new Error(detail || `Lỗi xử lý giọng nói (${res.status})`)
  }

  await consumeSseResponse(res, callbacks)
}

/**
 * Luồng 2: Gửi đoạn ghi âm từ nút Mic lên POST /stt để lấy văn bản hiển thị lên ô Input.
 */
export async function transcribeAudio(audio: Blob | File, filename = 'recording.webm'): Promise<SttResponse> {
  const url = buildAiUrl('/stt')
  const formData = new FormData()
  const name = audio instanceof File ? audio.name : filename
  formData.append('file', audio, name)

  const res = await fetch(url, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const detail = typeof errorData.detail === 'string' ? errorData.detail : undefined
    throw new Error(detail || `Lỗi nhận diện giọng nói (${res.status})`)
  }

  return res.json()
}

/**
 * Luồng 3: Đọc nội dung câu trả lời của AI qua POST /tts (trả về MP3 Blob).
 */
export async function synthesizeSpeech(text: string, signal?: AbortSignal): Promise<Blob> {
  const url = buildAiUrl('/tts')
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal,
  })

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}))
    const detail = typeof errorData.detail === 'string' ? errorData.detail : undefined
    throw new Error(detail || `Lỗi tổng hợp giọng nói (${res.status})`)
  }

  return res.blob()
}
