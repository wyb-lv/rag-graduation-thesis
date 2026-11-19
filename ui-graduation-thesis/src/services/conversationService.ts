import type { Conversation, Message, SourceCitation } from '@/types'
import rawData from '@/data/conversations.json'
import { authFetch, getKhoa } from '@/services/authService'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:9090').replace(/\/+$/, '')

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
  | { event: 'sources'; in_scope: boolean; sources: RawApiSource[]; session_id?: string }
  | { event: 'delta'; text: string }
  | { event: 'error'; detail: string }

export interface StreamCallbacks {
  onTranscription?: (text: string) => void
  onSources?: (sources: SourceCitation[], inScope: boolean) => void
  onDelta?: (text: string) => void
  onSession?: (sessionId: string) => void
}

export interface AskOptions {
  top_k?: number
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

// Ném lỗi có thông điệp cho người dùng khi response không ok (401 = hết phiên, còn lại lấy `detail` của server)
async function assertOk(res: Response, fallback: string): Promise<void> {
  if (res.ok) return
  if (res.status === 401) throw new Error('Phiên đăng nhập đã hết hạn, vui lòng đăng nhập lại.')
  const errorData = await res.json().catch(() => ({}))
  const detail = typeof errorData.detail === 'string' ? errorData.detail : errorData.message
  throw new Error(detail || `${fallback} (${res.status})`)
}

// Khớp với ConversationSummary / Conversation trong ai-service/app/api.py (GET /conversations, GET /conversations/{session_id})
interface RawConversationSummary {
  session_id: string
  title: string
  created_at: string
}

interface RawConversation extends RawConversationSummary {
  messages: { role: 'user' | 'assistant'; content: string; ts: string }[]
}

/** Danh sách hội thoại của người dùng đang đăng nhập (mới nhất trước), `messages` rỗng cho đến khi mở bằng getConversationMessages. */
export async function listConversations(): Promise<Conversation[]> {
  const res = await authFetch(buildAiUrl('/conversations'))
  await assertOk(res, 'Không tải được lịch sử chat')
  const rows: RawConversationSummary[] = await res.json()
  return rows.map((r) => ({ id: r.session_id, title: r.title, messages: [], createdAt: r.created_at }))
}

/** Toàn bộ tin nhắn của một hội thoại. Server không lưu nguồn trích dẫn nên `sources` không có. */
export async function getConversationMessages(sessionId: string): Promise<Message[]> {
  const res = await authFetch(buildAiUrl(`/conversations/${encodeURIComponent(sessionId)}`))
  await assertOk(res, 'Không tải được cuộc hội thoại')
  const conv: RawConversation = await res.json()
  return conv.messages.map((m, i) => ({ id: `${sessionId}-${i}`, role: m.role, content: m.content, timestamp: m.ts }))
}

async function consumeSseResponse(res: Response, callbacks: StreamCallbacks): Promise<void> {
  const { onTranscription, onSources, onDelta, onSession } = callbacks
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
    else if (payload.event === 'sources') {
      if (payload.session_id) onSession?.(payload.session_id)
      onSources?.(mapSources(payload.sources || []), payload.in_scope ?? true)
    } else if (payload.event === 'delta') onDelta?.(payload.text)
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
 * sessionId: server giữ lịch sử theo id này; gửi lại id của hội thoại để câu hỏi sau hiểu được câu trước.
 */
export async function askQuestionStream(
  question: string,
  callbacks: StreamCallbacks,
  top_k?: number,
  sessionId?: string
): Promise<void> {
  const { onSources, onDelta } = callbacks
  const url = buildAiUrl('/ask/stream')

  let res: Response
  try {
    res = await authFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({
        question,
        khoa: getKhoa(),
        ...(top_k !== undefined ? { top_k } : {}),
        ...(sessionId ? { session_id: sessionId } : {}),
      }),
    })
  } catch (err: unknown) {
    // Chỉ lỗi mạng (backend chưa chạy) mới dùng bản mô phỏng; lỗi khác phải báo cho người dùng
    if (!(err instanceof TypeError)) throw err
    console.warn(`[ChatHSU SSE] Không thể kết nối tới ${url}:`, err)
    onSources?.(rawData.sources as SourceCitation[], true)
    onDelta?.(
      `Mình đã nhận được câu hỏi: "${question}".\n\n(Lưu ý: Chưa thể kết nối tới Backend tại ${url}. Hãy chạy \`docker compose up -d\` rồi tải lại trang).`
    )
    return
  }

  await assertOk(res, 'Lỗi máy chủ')
  await consumeSseResponse(res, callbacks)
}

export interface AudioStreamCallbacks {
  onPlayStart?: () => void
  onEnded: () => void
  onBlobCollected?: (blob: Blob) => void
}

async function playAudioFromBlob(
  blob: Blob,
  callbacks: AudioStreamCallbacks,
  signal?: AbortSignal
): Promise<HTMLAudioElement> {
  if (signal?.aborted) {
    throw new DOMException('Aborted', 'AbortError')
  }
  if (blob.size === 0) {
    const emptyAudio = new Audio()
    callbacks.onEnded()
    return emptyAudio
  }

  callbacks.onBlobCollected?.(blob)
  const objectUrl = URL.createObjectURL(blob)
  const audio = new Audio(objectUrl)
  let settled = false

  const cleanup = (notifyEnded: boolean) => {
    if (settled) return
    settled = true
    signal?.removeEventListener('abort', onAbort)
    audio.onended = null
    audio.onerror = null
    URL.revokeObjectURL(objectUrl)
    if (notifyEnded && !signal?.aborted) {
      callbacks.onEnded()
    }
  }

  const onAbort = () => {
    audio.pause()
    audio.src = ''
    cleanup(false)
  }

  signal?.addEventListener('abort', onAbort, { once: true })
  audio.onended = () => cleanup(true)
  audio.onerror = () => cleanup(true)

  try {
    await audio.play()
    if (!signal?.aborted) {
      callbacks.onPlayStart?.()
    }
  } catch {
    cleanup(true)
  }
  return audio
}

function waitForSourceBufferUpdate(sb: SourceBuffer, signal?: AbortSignal): Promise<void> {
  if (!sb.updating) return Promise.resolve()
  return new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      sb.removeEventListener('updateend', onEnd)
      sb.removeEventListener('error', onErr)
      sb.removeEventListener('abort', onEnd)
      signal?.removeEventListener('abort', onEnd)
    }
    const onEnd = () => {
      cleanup()
      resolve()
    }
    const onErr = () => {
      cleanup()
      reject(new Error('SourceBuffer update failed'))
    }
    sb.addEventListener('updateend', onEnd, { once: true })
    sb.addEventListener('error', onErr, { once: true })
    sb.addEventListener('abort', onEnd, { once: true })
    signal?.addEventListener('abort', onEnd, { once: true })
  })
}

/**
 * Phát luồng âm thanh MP3 từ `Response` bằng MediaSource Extensions (MSE) ngay khi chunk đầu tiên về tới,
 * tự động fallback sang `res.blob()` nếu trình duyệt không hỗ trợ MSE hoặc `addSourceBuffer` lỗi.
 */
export async function playAudioResponseStream(
  res: Response,
  callbacks: AudioStreamCallbacks,
  signal?: AbortSignal
): Promise<HTMLAudioElement> {
  if (signal?.aborted) {
    throw new DOMException('Aborted', 'AbortError')
  }

  const canUseMse =
    typeof MediaSource !== 'undefined' &&
    MediaSource.isTypeSupported('audio/mpeg') &&
    Boolean(res.body)

  if (!canUseMse || !res.body) {
    const blob = await res.blob()
    return playAudioFromBlob(blob, callbacks, signal)
  }

  const mediaSource = new MediaSource()
  const objectUrl = URL.createObjectURL(mediaSource)
  const audio = new Audio()

  let sourceBuffer: SourceBuffer
  try {
    sourceBuffer = await new Promise<SourceBuffer>((resolve, reject) => {
      const onAbort = () => reject(new DOMException('Aborted', 'AbortError'))
      if (signal?.aborted) {
        onAbort()
        return
      }
      signal?.addEventListener('abort', onAbort, { once: true })
      mediaSource.addEventListener(
        'sourceopen',
        () => {
          signal?.removeEventListener('abort', onAbort)
          try {
            resolve(mediaSource.addSourceBuffer('audio/mpeg'))
          } catch (err) {
            reject(err)
          }
        },
        { once: true }
      )
      audio.src = objectUrl
    })
  } catch (err) {
    URL.revokeObjectURL(objectUrl)
    audio.pause()
    audio.src = ''
    if (err instanceof DOMException && err.name === 'AbortError') {
      throw err
    }
    const blob = await res.blob()
    return playAudioFromBlob(blob, callbacks, signal)
  }

  const reader = res.body.getReader()
  const collectedChunks: Uint8Array<ArrayBuffer>[] = []
  let settled = false
  let aborted = false
  let hasStartedPlayback = false

  const finish = (notifyEnded: boolean) => {
    if (settled) return
    settled = true
    signal?.removeEventListener('abort', onAbort)
    audio.onended = null
    audio.onerror = null
    URL.revokeObjectURL(objectUrl)
    if (notifyEnded && !aborted && !signal?.aborted) {
      callbacks.onEnded()
    }
  }

  const onAbort = () => {
    aborted = true
    reader.cancel().catch(() => {})
    audio.pause()
    audio.src = ''
    finish(false)
  }

  signal?.addEventListener('abort', onAbort, { once: true })
  audio.onended = () => finish(true)
  audio.onerror = () => finish(true)

  const pump = async () => {
    try {
      while (true) {
        if (aborted || settled || signal?.aborted) break
        const { done, value } = await reader.read()
        if (aborted || settled || signal?.aborted) break

        if (done) {
          await waitForSourceBufferUpdate(sourceBuffer, signal)
          if (!aborted && !signal?.aborted && mediaSource.readyState === 'open') {
            mediaSource.endOfStream()
          }
          if (!aborted && !signal?.aborted && callbacks.onBlobCollected && collectedChunks.length > 0) {
            callbacks.onBlobCollected(new Blob(collectedChunks, { type: 'audio/mpeg' }))
          }
          if (!hasStartedPlayback) {
            finish(true)
          }
          break
        }

        if (!value || value.byteLength === 0) continue

        const safeChunk = new Uint8Array(value)
        if (callbacks.onBlobCollected) {
          collectedChunks.push(safeChunk)
        }

        await waitForSourceBufferUpdate(sourceBuffer, signal)
        if (aborted || signal?.aborted || mediaSource.readyState !== 'open') break

        sourceBuffer.appendBuffer(safeChunk)
        await waitForSourceBufferUpdate(sourceBuffer, signal)

        if (!hasStartedPlayback) {
          hasStartedPlayback = true
          if (!aborted && !signal?.aborted) {
            audio
              .play()
              .then(() => {
                if (!aborted && !signal?.aborted) {
                  callbacks.onPlayStart?.()
                }
              })
              .catch(() => {
                finish(true)
              })
          }
        }
      }
    } catch (err) {
      if (!aborted && !signal?.aborted) {
        console.warn('[AudioStream] Lỗi stream MSE:', err)
        if (hasStartedPlayback && mediaSource.readyState === 'open') {
          try {
            mediaSource.endOfStream()
          } catch {
            finish(true)
          }
        } else {
          finish(true)
        }
      }
    } finally {
      reader.cancel().catch(() => {})
    }
  }

  void pump()
  return audio
}

/**
 * Luồng 1: Upload file âm thanh -> Gọi thẳng POST /ask/voice/stream
 * Server nhận file âm thanh, tách chữ (STT) và gọi trực tiếp tới luồng RAG stream rồi trả về SSE.
 */
export async function askVoiceStream(
  audioFile: File | Blob,
  callbacks: StreamCallbacks,
  options: AskOptions = {},
  sessionId?: string
): Promise<void> {
  const { signal, top_k, kho, reference_date } = options

  const formData = new FormData()
  const filename = audioFile instanceof File ? audioFile.name : 'voice_message.wav'
  formData.append('file', audioFile, filename)
  const khoa = getKhoa()
  if (khoa !== undefined) formData.append('khoa', String(khoa))
  if (sessionId) formData.append('session_id', sessionId)
  if (top_k !== undefined) formData.append('top_k', String(top_k))
  if (kho) formData.append('kho', kho)
  if (reference_date) formData.append('reference_date', reference_date)

  const res = await authFetch(buildAiUrl('/ask/voice/stream'), {
    method: 'POST',
    headers: { Accept: 'text/event-stream' },
    body: formData,
    signal,
  })

  await assertOk(res, 'Lỗi xử lý giọng nói')
  await consumeSseResponse(res, callbacks)
}

/**
 * Luồng 2: Gửi đoạn ghi âm từ nút Mic lên POST /stt để lấy văn bản hiển thị lên ô Input.
 */
export async function transcribeAudio(audio: Blob | File, filename = 'recording.webm'): Promise<SttResponse> {
  const formData = new FormData()
  const name = audio instanceof File ? audio.name : filename
  formData.append('file', audio, name)

  const res = await authFetch(buildAiUrl('/stt'), {
    method: 'POST',
    body: formData,
  })

  await assertOk(res, 'Lỗi nhận diện giọng nói')
  return res.json()
}

/**
 * Luồng 3: Gọi POST /tts và trả về `Response` để caller có thể phát trực tiếp theo luồng (hoặc `null` nếu 204).
 */
export async function synthesizeSpeechResponse(text: string, signal?: AbortSignal): Promise<Response | null> {
  const res = await authFetch(buildAiUrl('/tts'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal,
  })

  if (res.status === 204) return null
  await assertOk(res, 'Lỗi tổng hợp giọng nói')
  return res
}

/**
 * Luồng 3 (tương thích ngược): Đọc nội dung câu trả lời của AI qua POST /tts (trả về MP3 Blob).
 */
export async function synthesizeSpeech(text: string, signal?: AbortSignal): Promise<Blob> {
  const res = await synthesizeSpeechResponse(text, signal)
  if (!res) throw new Error('Không có nội dung văn bản để đọc.')
  return res.blob()
}

/**
 * Bắt đầu cuộc gọi: Nhận luồng âm thanh câu chào mở đầu và `X-Session-Id` từ Backend (POST /call/greeting).
 */
export async function fetchCallGreetingAudio(
  signal?: AbortSignal
): Promise<{ response: Response; sessionId: string | null }> {
  const res = await authFetch(buildAiUrl('/call/greeting'), {
    method: 'POST',
    signal,
  })
  await assertOk(res, 'Không thể khởi tạo cuộc gọi')
  return {
    response: res,
    sessionId: res.headers.get('X-Session-Id'),
  }
}

/**
 * Gửi gói ghi âm lượt nói của người dùng xuống Backend (POST /call/turn)
 * và nhận luồng `Response` âm thanh MP3 câu trả lời của AI (hoặc `null` nếu server trả 204 do khoảng lặng).
 */
export async function sendCallTurnAudio(
  audioBlob: Blob,
  filename = 'turn.webm',
  signal?: AbortSignal,
  sessionId?: string | null
): Promise<{ response: Response | null; action: string; sessionId: string | null }> {
  const formData = new FormData()
  formData.append('file', audioBlob, filename)
  const khoa = getKhoa()
  if (khoa !== undefined) {
    formData.append('khoa', String(khoa))
  }
  if (sessionId) {
    formData.append('session_id', sessionId)
  }

  const res = await authFetch(buildAiUrl('/call/turn'), {
    method: 'POST',
    body: formData,
    signal,
  })

  const action = res.headers.get('X-Call-Action') || 'listen'
  const nextSessionId = res.headers.get('X-Session-Id') ?? sessionId ?? null
  if (res.status === 204) {
    return { response: null, action, sessionId: nextSessionId }
  }
  await assertOk(res, 'Lỗi xử lý lượt hội thoại')
  return { response: res, action, sessionId: nextSessionId }
}

