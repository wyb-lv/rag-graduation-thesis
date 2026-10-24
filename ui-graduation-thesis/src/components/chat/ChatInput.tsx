import { useState, useRef, useEffect, type KeyboardEvent, type ChangeEvent } from 'react'
import { SendButton } from './SendButton'
import { transcribeAudio } from '@/services/conversationService'

interface ChatInputProps {
  onSend: (message: string) => void
  onSendVoice?: (file: File, durationSec: number) => void
  position: 'center' | 'bottom'
  disabled?: boolean
}

function formatSec(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const total = Math.round(seconds)
  const mins = Math.floor(total / 60)
  const secs = total % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function getAudioDuration(file: File): Promise<number> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file)
    const audio = new Audio()
    const cleanup = () => URL.revokeObjectURL(url)
    audio.preload = 'metadata'
    audio.onloadedmetadata = () => {
      const dur = Number.isFinite(audio.duration) ? audio.duration : 0
      cleanup()
      resolve(dur)
    }
    audio.onerror = () => {
      cleanup()
      resolve(0)
    }
    audio.src = url
  })
}

export function ChatInput({ onSend, onSendVoice, position, disabled = false }: ChatInputProps) {
  const [value, setValue] = useState('')
  const [attachedAudio, setAttachedAudio] = useState<{ file: File; duration: number } | null>(null)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingSec, setRecordingSec] = useState(0)
  const [isTranscribing, setIsTranscribing] = useState(false)
  const [errorBanner, setErrorBanner] = useState<string | null>(null)

  const inputRef = useRef<HTMLInputElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const timerRef = useRef<number | null>(null)

  const isBusy = disabled || isTranscribing

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
    }
  }, [])

  // Luồng 2: Sau khi bấm kết thúc ghi âm -> gửi về /stt rồi điền vào ô Input
  const handleMicBlobToInput = async (blob: Blob, filename: string) => {
    setIsTranscribing(true)
    setErrorBanner(null)
    try {
      const res = await transcribeAudio(blob, filename)
      if (res.is_speech && res.text.trim()) {
        setValue((prev) => (prev.trim() ? `${prev.trim()} ${res.text.trim()}` : res.text.trim()))
        setTimeout(() => inputRef.current?.focus(), 40)
      } else {
        setErrorBanner('Không nhận diện được tiếng nói rõ ràng. Vui lòng thử nói lại.')
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Lỗi chuyển giọng nói thành văn bản'
      setErrorBanner(msg)
    } finally {
      setIsTranscribing(false)
    }
  }

  const toggleRecording = async () => {
    if (isBusy) return
    setErrorBanner(null)

    // Nếu đang ghi âm -> bấm lần nữa để Kết thúc và gửi về /stt
    if (isRecording && mediaRecorderRef.current) {
      if (timerRef.current) {
        window.clearInterval(timerRef.current)
        timerRef.current = null
      }
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      return
    }

    // Bắt đầu phiên ghi âm mới
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const mimeType = recorder.mimeType || 'audio/webm'
        const ext = mimeType.includes('mp4') ? 'm4a' : mimeType.includes('ogg') ? 'ogg' : 'webm'
        const audioBlob = new Blob(chunksRef.current, { type: mimeType })
        if (audioBlob.size > 0) {
          void handleMicBlobToInput(audioBlob, `recording.${ext}`)
        }
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setRecordingSec(0)
      setIsRecording(true)
      timerRef.current = window.setInterval(() => {
        setRecordingSec((s) => s + 1)
      }, 1000)
    } catch {
      setErrorBanner('Không thể truy cập Microphone. Hãy cấp quyền sử dụng Mic trên trình duyệt.')
    }
  }

  // Luồng 1: Chọn file âm thanh -> đính kèm file (không hiện chữ lên input), chờ bấm Gửi
  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setErrorBanner(null)

    if (file.size > 5 * 1024 * 1024) {
      setErrorBanner('File âm thanh vượt quá giới hạn tối đa 5 MB.')
      return
    }

    const duration = await getAudioDuration(file)
    setAttachedAudio({ file, duration })
  }

  const handleSend = () => {
    if (isBusy || isRecording) return
    setErrorBanner(null)

    // Luồng 1: Nếu có file âm thanh đính kèm -> đẩy file tới /ask/voice/stream và hiện bong bóng voice
    if (attachedAudio && onSendVoice) {
      const { file, duration } = attachedAudio
      setAttachedAudio(null)
      onSendVoice(file, duration)
      return
    }

    // Luồng 2 / Gõ phím thường: Gửi câu hỏi text trên input tới /ask/stream
    const trimmed = value.trim()
    if (!trimmed) return
    onSend(trimmed)
    setValue('')
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const wrapperClass = position === 'center' ? 'w-full max-w-[760px]' : 'w-full max-w-[760px]'

  const placeholderText = isRecording
    ? `Đang ghi âm (${formatSec(recordingSec)})... Bấm nút dừng màu đỏ để hiện chữ lên đây`
    : isTranscribing
      ? 'Đang chuyển giọng nói thành văn bản...'
      : attachedAudio
        ? 'Bấm nút Gửi để hỏi bằng file âm thanh đã chọn...'
        : disabled
          ? 'Đang suy nghĩ...'
          : 'Cứ hỏi nhé'

  const canSend = !isBusy && !isRecording && (Boolean(attachedAudio) || Boolean(value.trim()))

  return (
    <div className={wrapperClass}>
      {/* Chip hiển thị file âm thanh đã chọn (Luồng 1) */}
      {attachedAudio && (
        <div className="mb-2 flex items-center gap-2">
          <div className="inline-flex items-center gap-2.5 bg-[#1e3c66] text-white text-xs px-3 py-1.5 rounded-full border border-white/15 shadow-sm">
            <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 18V5l12-2v13" />
              <circle cx="6" cy="18" r="3" />
              <circle cx="18" cy="16" r="3" />
            </svg>
            <span className="truncate max-w-[220px] font-medium">{attachedAudio.file.name}</span>
            <span className="text-white/80 font-mono">({formatSec(attachedAudio.duration)})</span>
            <button
              type="button"
              onClick={() => setAttachedAudio(null)}
              title="Bỏ chọn file âm thanh"
              className="ml-1 hover:text-red-300 transition-colors cursor-pointer"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Thông báo lỗi nhỏ nếu có */}
      {errorBanner && (
        <div className="mb-2 text-xs text-red-400 px-3 flex items-center justify-between">
          <span>{errorBanner}</span>
          <button type="button" onClick={() => setErrorBanner(null)} className="text-red-300 hover:text-white ml-2">
            ✕
          </button>
        </div>
      )}

      <div
        className={`w-full bg-[#212121] rounded-full h-[52px] px-3.5 flex items-center justify-between border transition-colors ${
          isRecording
            ? 'border-red-500/80'
            : isBusy
              ? 'opacity-70 cursor-not-allowed border-transparent'
              : 'border-transparent focus-within:border-[#383838]'
        }`}
      >
        {/* Input chọn file âm thanh ẩn (Luồng 1) */}
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.wav,.mp3,.m4a,.webm,.ogg,.flac"
          className="hidden"
          onChange={handleFileChange}
        />

        <div className="flex items-center flex-1 gap-2 h-full px-1">
          {/* Nút Upload File Âm thanh (Luồng 1) */}
          <button
            type="button"
            aria-label="Tải lên file âm thanh"
            title="Đính kèm file âm thanh (.wav, .mp3, .m4a...)"
            disabled={isBusy || isRecording}
            onClick={() => fileInputRef.current?.click()}
            className="w-8 h-8 rounded-full flex items-center justify-center text-[#9e9e9e] hover:text-white hover:bg-[#303030] transition-colors disabled:opacity-40 cursor-pointer flex-shrink-0"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48" />
            </svg>
          </button>

          <input
            ref={inputRef}
            className="w-full bg-transparent border-0 text-[15px] text-white placeholder-[#858585] focus:outline-none focus:ring-0 p-0 font-normal leading-normal text-left disabled:cursor-not-allowed"
            placeholder={placeholderText}
            type="text"
            value={value}
            disabled={isBusy || isRecording}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
          />
        </div>

        <div className="flex items-center gap-1.5 pr-0.5">
          {/* Nút Mic Ghi âm -> Kết thúc hiện lên Input (Luồng 2) */}
          <button
            type="button"
            aria-label={isRecording ? 'Kết thúc ghi âm' : 'Ghi âm giọng nói'}
            title={isRecording ? 'Bấm để kết thúc và chuyển thành văn bản' : 'Ghi âm giọng nói'}
            disabled={isBusy}
            onClick={toggleRecording}
            className={`w-8 h-8 rounded-full flex items-center justify-center transition-colors cursor-pointer ${
              isRecording
                ? 'bg-red-500 text-white animate-pulse'
                : 'text-[#9e9e9e] hover:text-white hover:bg-[#303030] disabled:opacity-40'
            }`}
          >
            {isRecording ? (
              /* Icon Kết thúc (Stop Square) */
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              /* Icon Microphone */
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
                <rect x="9" y="2" width="6" height="12" rx="3" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 10v1a7 7 0 0014 0v-1M12 18v4m-4 0h8" />
              </svg>
            )}
          </button>

          <SendButton onClick={handleSend} disabled={!canSend} />
        </div>
      </div>
    </div>
  )
}
