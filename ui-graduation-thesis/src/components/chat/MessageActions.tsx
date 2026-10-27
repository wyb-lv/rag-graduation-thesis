import { useState, useRef, useEffect } from 'react'
import { CopyIcon, ThumbsDownIcon, ShareIcon, RegenerateIcon, MoreDotsIcon } from '@/components/icons'
import { synthesizeSpeech } from '@/services/conversationService'

interface MessageActionsProps {
  content?: string
}

export function MessageActions({ content = '' }: MessageActionsProps) {
  const [ttsState, setTtsState] = useState<'idle' | 'loading' | 'playing'>('idle')
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const cachedBlobRef = useRef<{ text: string; blob: Blob } | null>(null)

  const stopPlayback = () => {
    abortRef.current?.abort()
    abortRef.current = null
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.src = ''
      audioRef.current = null
    }
    setTtsState('idle')
  }

  useEffect(() => {
    return () => stopPlayback()
  }, [])

  const handleToggleTts = async () => {
    const cleanText = content.trim()
    if (!cleanText) return

    if (ttsState !== 'idle') {
      stopPlayback()
      return
    }

    const controller = new AbortController()
    abortRef.current = controller
    setTtsState('loading')

    try {
      let audioBlob: Blob
      if (cachedBlobRef.current && cachedBlobRef.current.text === cleanText) {
        audioBlob = cachedBlobRef.current.blob
      } else {
        audioBlob = await synthesizeSpeech(cleanText, controller.signal)
        cachedBlobRef.current = { text: cleanText, blob: audioBlob }
      }

      if (controller.signal.aborted) return

      const objectUrl = URL.createObjectURL(audioBlob)
      const audio = new Audio(objectUrl)
      audioRef.current = audio

      audio.onended = () => {
        URL.revokeObjectURL(objectUrl)
        setTtsState('idle')
      }
      audio.onerror = () => {
        URL.revokeObjectURL(objectUrl)
        setTtsState('idle')
      }

      setTtsState('playing')
      await audio.play()
    } catch (err) {
      if (!controller.signal.aborted) {
        console.error('[TTS Error]:', err)
      }
      setTtsState('idle')
    }
  }

  return (
    <div className="flex items-center gap-3 pt-1 text-[#8f8f8f]">
      {/* Nút Phát / Dừng đọc câu trả lời (Luồng 3: POST /tts) */}
      <button
        aria-label={ttsState === 'idle' ? 'Đọc phản hồi' : 'Dừng đọc'}
        title={ttsState === 'idle' ? 'Phát âm thanh câu trả lời' : 'Dừng đọc'}
        onClick={handleToggleTts}
        disabled={!content.trim()}
        className={`transition-colors cursor-pointer disabled:opacity-40 ${
          ttsState === 'playing'
            ? 'text-emerald-400 hover:text-emerald-300'
            : ttsState === 'loading'
              ? 'text-amber-400 animate-pulse'
              : 'hover:text-white'
        }`}
        type="button"
      >
        {ttsState === 'playing' ? (
          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : (
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
            strokeLinejoin="round"
            viewBox="0 0 24 24"
          >
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
          </svg>
        )}
      </button>

      <button aria-label="Sao chép" className="hover:text-white transition-colors" type="button">
        <CopyIcon />
      </button>
      <button aria-label="Không thích" className="hover:text-white transition-colors" type="button">
        <ThumbsDownIcon />
      </button>
      <button aria-label="Chia sẻ" className="hover:text-white transition-colors" type="button">
        <ShareIcon />
      </button>
      <button aria-label="Tạo lại phản hồi" className="hover:text-white transition-colors" type="button">
        <RegenerateIcon />
      </button>
      <button aria-label="Tùy chọn khác" className="hover:text-white transition-colors" type="button">
        <MoreDotsIcon />
      </button>
    </div>
  )
}
