import { useState, useRef, useEffect } from 'react'
import { CopyIcon } from '@/components/icons'
import { playAudioResponseStream, synthesizeSpeechResponse } from '@/services/conversationService'

interface MessageActionsProps {
  content?: string
}

export function MessageActions({ content = '' }: MessageActionsProps) {
  const [ttsState, setTtsState] = useState<'idle' | 'loading' | 'playing'>('idle')
  const [copied, setCopied] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const cachedBlobRef = useRef<{ text: string; blob: Blob } | null>(null)
  const cachedObjectUrlRef = useRef<string | null>(null)
  const copyTimerRef = useRef<number | null>(null)

  const stopPlayback = () => {
    abortRef.current?.abort()
    abortRef.current = null
    if (audioRef.current) {
      audioRef.current.onended = null
      audioRef.current.onerror = null
      audioRef.current.pause()
      audioRef.current.src = ''
      audioRef.current = null
    }
    if (cachedObjectUrlRef.current) {
      URL.revokeObjectURL(cachedObjectUrlRef.current)
      cachedObjectUrlRef.current = null
    }
    setTtsState('idle')
  }

  useEffect(() => {
    return () => {
      stopPlayback()
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current)
      }
    }
  }, [])

  const handleCopy = async () => {
    const cleanText = content.trim()
    if (!cleanText) return

    try {
      await navigator.clipboard.writeText(cleanText)
      setCopied(true)
      if (copyTimerRef.current !== null) {
        window.clearTimeout(copyTimerRef.current)
      }
      copyTimerRef.current = window.setTimeout(() => {
        setCopied(false)
        copyTimerRef.current = null
      }, 2000)
    } catch (err) {
      console.error('[Copy Error]:', err)
    }
  }

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
      if (cachedBlobRef.current && cachedBlobRef.current.text === cleanText) {
        const objectUrl = URL.createObjectURL(cachedBlobRef.current.blob)
        cachedObjectUrlRef.current = objectUrl
        const audio = new Audio(objectUrl)
        audioRef.current = audio

        const done = () => {
          URL.revokeObjectURL(objectUrl)
          if (cachedObjectUrlRef.current === objectUrl) {
            cachedObjectUrlRef.current = null
          }
          if (audioRef.current === audio) {
            audioRef.current = null
          }
          setTtsState('idle')
        }

        audio.onended = done
        audio.onerror = done
        setTtsState('playing')
        await audio.play()
        return
      }

      const res = await synthesizeSpeechResponse(cleanText, controller.signal)
      if (!res || controller.signal.aborted) {
        setTtsState('idle')
        return
      }

      const audio = await playAudioResponseStream(
        res,
        {
          onPlayStart: () => {
            if (!controller.signal.aborted) {
              setTtsState('playing')
            }
          },
          onEnded: () => {
            if (audioRef.current === audio) {
              audioRef.current = null
            }
            setTtsState('idle')
          },
          onBlobCollected: (blob) => {
            cachedBlobRef.current = { text: cleanText, blob }
          },
        },
        controller.signal
      )

      if (controller.signal.aborted) {
        audio.pause()
        audio.src = ''
        return
      }
      if (audio.src) {
        audioRef.current = audio
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        console.error('[TTS Error]:', err)
      }
      stopPlayback()
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

      <button
        aria-label={copied ? 'Đã sao chép' : 'Sao chép'}
        title={copied ? 'Đã sao chép' : 'Sao chép'}
        onClick={handleCopy}
        disabled={!content.trim()}
        className={`transition-colors cursor-pointer disabled:opacity-40 ${
          copied ? 'text-emerald-400' : 'hover:text-white'
        }`}
        type="button"
      >
        {copied ? (
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            viewBox="0 0 24 24"
          >
            <polyline points="20 6 9 17 4 12" />
          </svg>
        ) : (
          <CopyIcon />
        )}
      </button>
    </div>
  )
}
