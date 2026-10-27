import { useState, useRef, useEffect } from 'react'
import type { Message } from '@/types'

interface UserMessageProps {
  message: Message
}

function formatDuration(seconds?: number): string {
  if (!seconds || !Number.isFinite(seconds) || seconds < 0) return '0:00'
  const totalSec = Math.round(seconds)
  const mins = Math.floor(totalSec / 60)
  const secs = totalSec % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

const WAVE_BARS = [40, 75, 55, 90, 65, 100, 80, 50, 85, 60, 95, 70, 45, 80, 60, 50]

export function UserMessage({ message }: UserMessageProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [duration, setDuration] = useState<number>(message.audioDuration || 0)
  const [currentTime, setCurrentTime] = useState<number>(0)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    if (message.audioDuration && message.audioDuration > 0) {
      setDuration(message.audioDuration)
    }
  }, [message.audioDuration])

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
    }
  }, [])

  const togglePlay = () => {
    if (!message.audioUrl) return

    if (!audioRef.current) {
      const audio = new Audio(message.audioUrl)
      audio.onloadedmetadata = () => {
        if (Number.isFinite(audio.duration) && audio.duration > 0) {
          setDuration(audio.duration)
        }
      }
      audio.ontimeupdate = () => {
        setCurrentTime(audio.currentTime)
      }
      audio.onended = () => {
        setIsPlaying(false)
        setCurrentTime(0)
      }
      audio.onerror = () => {
        setIsPlaying(false)
      }
      audioRef.current = audio
    }

    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
    } else {
      audioRef.current
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => setIsPlaying(false))
    }
  }

  if (message.audioUrl) {
    const progressRatio = duration > 0 ? Math.min(1, currentTime / duration) : 0
    const activeBars = Math.round(progressRatio * WAVE_BARS.length)

    return (
      <article aria-label="Tin nhắn thoại của bạn" className="flex justify-end w-full">
        <div className="bg-[#1e3c66] text-white px-3.5 py-2.5 rounded-full flex items-center gap-3 shadow-sm select-none">
          <button
            type="button"
            onClick={togglePlay}
            aria-label={isPlaying ? 'Tạm dừng' : 'Phát đoạn ghi âm'}
            title={isPlaying ? 'Tạm dừng' : 'Bấm để nghe lại đoạn voice'}
            className="w-8 h-8 rounded-full bg-white text-[#1e3c66] flex items-center justify-center hover:bg-white/90 transition-colors cursor-pointer flex-shrink-0"
          >
            {isPlaying ? (
              <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="5" width="4" height="14" rx="1" />
                <rect x="14" y="5" width="4" height="14" rx="1" />
              </svg>
            ) : (
              <svg className="w-3.5 h-3.5 ml-0.5" fill="currentColor" viewBox="0 0 24 24">
                <polygon points="6 4 20 12 6 20 6 4" />
              </svg>
            )}
          </button>

          {/* Sóng âm trực quan */}
          <div className="flex items-center gap-[3px] h-5 px-1" aria-hidden="true">
            {WAVE_BARS.map((heightPct, idx) => {
              const isHighlighted = isPlaying && idx <= activeBars
              return (
                <span
                  key={idx}
                  style={{ height: `${Math.max(25, heightPct)}%` }}
                  className={`w-[3px] rounded-full transition-colors ${
                    isHighlighted ? 'bg-white' : 'bg-white/60'
                  }`}
                />
              )
            })}
          </div>

          {/* Thời lượng đoạn voice */}
          <span className="text-xs font-mono text-white/90 min-w-[34px] text-right pr-1">
            {formatDuration(isPlaying ? currentTime : duration)}
          </span>
        </div>
      </article>
    )
  }

  return (
    <article aria-label="Tin nhắn của bạn" className="flex justify-end w-full">
      <div className="bg-[#1e3c66] text-white text-[15px] px-[18px] py-[8px] rounded-full max-w-[85%] leading-relaxed select-text">
        {message.content}
      </div>
    </article>
  )
}
