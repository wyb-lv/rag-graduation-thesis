import { useEffect, useRef, useState } from 'react'
import {
  fetchCallGreetingAudio,
  playAudioResponseStream,
  sendCallTurnAudio,
} from '@/services/conversationService'

interface VoiceCallViewProps {
  onEndCall: () => void
}

type CallPhase = 'connecting' | 'speaking' | 'listening' | 'thinking' | 'error'

const SPEECH_THRESHOLD_DB = -40   // Ngưỡng phát hiện bắt đầu nói (dBFS)
const BARGE_IN_THRESHOLD_DB = -26 // Ngưỡng nói cắt ngang khi trợ lý đang nói (dBFS)
const BARGE_IN_HOLD_FRAMES = 3    // Số khung 60ms liên tiếp (~180ms) để xác nhận cắt ngang
const SILENCE_HOLD_MS = 1100      // Ngừng nói 1.1s -> tự động chốt câu gửi về Backend
const MIN_SPEECH_MS = 350         // Bỏ qua tiếng động ngắn dưới 350ms

function computeMicDb(analyser: AnalyserNode, pcmData: Float32Array<ArrayBuffer>): number {
  analyser.getFloatTimeDomainData(pcmData)
  let sumSq = 0
  for (let i = 0; i < pcmData.length; i++) {
    sumSq += pcmData[i] * pcmData[i]
  }
  const rms = Math.sqrt(sumSq / pcmData.length)
  return 20 * Math.log10(Math.max(rms, 1e-7))
}

export function VoiceCallView({ onEndCall }: VoiceCallViewProps) {
  const [phase, setPhase] = useState<CallPhase>('connecting')
  const [isMuted, setIsMuted] = useState(false)
  const [volumeLevel, setVolumeLevel] = useState(0)

  const abortRef = useRef<AbortController | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const audioElRef = useRef<HTMLAudioElement | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const vadTimerRef = useRef<number | null>(null)
  const bargeInTimerRef = useRef<number | null>(null)
  const isMountedRef = useRef(true)
  const isMutedRef = useRef(false)

  useEffect(() => {
    isMutedRef.current = isMuted
  }, [isMuted])

  const clearTimers = () => {
    if (vadTimerRef.current !== null) {
      window.clearInterval(vadTimerRef.current)
      vadTimerRef.current = null
    }
    if (bargeInTimerRef.current !== null) {
      window.clearInterval(bargeInTimerRef.current)
      bargeInTimerRef.current = null
    }
  }

  const stopPlaybackAndTimers = () => {
    clearTimers()
    abortRef.current?.abort()
    abortRef.current = null
    if (audioElRef.current) {
      audioElRef.current.onended = null
      audioElRef.current.onerror = null
      audioElRef.current.pause()
      audioElRef.current.src = ''
      audioElRef.current = null
    }
  }

  const cleanupAll = () => {
    stopPlaybackAndTimers()
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.onstop = null
      recorderRef.current.stop()
    }
    recorderRef.current = null
    analyserRef.current = null
    micStreamRef.current?.getTracks().forEach((t) => t.stop())
    micStreamRef.current = null
    audioCtxRef.current?.close().catch(() => {})
    audioCtxRef.current = null
  }

  const startBargeInMonitor = () => {
    if (bargeInTimerRef.current !== null) {
      window.clearInterval(bargeInTimerRef.current)
      bargeInTimerRef.current = null
    }
    const analyser = analyserRef.current
    if (!analyser) return

    const pcmData = new Float32Array(analyser.fftSize)
    let hotFrames = 0

    bargeInTimerRef.current = window.setInterval(() => {
      if (!isMountedRef.current || isMutedRef.current) {
        hotFrames = 0
        return
      }
      const db = computeMicDb(analyser, pcmData)
      if (db > BARGE_IN_THRESHOLD_DB) {
        hotFrames += 1
        if (hotFrames >= BARGE_IN_HOLD_FRAMES) {
          stopPlaybackAndTimers()
          startListeningTurn(true)
        }
      } else {
        hotFrames = 0
      }
    }, 60)
  }

  const playBackendAudioStream = async (
    res: Response,
    onFinished: () => void,
    signal?: AbortSignal
  ) => {
    if (!isMountedRef.current || signal?.aborted) return

    // Dừng ngay audio/timer trước đó (nếu có) để triệt tiêu hoàn toàn hiện tượng chồng âm/vang
    if (bargeInTimerRef.current !== null) {
      window.clearInterval(bargeInTimerRef.current)
      bargeInTimerRef.current = null
    }
    if (audioElRef.current) {
      audioElRef.current.onended = null
      audioElRef.current.onerror = null
      audioElRef.current.pause()
      audioElRef.current.src = ''
      audioElRef.current = null
    }

    const audio = await playAudioResponseStream(
      res,
      {
        onPlayStart: () => {
          if (!isMountedRef.current || signal?.aborted) return
          setPhase('speaking')
          startBargeInMonitor()
        },
        onEnded: () => {
          if (bargeInTimerRef.current !== null) {
            window.clearInterval(bargeInTimerRef.current)
            bargeInTimerRef.current = null
          }
          if (audioElRef.current === audio) {
            audioElRef.current = null
          }
          if (isMountedRef.current && !signal?.aborted) {
            onFinished()
          }
        },
      },
      signal
    )

    if (!isMountedRef.current || signal?.aborted) {
      audio.pause()
      audio.src = ''
      return
    }
    if (audio.src) {
      audioElRef.current = audio
    }
  }

  const startListeningTurn = (initialHasSpoken = false) => {
    if (!isMountedRef.current || !micStreamRef.current || !analyserRef.current) return
    clearTimers()
    setPhase('listening')

    const stream = micStreamRef.current
    const analyser = analyserRef.current
    const pcmData = new Float32Array(analyser.fftSize)
    const chunks: BlobPart[] = []
    const recorder = new MediaRecorder(stream)
    recorderRef.current = recorder

    let hasSpoken = initialHasSpoken
    let speechStartAt = initialHasSpoken ? Date.now() - 180 : 0
    let silenceStartAt: number | null = null

    recorder.ondataavailable = (e) => {
      if (e.data.size > 0) chunks.push(e.data)
    }

    recorder.onstop = async () => {
      clearTimers()
      setVolumeLevel(0)
      if (!isMountedRef.current) return

      const spokenDuration = hasSpoken ? Date.now() - speechStartAt : 0
      const mimeType = recorder.mimeType || 'audio/webm'
      const ext = mimeType.includes('mp4') ? 'm4a' : mimeType.includes('ogg') ? 'ogg' : 'webm'
      const blob = new Blob(chunks, { type: mimeType })

      if (!hasSpoken || spokenDuration < MIN_SPEECH_MS || blob.size < 1024) {
        startListeningTurn()
        return
      }

      setPhase('thinking')
      const turnController = new AbortController()
      abortRef.current = turnController
      try {
        const { response, action, sessionId } = await sendCallTurnAudio(
          blob,
          `turn.${ext}`,
          turnController.signal,
          sessionIdRef.current
        )
        if (!isMountedRef.current || turnController.signal.aborted) return

        if (sessionId) {
          sessionIdRef.current = sessionId
        }

        if (response) {
          await playBackendAudioStream(
            response,
            () => {
              if (action === 'hangup') onEndCall()
              else startListeningTurn()
            },
            turnController.signal
          )
        } else if (action === 'hangup') {
          onEndCall()
        } else {
          startListeningTurn()
        }
      } catch {
        if (isMountedRef.current && !turnController.signal.aborted) {
          setTimeout(() => startListeningTurn(), 1000)
        }
      }
    }

    recorder.start()

    vadTimerRef.current = window.setInterval(() => {
      if (isMutedRef.current) {
        setVolumeLevel(0)
        return
      }
      const db = computeMicDb(analyser, pcmData)
      const normalizedVol = Math.min(1, Math.max(0, (db + 55) / 40))
      setVolumeLevel(normalizedVol)

      const now = Date.now()
      if (db > SPEECH_THRESHOLD_DB) {
        if (!hasSpoken) {
          hasSpoken = true
          speechStartAt = now
        }
        silenceStartAt = null
      } else if (hasSpoken) {
        if (silenceStartAt === null) {
          silenceStartAt = now
        } else if (now - silenceStartAt >= SILENCE_HOLD_MS) {
          if (recorder.state === 'recording') {
            recorder.stop()
          }
        }
      }
    }, 60)
  }

  useEffect(() => {
    isMountedRef.current = true
    let cancelled = false
    const controller = new AbortController()
    abortRef.current = controller

    const initCall = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        })

        // Nếu lượt mount này đã bị hủy (do React StrictMode hoặc người dùng tắt nhanh)
        if (cancelled || !isMountedRef.current) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }

        const audioCtx = new AudioContext()
        if (audioCtx.state === 'suspended') {
          await audioCtx.resume().catch(() => {})
        }
        if (cancelled || !isMountedRef.current) {
          stream.getTracks().forEach((t) => t.stop())
          audioCtx.close().catch(() => {})
          return
        }

        const source = audioCtx.createMediaStreamSource(stream)
        const analyser = audioCtx.createAnalyser()
        analyser.fftSize = 512
        source.connect(analyser)

        micStreamRef.current = stream
        audioCtxRef.current = audioCtx
        analyserRef.current = analyser

        const { response, sessionId } = await fetchCallGreetingAudio(controller.signal)
        if (cancelled || !isMountedRef.current) return

        if (sessionId) {
          sessionIdRef.current = sessionId
        }

        await playBackendAudioStream(
          response,
          () => {
            if (!cancelled && isMountedRef.current) {
              startListeningTurn()
            }
          },
          controller.signal
        )
      } catch {
        if (!cancelled && isMountedRef.current && !controller.signal.aborted) {
          setPhase('error')
        }
      }
    }

    void initCall()

    return () => {
      cancelled = true
      isMountedRef.current = false
      controller.abort()
      cleanupAll()
    }
  }, [])

  const handleHangup = () => {
    cleanupAll()
    onEndCall()
  }

  const orbScale = phase === 'listening' ? 1 + volumeLevel * 0.35 : phase === 'speaking' ? 1.12 : 1

  return (
    <div className="flex-1 flex flex-col items-center justify-between py-14 px-6 bg-black text-white select-none">
      <div className="text-center">
        <p className="text-sm text-[#8e8e93] tracking-wide uppercase">Cuộc gọi trực tiếp • HSU AI</p>
      </div>

      <div className="flex flex-col items-center justify-center gap-8 my-auto">
        <div className="relative flex items-center justify-center w-48 h-48">
          <div
            className={`absolute inset-0 rounded-full transition-all duration-150 ${
              phase === 'speaking'
                ? 'bg-blue-500/25 animate-ping'
                : phase === 'listening'
                  ? 'bg-emerald-500/20'
                  : 'bg-amber-500/15 animate-pulse'
            }`}
            style={{ transform: `scale(${orbScale * 1.15})` }}
          />
          <div
            className={`relative w-32 h-32 rounded-full flex items-center justify-center shadow-2xl transition-transform duration-100 ${
              phase === 'speaking'
                ? 'bg-gradient-to-br from-blue-500 to-indigo-600'
                : phase === 'listening'
                  ? 'bg-gradient-to-br from-emerald-500 to-teal-600'
                  : phase === 'thinking'
                    ? 'bg-gradient-to-br from-amber-500 to-orange-600 animate-spin'
                    : 'bg-[#2c2c2e]'
            }`}
            style={{ transform: `scale(${orbScale})` }}
          >
            <svg className="w-12 h-12 text-white" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v18m-4-14v10m8-10v10M4 10v4m16-4v4" />
            </svg>
          </div>
        </div>

        <div className="h-6 flex items-center gap-2 text-sm text-[#a1a1aa]">
          {phase === 'connecting' && <span>Đang kết nối cuộc gọi...</span>}
          {phase === 'speaking' && <span className="text-blue-400">Trợ lý đang nói...</span>}
          {phase === 'listening' && (
            <span className="text-emerald-400">{isMuted ? 'Đang tắt Mic' : 'Đang nghe bạn nói...'}</span>
          )}
          {phase === 'thinking' && <span className="text-amber-400">Đang xử lý...</span>}
          {phase === 'error' && <span className="text-red-400">Không thể truy cập Mic hoặc kết nối máy chủ</span>}
        </div>
      </div>

      <div className="flex items-center gap-6">
        <button
          type="button"
          onClick={() => setIsMuted((m) => !m)}
          title={isMuted ? 'Bật lại Mic' : 'Tạm tắt Mic'}
          className={`w-14 h-14 rounded-full flex items-center justify-center transition-colors cursor-pointer ${
            isMuted ? 'bg-white text-black' : 'bg-[#2c2c2e] text-white hover:bg-[#3a3a3c]'
          }`}
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
            {isMuted ? (
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 3l18 18M9 9v3a3 3 0 005.12 2.12M15 9.34V5a3 3 0 00-5.94-.6M5 10v1a7 7 0 0011.95 4.95M12 18v4m-4 0h8" />
            ) : (
              <>
                <rect x="9" y="2" width="6" height="12" rx="3" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M5 10v1a7 7 0 0014 0v-1M12 18v4m-4 0h8" />
              </>
            )}
          </svg>
        </button>

        <button
          type="button"
          onClick={handleHangup}
          title="Kết thúc cuộc gọi"
          className="w-16 h-16 rounded-full bg-red-600 hover:bg-red-500 text-white flex items-center justify-center shadow-lg transition-colors cursor-pointer"
        >
          <svg className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M3 15.5c2.5-3 6-4.5 9-4.5s6.5 1.5 9 4.5l-2.5 2.5c-.7.7-1.8.7-2.5 0l-1.5-1.5v-2.2a11.5 11.5 0 00-5 0v2.2L8 18c-.7.7-1.8.7-2.5 0L3 15.5z"
            />
          </svg>
        </button>
      </div>
    </div>
  )
}

