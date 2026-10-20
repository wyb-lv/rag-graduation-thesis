import { useState, type KeyboardEvent } from 'react'
import { SendButton } from './SendButton'

interface ChatInputProps {
  onSend: (message: string) => void
  position: 'center' | 'bottom'
  disabled?: boolean
}

export function ChatInput({ onSend, position, disabled = false }: ChatInputProps) {
  const [value, setValue] = useState('')

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const wrapperClass = position === 'center'
    ? 'w-full max-w-[760px]'
    : 'w-full max-w-[760px]'

  return (
    <div className={wrapperClass}>
      <div
        className={`w-full bg-[#212121] rounded-full h-[52px] px-3.5 flex items-center justify-between border border-transparent transition-colors ${
          disabled ? 'opacity-70 cursor-not-allowed' : 'focus-within:border-[#383838]'
        }`}
      >
        <div className="flex items-center flex-1 gap-2.5 h-full px-2">
          <input
            className="w-full bg-transparent border-0 text-[15px] text-white placeholder-[#858585] focus:outline-none focus:ring-0 p-0 font-normal leading-normal text-left disabled:cursor-not-allowed"
            placeholder={disabled ? 'Đang suy nghĩ...' : 'Cứ hỏi nhé'}
            type="text"
            value={value}
            disabled={disabled}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
          />
        </div>
        <div className="flex items-center gap-2 pr-0.5">
          <SendButton onClick={handleSend} disabled={disabled || !value.trim()} />
        </div>
      </div>
    </div>
  )
}
