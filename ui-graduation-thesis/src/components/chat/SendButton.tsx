import { SendArrowIcon } from '@/components/icons'

interface SendButtonProps {
  onClick: () => void
  disabled?: boolean
}

export function SendButton({ onClick, disabled }: SendButtonProps) {
  return (
    <button
      aria-label="Gửi tin nhắn"
      className="w-[34px] h-[34px] bg-[#1d63d6] hover:bg-[#256deb] text-white rounded-full flex items-center justify-center transition-all ml-0.5 shadow-sm disabled:opacity-50"
      type="button"
      onClick={onClick}
      disabled={disabled}
    >
      <SendArrowIcon className="w-4 h-4 text-white" />
    </button>
  )
}
