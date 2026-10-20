interface ChatHistoryItemProps {
  title: string
  isActive: boolean
  onClick: () => void
}

export function ChatHistoryItem({ title, isActive, onClick }: ChatHistoryItemProps) {
  return (
    <button
      className={`w-full flex items-center px-3 py-2 rounded-xl text-left font-normal truncate transition-colors ${
        isActive ? 'bg-[#212121] text-white' : 'text-white/90 hover:bg-[#212121]'
      }`}
      type="button"
      onClick={onClick}
    >
      {title}
    </button>
  )
}