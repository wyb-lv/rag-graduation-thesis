import type { Conversation } from '@/types'
import { ChatHistoryItem } from './ChatHistoryItem'

interface ChatHistoryListProps {
  conversations: Conversation[]
  activeId: string | null
  onSelect: (id: string) => void
}

export function ChatHistoryList({ conversations, activeId, onSelect }: ChatHistoryListProps) {
  return (
    <div className="mb-2">
      <div className="px-2.5 py-1 text-xs text-[#8e8e8e] font-medium">Đoạn chat</div>
      <div className="space-y-0.5 text-[14px]">
        {conversations.map(conv => (
          <ChatHistoryItem
            key={conv.id}
            title={conv.title}
            isActive={conv.id === activeId}
            onClick={() => onSelect(conv.id)}
          />
        ))}
      </div>
    </div>
  )
}
