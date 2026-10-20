import type { Message } from '@/types'
import { MessageActions } from './MessageActions'
import { useChat } from "@/context/ChatContext.tsx";

interface AssistantMessageProps {
  message: Message
}

function formatAnswer(content: string): string {
  if (!content) return ''

  return content
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/(\S)\s+([–—-]\s+)/g, '$1\n$2')
    .replace(/(?:^|\r\n|\r|\n|\s)\*+\s*/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trimStart() // Dùng .trimStart() thay cho .trim() để không bị nuốt dấu cách ở cuối mỗi token khi đang stream ("Thi ")
}

export function AssistantMessage({ message }: AssistantMessageProps) {
  const { openSourceDetail } = useChat()
  const topSource =
    message.sources && message.sources.length > 0
      ? [...message.sources].sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0]
      : null
  const extraCount =
    message.sources && message.sources.length > 1
      ? message.sources.length - 1
      : 0
  const formattedContent = formatAnswer(message.content)

  return (
    <article aria-label="Phản hồi của ChatGPT" className="flex flex-col items-start w-full space-y-2 select-text">
      <div className="text-[#e2e2e2] text-[15px] leading-relaxed whitespace-pre-wrap">
        <span>{formattedContent}</span>
        {topSource && (
          <button
            type="button"
            className="inline-flex items-center gap-1 ml-2 px-2 py-0.5 rounded-full bg-[#2a2a2a] hover:bg-[#333333] border border-white/10 hover:border-white/30 text-xs text-[#e2e2e2] align-middle select-none transition-all cursor-pointer"
            title={`Click để xem đoạn trích dẫn từ: ${topSource.title}${topSource.score ? ` (Độ khớp: ${(topSource.score * 100).toFixed(0)}%)` : ''}`}
            onClick={() => openSourceDetail(0, message.sources)}
          >
            <span className="font-normal truncate max-w-[160px]">{topSource.title}</span>
            {extraCount > 0 && (
              <span className="text-[10.5px] text-[#8e8e8e] font-mono font-medium">+{extraCount}</span>
            )}
          </button>
        )}
      </div>
      <MessageActions />
    </article>
  )
}


