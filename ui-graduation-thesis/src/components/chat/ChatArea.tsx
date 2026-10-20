import { useEffect, useRef } from 'react'
import type { Message } from '@/types'
import { UserMessage } from './UserMessage'
import { AssistantMessage } from './AssistantMessage'

interface ChatAreaProps {
  messages: Message[]
  isLoading?: boolean
}

export function ChatArea({ messages, isLoading = false }: ChatAreaProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isLoading])

  const lastMessage = messages[messages.length - 1]
  const isWaitingFirstToken = isLoading && (!lastMessage || lastMessage.role === 'user' || !lastMessage.content)

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto no-scrollbar px-4 sm:px-6">
      <div className="max-w-[760px] mx-auto w-full pt-3 pb-8 flex flex-col space-y-6">
        {messages.map((msg) =>
          msg.role === 'user' ? (
            <UserMessage key={msg.id} message={msg} />
          ) : (
            <AssistantMessage key={msg.id} message={msg} />
          )
        )}

        {isWaitingFirstToken && (
          <article aria-label="Đang suy nghĩ" className="flex flex-col items-start w-full space-y-2">
            <div className="flex items-center gap-1.5 py-2 px-1">
              <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
              <span className="w-2 h-2 rounded-full bg-white animate-pulse [animation-delay:200ms]" />
              <span className="w-2 h-2 rounded-full bg-white animate-pulse [animation-delay:400ms]" />
              <span className="text-xs text-[#8e8e8e] ml-2 font-mono">Đang tra cứu tài liệu & suy luận...</span>
            </div>
          </article>
        )}
      </div>
    </div>
  )
}
