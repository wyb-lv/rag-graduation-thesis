import type { Message } from '@/types'

interface UserMessageProps {
  message: Message
}

export function UserMessage({ message }: UserMessageProps) {
  return (
    <article aria-label="Tin nhắn của bạn" className="flex justify-end w-full">
      <div className="bg-[#1e3c66] text-white text-[15px] px-[18px] py-[8px] rounded-full max-w-[85%] leading-relaxed select-text">
        {message.content}
      </div>
    </article>
  )
}
