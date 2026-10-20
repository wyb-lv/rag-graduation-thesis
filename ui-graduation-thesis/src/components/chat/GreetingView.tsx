import { ChatInput } from './ChatInput'

interface GreetingViewProps {
  onSend: (message: string) => void
  disabled?: boolean
}

export function GreetingView({ onSend, disabled = false }: GreetingViewProps) {
  return (
    <main className="flex-1 flex flex-col items-center justify-center px-4 -mt-16">
      <div className="w-full max-w-[760px] flex flex-col items-center">
        <h1 className="text-white text-[28px] font-medium tracking-tight mb-8">
          Tôi có thể giúp gì cho bạn?
        </h1>
        <ChatInput onSend={onSend} position="center" disabled={disabled} />
      </div>
    </main>
  )
}
