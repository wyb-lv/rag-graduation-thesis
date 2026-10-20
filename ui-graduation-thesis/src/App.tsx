import { ChatProvider, useChat } from '@/context/ChatContext'
import { AppLayout } from '@/components/layout/AppLayout'
import { GreetingView } from '@/components/chat/GreetingView'
import { ChatArea } from '@/components/chat/ChatArea'
import { ChatInput } from '@/components/chat/ChatInput'

function ChatView() {
  const { messages, isLoading, sendMessage } = useChat()

  return (
    <AppLayout>
      {messages.length === 0 ? (
        <GreetingView onSend={sendMessage} disabled={isLoading} />
      ) : (
        <>
          <ChatArea messages={messages} isLoading={isLoading} />
          <footer className="w-full pb-3 pt-1 px-4 flex flex-col items-center flex-shrink-0">
            <ChatInput onSend={sendMessage} position="bottom" disabled={isLoading} />
          </footer>
        </>
      )}
    </AppLayout>
  )
}

export default function App() {
  return (
    <ChatProvider>
      <ChatView />
    </ChatProvider>
  )
}