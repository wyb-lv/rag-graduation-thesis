import { ChatProvider, useChat } from '@/context/ChatContext'
import { AppLayout } from '@/components/layout/AppLayout'
import { GreetingView } from '@/components/chat/GreetingView'
import { ChatArea } from '@/components/chat/ChatArea'
import { ChatInput } from '@/components/chat/ChatInput'
import { AuthScreen } from '@/components/auth/AuthScreen'
import { OcrDemoView } from '@/components/ocr/OcrDemoView'

function MainContent() {
  const { token, activeView, messages, isLoading, sendMessage } = useChat()

  if (!token) {
    return <AuthScreen />
  }

  return (
    <AppLayout>
      {activeView === 'ocr' ? (
        <OcrDemoView />
      ) : messages.length === 0 ? (
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
      <MainContent />
    </ChatProvider>
  )
}