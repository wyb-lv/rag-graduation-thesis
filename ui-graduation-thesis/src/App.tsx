import { ChatProvider, useChat } from '@/context/ChatContext'
import { AppLayout } from '@/components/layout/AppLayout'
import { GreetingView } from '@/components/chat/GreetingView'
import { ChatArea } from '@/components/chat/ChatArea'
import { ChatInput } from '@/components/chat/ChatInput'
import { VoiceCallView } from '@/components/chat/VoiceCallView'
import { AuthScreen } from '@/components/auth/AuthScreen'
import { OcrDemoView } from '@/components/ocr/OcrDemoView'
import { AccountManagementView } from '@/components/account/AccountManagementView'

function MainContent() {
  const { token, isAdmin, activeView, setActiveView, messages, isLoading, sendMessage, sendVoiceMessage } = useChat()

  if (!token) {
    return <AuthScreen />
  }

  return (
    <AppLayout>
      {activeView === 'accounts' ? (
        <AccountManagementView />
      ) : activeView === 'ocr' && isAdmin ? (
        <OcrDemoView />
      ) : activeView === 'call' ? (
        <VoiceCallView onEndCall={() => setActiveView('chat')} />
      ) : messages.length === 0 ? (
        <GreetingView onSend={sendMessage} onSendVoice={sendVoiceMessage} disabled={isLoading} />
      ) : (
        <>
          <ChatArea messages={messages} isLoading={isLoading} />
          <footer className="w-full pb-3 pt-1 px-4 flex flex-col items-center flex-shrink-0">
            <ChatInput onSend={sendMessage} onSendVoice={sendVoiceMessage} position="bottom" disabled={isLoading} />
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
