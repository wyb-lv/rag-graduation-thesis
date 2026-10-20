import { createContext, useContext, useState, type ReactNode } from 'react'
import type { Conversation, Message, User, SourceCitation } from '@/types'
import rawData from '@/data/conversations.json'
import {askQuestionStream} from "@/services/conversationService.ts";

// import { askQuestion } from '@/services/conversationService'
export type RightSidebarTab = 'hidden' | 'sources' | 'detail'

interface ChatContextType {
  // Left Sidebar
  isLeftSidebarExpanded: boolean
  toggleLeftSidebar: () => void

  // Right Sidebar (Tài liệu tham chiếu)
  rightSidebarTab: RightSidebarTab
  setRightSidebarTab: (tab: RightSidebarTab) => void
  toggleRightSidebar: () => void
  selectedSourceIndex: number
  setSelectedSourceIndex: (idx: number) => void
  openSourceDetail: (index?: number, msgSources?: SourceCitation[]) => void

  // User
  user: User

  // Chat & Conversations
  conversations: Conversation[]
  activeId: string | null
  messages: Message[]
  isLoading: boolean
  sources: SourceCitation[]

  // Actions
  selectConversation: (id: string) => void
  startNewChat: () => void
  sendMessage: (content: string) => Promise<void>
}

const ChatContext = createContext<ChatContextType | null>(null)

export function ChatProvider({ children }: { children: ReactNode }) {
const [isLeftSidebarExpanded, setIsLeftSidebarExpanded] = useState(false)
  const [rightSidebarTab, setRightSidebarTab] = useState<RightSidebarTab>('hidden')
  const [selectedSourceIndex, setSelectedSourceIndex] = useState(0)

  const [conversations, setConversations] = useState<Conversation[]>(rawData.conversations as Conversation[])
  const [activeId, setActiveId] = useState<string | null>('1')
  const [sources, setSources] = useState<SourceCitation[]>(rawData.sources as SourceCitation[])
  const [isLoading, setIsLoading] = useState(false)

  const activeConversation = conversations.find((c) => c.id === activeId)
  const [currentMessages, setCurrentMessages] = useState<Message[]>(activeConversation?.messages || [])

  const user: User = rawData.user as User

  const toggleLeftSidebar = () => setIsLeftSidebarExpanded((prev) => !prev)

  const toggleRightSidebar = () => {
    setRightSidebarTab((prev) => (prev === 'hidden' ? 'sources' : 'hidden'))
  }

  const openSourceDetail = (index = 0, msgSources?: SourceCitation[]) => {
    if (msgSources && msgSources.length > 0) {
      setSources(msgSources)
    }
    setSelectedSourceIndex(index)
    setRightSidebarTab('detail')
  }

  const selectConversation = (id: string) => {
    setActiveId(id)
    const conv = conversations.find((c) => c.id === id)
    setCurrentMessages(conv?.messages || [])
    // Reset sources theo tin nhắn cuối cùng của hội thoại đó nếu có
    const lastMsgWithSources = conv?.messages.slice().reverse().find((m) => m.sources && m.sources.length > 0)
    if (lastMsgWithSources?.sources) {
      setSources(lastMsgWithSources.sources)
    }
  }

  const startNewChat = () => {
    setActiveId(null)
    setCurrentMessages([])
    setRightSidebarTab('hidden')
  }
  const sendMessage = async (content: string) => {
    const userMsg: Message = { id: `user-${Date.now()}`, role: 'user', content }
    const botMsgId = `assistant-${Date.now()}`
    const botMsgPlaceholder: Message = { id: botMsgId, role: 'assistant', content: '', sources: [] }
    const updatedMessages = [...currentMessages, userMsg, botMsgPlaceholder]
    setCurrentMessages(updatedMessages)
    setIsLoading(true)
    let currentChatId = activeId

    if (!currentChatId) {
      currentChatId = `chat-${Date.now()}`
      setActiveId(currentChatId)
      const newConv: Conversation = {
        id: currentChatId,
        title: content.slice(0, 30),
        messages: updatedMessages,
      }
      setConversations((prev) => [newConv, ...prev])
    }
    let accumulatedContent = ''
    let latestSources: SourceCitation[] = []
    try {
      await askQuestionStream(content, {
        onSources: (incomingSources) => {
          latestSources = incomingSources
          setSources(incomingSources)
          setCurrentMessages((prev) =>
            prev.map((msg) => (msg.id === botMsgId ? { ...msg, sources: incomingSources } : msg))
          )
        },
        onDelta: (deltaText) => {
          accumulatedContent += deltaText
          setCurrentMessages((prev) =>
            prev.map((msg) => (msg.id === botMsgId ? { ...msg, content: accumulatedContent } : msg))
          )
        },
      })

      const finalBotMsg: Message = {
        id: botMsgId,
        role: 'assistant',
        content: accumulatedContent,
        sources: latestSources,
      }

      const finalConversationMessages = [...currentMessages, userMsg, finalBotMsg]
      setConversations((prev) =>
        prev.map((c) => (c.id === currentChatId ? { ...c, messages: finalConversationMessages } : c))
      )
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : 'Có lỗi xảy ra'
      // Giữ phần đã stream được (nếu có) và nối thông báo lỗi vào chính tin nhắn đang stream
      const errorContent = `${accumulatedContent ? `${accumulatedContent}\n\n` : ''}Xin lỗi, đã xảy ra lỗi: ${errMsg}`
      const errorMsg: Message = { id: botMsgId, role: 'assistant', content: errorContent, sources: latestSources }
      setCurrentMessages((prev) => prev.map((msg) => (msg.id === botMsgId ? errorMsg : msg)))
      setConversations((prev) =>
        prev.map((c) => (c.id === currentChatId ? { ...c, messages: [...currentMessages, userMsg, errorMsg] } : c))
      )
    } finally {
      setIsLoading(false)
    }
  }

  // const sendMessage = async (content: string) => {
  //   const userMsg: Message = {
  //     id: `user-${Date.now()}`,
  //     role: 'user',
  //     content,
  //   }
  //
  //   // 1. Cập nhật ngay tin nhắn của người dùng lên UI
  //   const updatedMessages = [...currentMessages, userMsg]
  //   setCurrentMessages(updatedMessages)
  //   setIsLoading(true)
  //
  //   // 2. Tự động đồng bộ hoặc tạo đoạn chat mới trong danh sách bên trái
  //   let currentChatId = activeId
  //   if (!currentChatId) {
  //     currentChatId = `chat-${Date.now()}`
  //     setActiveId(currentChatId)
  //     const newConv: Conversation = {
  //       id: currentChatId,
  //       title: content.slice(0, 30),
  //       messages: updatedMessages,
  //     }
  //     setConversations((prev) => [newConv, ...prev])
  //   } else {
  //     setConversations((prev) =>
  //       prev.map((c) => (c.id === currentChatId ? { ...c, messages: updatedMessages } : c))
  //     )
  //   }
  //
  //   try {
  //     // 3. Gọi API (kèm fallback mock)
  //     const res = await askQuestion(content)
  //     const botMsg: Message = {
  //       id: `assistant-${Date.now()}`,
  //       role: 'assistant',
  //       content: res.answer,
  //       sources: res.sources,
  //     }
  //
  //     const finalMessages = [...updatedMessages, botMsg]
  //     setCurrentMessages(finalMessages)
  //     setSources(res.sources)
  //
  //     // Cập nhật lại lịch sử với câu trả lời của bot
  //     setConversations((prev) =>
  //       prev.map((c) => (c.id === currentChatId ? { ...c, messages: finalMessages } : c))
  //     )
  //   } catch (err: unknown) {
  //     const errMsg = err instanceof Error ? err.message : 'Có lỗi xảy ra'
  //     const errorMsg: Message = {
  //       id: `assistant-err-${Date.now()}`,
  //       role: 'assistant',
  //       content: `Xin lỗi, đã xảy ra lỗi: ${errMsg}`,
  //     }
  //     setCurrentMessages([...updatedMessages, errorMsg])
  //   } finally {
  //     setIsLoading(false)
  //   }
  // }

  return (
    <ChatContext.Provider
      value={{
        isLeftSidebarExpanded,
        toggleLeftSidebar,
        rightSidebarTab,
        setRightSidebarTab,
        toggleRightSidebar,
        selectedSourceIndex,
        setSelectedSourceIndex,
        openSourceDetail,
        user,
        conversations,
        activeId,
        messages: currentMessages,
        isLoading,
        sources,
        selectConversation,
        startNewChat,
        sendMessage,
      }}
    >
      {children}
    </ChatContext.Provider>
  )
}

export function useChat() {
  const ctx = useContext(ChatContext)
  if (!ctx) throw new Error('useChat must be used within a ChatProvider')
  return ctx
}