import { SidebarLeftIcon, SearchIcon, PencilIcon, MoreDotsIcon } from '@/components/icons'
import { ChatHistoryList } from '@/components/sidebar/ChatHistoryList'
import { UserProfile } from '@/components/sidebar/UserProfile'
import { useChat } from '@/context/ChatContext'

export function LeftSidebar() {
  const { user, conversations, activeId, toggleLeftSidebar, startNewChat, selectConversation } = useChat()

  return (
    <aside className="w-[260px] h-full flex flex-col justify-between bg-black text-[#ECECEC] z-20 flex-shrink-0 select-none border-r border-white/5">
      {/* Header: Title bên trái, Search + Toggle bên phải chữ */}
      <div className="h-14 flex items-center justify-between px-3 flex-shrink-0">
        <span className="text-[17px] font-semibold tracking-tight text-white pl-1">
          {import.meta.env.VITE_APP_TITLE || 'ChatHSU'}
        </span>
        <div className="flex items-center gap-0.5 text-[#b4b4b4]">
          <button
            aria-label="Tìm kiếm"
            className="w-9 h-9 flex items-center justify-center hover:text-white rounded-lg hover:bg-white/5 transition-colors"
            type="button"
          >
            <SearchIcon className="w-5 h-5" />
          </button>
          <button
            aria-label="Đóng thanh bên"
            className="w-9 h-9 flex items-center justify-center hover:text-white rounded-lg hover:bg-white/5 transition-colors"
            type="button"
            onClick={toggleLeftSidebar}
          >
            <SidebarLeftIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Main scroll list: Container px-2 (8px), pt-1.5 (6px) */}
      <div className="flex-1 flex flex-col min-h-0 overflow-y-auto no-scrollbar px-2 pt-1.5">
        <div className="space-y-1 mb-5 text-[14px]">
          {/* Hàng 2: Đoạn chat mới - pl-2 (8px) + container 8px = đúng 16px (Tâm X = 26px) */}
          <button
            className="w-full h-9 flex items-center gap-3 pl-2 pr-3 rounded-lg text-white hover:bg-[#212121] transition-colors text-left font-normal"
            type="button"
            onClick={startNewChat}
          >
            <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
              <PencilIcon className="w-5 h-5 text-white/90" />
            </div>
            <span className="truncate">Đoạn chat mới</span>
          </button>

          {/* Hàng 3: Thêm - pl-2 (8px) + container 8px = đúng 16px (Tâm X = 26px) */}
          <button
            className="w-full h-9 flex items-center gap-3 pl-2 pr-3 rounded-lg text-white hover:bg-[#212121] transition-colors text-left font-normal"
            type="button"
          >
            <div className="w-5 h-5 flex items-center justify-center flex-shrink-0">
              <MoreDotsIcon className="w-5 h-5 text-white/90" />
            </div>
            <span className="truncate">Thêm</span>
          </button>
        </div>

        {/* Danh sách lịch sử chat */}
        <ChatHistoryList
          conversations={conversations}
          activeId={activeId}
          onSelect={selectConversation}
        />
      </div>

      {/* Hồ sơ người dùng ở đáy */}
      <UserProfile user={user} />
    </aside>
  )
}