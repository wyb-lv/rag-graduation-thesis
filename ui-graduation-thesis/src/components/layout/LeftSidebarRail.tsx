import { SidebarLeftIcon, PencilIcon, MoreDotsIcon, ChatBubbleIcon } from '@/components/icons'
import { UserAvatar } from '@/components/sidebar/UserAvatar'
import { useChat } from '@/context/ChatContext'

export function LeftSidebarRail() {
  const { toggleLeftSidebar, startNewChat, user } = useChat()

  return (
    <aside className="w-[52px] h-full flex flex-col justify-between items-center z-20 flex-shrink-0 bg-black select-none border-r border-white/5">
      {/* Top Group */}
      <div className="flex flex-col items-center w-full">
        {/* Hàng 1: Nút mở thanh bên (h-14, Tâm Y = 28px) */}
        <div className="h-14 w-full flex items-center justify-center flex-shrink-0">
          <button
            aria-label="Mở thanh bên trái"
            className="w-9 h-9 flex items-center justify-center text-white/90 hover:text-white rounded-lg hover:bg-white/5 transition-colors"
            type="button"
            onClick={toggleLeftSidebar}
          >
            <SidebarLeftIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Action icons: Container px-2 (8px), pt-1.5 (6px) */}
        <div className="flex flex-col items-center space-y-1 w-full px-2 pt-1.5">
          {/* Hàng 2: Đoạn chat mới - Nút w-9 (36px), Icon cách lề đúng 16px (Tâm X = 26px) */}
          <button
            aria-label="Đoạn chat mới"
            className="w-9 h-9 flex items-center justify-center text-[#b4b4b4] hover:text-white rounded-lg hover:bg-[#212121] transition-colors"
            type="button"
            onClick={startNewChat}
          >
            <PencilIcon className="w-5 h-5" />
          </button>

          {/* Hàng 3: Khớp 1-1 với nút "Thêm" bên Expanded */}
          <button
            aria-label="Thêm"
            className="w-9 h-9 flex items-center justify-center text-[#b4b4b4] hover:text-white rounded-lg hover:bg-[#212121] transition-colors"
            type="button"
          >
            <MoreDotsIcon className="w-5 h-5" />
          </button>

          {/* Hàng 4: Lịch sử chat */}
          <button
            aria-label="Lịch sử chat"
            className="w-9 h-9 flex items-center justify-center text-[#b4b4b4] hover:text-white rounded-lg hover:bg-[#212121] transition-colors"
            type="button"
            onClick={toggleLeftSidebar}
          >
            <ChatBubbleIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Đáy: Avatar căn giữa cột 52px (Cách lề đúng 10px, Tâm X = 26px) */}
      <div className="h-14 w-full flex items-center justify-center border-t border-white/5 flex-shrink-0">
        <button aria-label="Hồ sơ người dùng" type="button">
          <UserAvatar initials={user.initials} color={user.avatarColor} />
        </button>
      </div>
    </aside>
  )
}