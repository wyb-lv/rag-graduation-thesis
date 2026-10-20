import { SidebarRightIcon } from '@/components/icons'
import { useChat } from '@/context/ChatContext'

export function RightSidebarRail() {
  const { toggleRightSidebar, sources } = useChat()

  return (
    <aside className="w-[52px] h-full flex flex-col justify-between items-center z-20 flex-shrink-0 bg-black border-l border-white/5 select-none">
      <div className="flex flex-col items-center w-full">
        {/* Nút Toggle mở thanh tài liệu - chuẩn h-14 (Y = 28px) */}
        <div className="h-14 w-full flex items-center justify-center flex-shrink-0">
          <button
            aria-label="Mở thanh tài liệu"
            className="relative p-1.5 text-white/90 hover:text-white rounded-lg hover:bg-white/5 transition-colors"
            type="button"
            onClick={toggleRightSidebar}
          >
            <SidebarRightIcon className="w-5 h-5" />
            {sources.length > 0 && (
              <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-[#1d63d6]" />
            )}
          </button>
        </div>
      </div>
    </aside>
  )
}