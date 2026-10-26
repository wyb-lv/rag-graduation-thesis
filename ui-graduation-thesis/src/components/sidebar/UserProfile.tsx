import type { User } from '@/types'
import { UserAvatar } from './UserAvatar'
import { useChat } from '@/context/ChatContext'

interface UserProfileProps {
  user: User
}

export function UserProfile({ user }: UserProfileProps) {
  const { logout } = useChat()

  return (
    <div className="h-14 px-2.5 border-t border-white/5 flex items-center justify-between gap-2 flex-shrink-0">
      <div className="flex items-center gap-3 min-w-0">
        <UserAvatar initials={user.initials} color={user.avatarColor} />
        <div className="min-w-0 flex flex-col">
          <span className="text-[14px] text-white font-medium truncate leading-tight">
            {user.name}
          </span>
          <span className="text-xs text-[#8e8e8e] truncate">
            MSSV: {user.studentId}
          </span>
        </div>
      </div>

      <button
        type="button"
        onClick={logout}
        title="Đăng xuất"
        aria-label="Đăng xuất"
        className="w-8 h-8 rounded-lg flex items-center justify-center text-[#8e8e8e] hover:text-red-400 hover:bg-white/5 transition-colors flex-shrink-0"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.8} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
        </svg>
      </button>
    </div>
  )
}