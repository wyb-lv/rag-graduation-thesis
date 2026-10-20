import type { ReactNode } from 'react'
import { LeftSidebarRail } from './LeftSidebarRail'
import { LeftSidebar } from './LeftSidebar.tsx'
import { RightSidebar } from './RightSidebar'
import { RightSidebarRail } from './RightSidebarRail.tsx'
import { TopHeader } from './TopHeader'
import { useChat } from '@/context/ChatContext'

export function AppLayout({ children }: { children: ReactNode }) {
const { isLeftSidebarExpanded, rightSidebarTab } = useChat()

  return (
    <div className="h-full flex overflow-hidden select-none text-[15px]">
      {/* Left Sidebar */}
      {isLeftSidebarExpanded ? <LeftSidebar /> : <LeftSidebarRail />}

      {/* Main chat area */}
      <div className="flex-1 flex flex-col h-full bg-black relative overflow-hidden">
        <TopHeader />
        {children}
      </div>

      {/* Right Sidebar */}
      {rightSidebarTab === 'hidden' ? <RightSidebarRail /> : <RightSidebar />}    </div>
  )
}