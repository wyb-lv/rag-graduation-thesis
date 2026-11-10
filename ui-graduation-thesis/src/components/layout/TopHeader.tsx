
import { useChat } from '@/context/ChatContext'

export function TopHeader() {
  const { activeView, setActiveView } = useChat()

  return (
    <header className="h-14 flex items-center justify-end px-5 gap-3.5 flex-shrink-0 z-10">
      {activeView !== 'call' && (
        <button
          onClick={() => setActiveView('call')}
          className="flex items-center gap-2 bg-[#1e3c66] hover:bg-[#274d82] text-white text-[13.5px] font-medium transition-colors py-1.5 px-3.5 rounded-full border border-white/15 cursor-pointer"
          type="button"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72c.127.96.361 1.903.7 2.81a2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0122 16.92z"
            />
          </svg>
          <span>Gọi trực tiếp</span>
        </button>
      )}
    </header>
  )
}
