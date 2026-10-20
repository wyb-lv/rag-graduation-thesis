import { SidebarRightIcon, BookIcon, EyeIcon, DocumentIcon, ExternalLinkIcon } from '@/components/icons'
import { useChat } from '@/context/ChatContext'

export function RightSidebar() {
  const {
    rightSidebarTab,
    setRightSidebarTab,
    toggleRightSidebar,
    sources,
    selectedSourceIndex,
    setSelectedSourceIndex,
  } = useChat()

  const selectedSource = sources[selectedSourceIndex] || sources[0]

  const handleSelectSource = (index: number) => {
    setSelectedSourceIndex(index)
    setRightSidebarTab('detail')
  }

  return (
    <aside className="w-[340px] h-full flex flex-col justify-between bg-black text-[#ECECEC] z-20 flex-shrink-0 border-l border-white/5 select-none">
      {/* Header Row 1: Cao chuẩn h-14 (56px) - Trùng baseline với TopHeader */}
      <div className="h-14 flex items-center justify-between pl-4 pr-2.5 border-b border-white/5 flex-shrink-0">
        <span className="text-[14px] font-semibold text-white tracking-tight">
          Tài liệu tham chiếu
        </span>
        <button
          aria-label="Đóng thanh tài liệu"
          className="p-1.5 text-white/90 hover:text-white rounded-lg hover:bg-white/5 transition-colors"
          type="button"
          onClick={toggleRightSidebar}
        >
          <SidebarRightIcon className="w-5 h-5" />
        </button>
      </div>

      {/* Header Row 2: Tab switcher */}
      <div className="p-3 border-b border-white/5 flex-shrink-0">
        <div className="flex bg-[#212121] p-1 rounded-xl gap-1 text-[13px]">
          <button
            className={`flex-1 py-1.5 px-2 rounded-lg font-medium flex items-center justify-center gap-1.5 transition-colors ${
              rightSidebarTab === 'sources'
                ? 'bg-black/60 text-white shadow-sm'
                : 'text-[#8e8e8e] hover:text-white'
            }`}
            type="button"
            onClick={() => setRightSidebarTab('sources')}
          >
            <BookIcon className={`w-3.5 h-3.5 ${rightSidebarTab === 'sources' ? 'text-white' : ''}`} />
            <span>Nguồn trích</span>
            <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-white/10 text-white/90 ml-0.5">
              {sources.length}
            </span>
          </button>
          <button
            className={`flex-1 py-1.5 px-2 rounded-lg font-medium flex items-center justify-center gap-1.5 transition-colors ${
              rightSidebarTab === 'detail'
                ? 'bg-black/60 text-white shadow-sm'
                : 'text-[#8e8e8e] hover:text-white'
            }`}
            type="button"
            onClick={() => setRightSidebarTab('detail')}
          >
            <EyeIcon className={`w-3.5 h-3.5 ${rightSidebarTab === 'detail' ? 'text-white' : ''}`} />
            <span>Xem chi tiết</span>
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col min-h-0 overflow-y-auto no-scrollbar p-3.5 space-y-4">
        {rightSidebarTab === 'sources' && (
          <div className="space-y-2">
            {sources.map((source, idx) => (
              <div
                key={source.id}
                className="p-2.5 rounded-xl bg-[#212121]/60 hover:bg-[#212121] border border-white/5 hover:border-white/15 transition-all cursor-pointer group"
                onClick={() => handleSelectSource(idx)}
              >
                <div className="flex items-start gap-2.5">
                  <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 text-[#55b9ff] bg-[#1e3c66]/40 border border-[#55b9ff]/20">
                    <DocumentIcon className="w-4 h-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-1">
                      <h4 className="text-[13px] font-medium text-white truncate group-hover:text-[#55b9ff] transition-colors">
                        {source.title}
                      </h4>
                      {source.score !== undefined && (
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-[#55b9ff] font-mono flex-shrink-0">
                          {(source.score * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-[#8e8e8e] mt-0.5 line-clamp-2">
                      {source.text || source.description || 'Xem nội dung đối chiếu'}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {rightSidebarTab === 'detail' && selectedSource && (
          <div className="space-y-4 text-left select-text">
            <div className="rounded-xl bg-[#212121]/70 border border-white/10 overflow-hidden shadow-sm flex flex-col">
              <div className="p-3 bg-[#1e232d]/80 border-b border-white/5 flex items-start justify-between gap-2">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 text-[#55b9ff] bg-[#1e3c66]/50 border border-[#55b9ff]/20">
                    <DocumentIcon className="w-4 h-4" />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <h4 className="text-[13px] font-semibold text-white truncate">
                        {selectedSource.title}
                      </h4>
                      <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-[#1e3c66] text-[#55b9ff] font-semibold flex-shrink-0">
                        {selectedSource.fileType || 'PDF'}
                      </span>
                    </div>
                    <p className="text-[11px] text-[#8e8e8e] truncate">
                      {selectedSource.description || 'Tài liệu trường Đại học Hoa Sen'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Excerpt Body */}
              <div className="p-3.5 bg-black/50 space-y-3 font-sans text-xs leading-relaxed text-[#b4b4b4]">
                <div className="border-b border-white/5 pb-2">
                  <h5 className="text-[12px] font-semibold text-[#55b9ff] tracking-tight">
                    Đoạn văn bản trích dẫn đối chiếu
                  </h5>
                </div>
                <p className="text-[#e2e2e2] bg-white/5 p-2.5 rounded-lg border border-white/5 leading-relaxed">
                  {selectedSource.text || selectedSource.previewText || 'Không có nội dung xem trước.'}
                </p>
              </div>

              {/* Footer */}
              <div className="p-2.5 bg-[#171717] border-t border-white/5 flex items-center justify-between text-[11px] text-[#8e8e8e]">
                <span className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#10A37F]" />
                  Đã xác thực bởi HSU RAG
                </span>
                <button
                  type="button"
                  className="text-[#55b9ff] hover:underline flex items-center gap-1 font-medium"
                  onClick={() => setRightSidebarTab('sources')}
                >
                  <span>Xem nguồn khác</span>
                  <ExternalLinkIcon className="w-3 h-3" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bottom bar */}
      <div className="p-3 border-t border-white/5 flex items-center justify-between text-xs text-[#8e8e8e] bg-black">
        <span className="truncate text-[11px]">
          {sources.length}&nbsp;tài liệu liên kết chat
        </span>
      </div>
    </aside>
  )
}