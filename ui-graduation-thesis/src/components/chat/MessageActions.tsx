import { CopyIcon, ThumbsDownIcon, ShareIcon, RegenerateIcon, MoreDotsIcon } from '@/components/icons'

export function MessageActions() {
  return (
    <div className="flex items-center gap-3 pt-1 text-[#8f8f8f]">
      <button aria-label="Sao chép" className="hover:text-white transition-colors" type="button">
        <CopyIcon />
      </button>
      <button aria-label="Không thích" className="hover:text-white transition-colors" type="button">
        <ThumbsDownIcon />
      </button>
      <button aria-label="Chia sẻ" className="hover:text-white transition-colors" type="button">
        <ShareIcon />
      </button>
      <button aria-label="Tạo lại phản hồi" className="hover:text-white transition-colors" type="button">
        <RegenerateIcon />
      </button>
      <button aria-label="Tùy chọn khác" className="hover:text-white transition-colors" type="button">
        <MoreDotsIcon />
      </button>
    </div>
  )
}
