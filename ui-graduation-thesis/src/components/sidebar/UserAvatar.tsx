interface UserAvatarProps {
  initials: string
  color: string
  size?: 'sm' | 'md'
}

export function UserAvatar({ initials, color, size = 'md' }: UserAvatarProps) {
  const sizeClasses = size === 'sm' ? 'w-8 h-8 text-xs' : 'w-8 h-8 text-xs'
  return (
    <div
      className={`${sizeClasses} rounded-full flex items-center justify-center font-bold tracking-tight text-white flex-shrink-0`}
      style={{ backgroundColor: color }}
    >
      {initials}
    </div>
  )
}
