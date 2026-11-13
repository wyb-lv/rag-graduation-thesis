export interface User {
  name: string
  initials: string
  studentId: string
  avatarColor: string
}

export interface SourceCitation {
  id: string
  type: 'link' | 'document'
  title: string
  description?: string
  url?: string
  fileType?: string
  fileSize?: string
  pageCount?: number
  previewText?: string
  text?: string
  score?: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
  sources?: SourceCitation[]
  audioUrl?: string
  audioDuration?: number
  audioFileName?: string
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt?: string
}

export type ActiveView = 'chat' | 'ocr' | 'call' | 'accounts'

export type UserRoleType = 'ADMIN' | 'MANAGER' | 'EMPLOYEE'

export interface AccountItem {
  id: number
  username: string
  email: string
  departmentId: number
  departmentName?: string
  role: UserRoleType
  isActive: boolean
  createdAt?: string
  updatedAt?: string
}

export interface CreateAccountInput {
  username: string
  email: string
  password?: string
  departmentId: number
  role: UserRoleType
}

export interface UpdateAccountInput {
  email?: string
  departmentId?: number
  role?: UserRoleType
  isActive?: boolean
  password?: string
}

export type OcrStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED'

export interface DocumentItem {
  id: number
  title: string
  fileName?: string
  fileSize?: string
  ocrStatus: OcrStatus
  extractedText?: string
  createdAt?: string
  isEmbedded?: boolean
}
