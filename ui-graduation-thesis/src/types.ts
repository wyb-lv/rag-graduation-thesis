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
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt?: string
}

export type ActiveView = 'chat' | 'ocr'

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
