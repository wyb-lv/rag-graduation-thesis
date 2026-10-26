import type { DocumentItem } from '@/types'
import { authFetch } from '@/services/authService'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:9090'

async function request(path: string, init: RequestInit = {}): Promise<Response> {
  const res = await authFetch(`${API_BASE_URL}/api/documents${path}`, init)
  if (!res.ok) throw new Error(`Yêu cầu thất bại (${res.status})`)
  return res
}

function formatFileSize(bytes?: number | string): string {
  if (typeof bytes === 'string') return bytes
  if (typeof bytes !== 'number' || Number.isNaN(bytes)) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function mapBackendDoc(raw: Record<string, unknown>): DocumentItem {
  const status = (raw.ocrStatus as DocumentItem['ocrStatus']) || 'PENDING'
  const uploadDate = raw.uploadDate as string | undefined
  return {
    id: Number(raw.id),
    title: (raw.title as string) || (raw.originalFileName as string) || 'Tài liệu không tên',
    fileName: (raw.originalFileName as string) || (raw.fileName as string) || 'document.pdf',
    fileSize: formatFileSize(raw.fileSize as number | string | undefined),
    ocrStatus: status,
    createdAt: uploadDate
      ? new Date(uploadDate).toLocaleString('vi-VN')
      : (raw.createdAt as string) || '',
    isEmbedded: typeof raw.isEmbedded === 'boolean' ? raw.isEmbedded : status === 'COMPLETED',
    extractedText: raw.extractedText as string | undefined,
  }
}

export async function fetchDocuments(): Promise<DocumentItem[]> {
  const data = await (await request('')).json()
  return Array.isArray(data) ? data.map((item) => mapBackendDoc(item as Record<string, unknown>)) : []
}

export async function uploadDocument(file: File, title: string): Promise<DocumentItem> {
  const finalTitle = title.trim() || file.name
  const formData = new FormData()
  formData.append('file', file)
  formData.append('title', finalTitle)

  const data = await (await request('/upload', { method: 'POST', body: formData })).json()
  return {
    id: Number(data.id),
    title: finalTitle,
    fileName: file.name,
    fileSize: formatFileSize(file.size),
    ocrStatus: (data.ocrStatus as DocumentItem['ocrStatus']) || 'PENDING',
    createdAt: new Date().toLocaleString('vi-VN'),
    isEmbedded: false,
  }
}

export async function fetchDocumentText(id: number): Promise<string> {
  const data = await (await request(`/${id}/text`)).json()
  if (typeof data === 'string') return data
  if (data.text) return data.text
  if (data.ocrStatus === 'PENDING' || data.ocrStatus === 'PROCESSING') {
    return '⏳ Tài liệu đang trong hàng đợi xử lý Tesseract OCR. Vui lòng bấm "Làm mới danh sách" sau vài giây...'
  }
  if (data.ocrStatus === 'FAILED') {
    return `❌ Quét OCR thất bại: ${data.ocrError || 'Không thể đọc tệp tin'}`
  }
  return 'Tài liệu không có văn bản trích xuất.'
}

export async function retryOcr(id: number): Promise<void> {
  await request(`/${id}/ocr`, { method: 'POST' })
}

export async function embedDocument(id: number): Promise<void> {
  await request(`/${id}/embed`, { method: 'POST' })
}
