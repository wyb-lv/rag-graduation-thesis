import { useState, useEffect, type FormEvent } from 'react'
import type { DocumentItem, OcrStatus } from '@/types'
import {
  fetchDocuments,
  uploadDocument,
  fetchDocumentText,
  retryOcr,
  embedDocument,
} from '@/services/documentService'

export function OcrDemoView() {
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedDocId, setSelectedDocId] = useState<number | null>(null)
  const [selectedDocText, setSelectedDocText] = useState<string>('')
  const [loadingText, setLoadingText] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fail = (e: unknown) => setError(e instanceof Error ? e.message : 'Đã có lỗi xảy ra')

  // Upload state
  const [title, setTitle] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [copySuccess, setCopySuccess] = useState(false)

  const loadDocuments = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchDocuments()
      setDocuments(data)
      if (selectedDocId === null && data.length > 0) {
        setSelectedDocId(data[0].id)
      }
    } catch (e) {
      fail(e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDocuments()
  }, [])

  // Khi chọn document -> nạp text
  useEffect(() => {
    if (selectedDocId === null) return
    let isCancelled = false
    setLoadingText(true)
    fetchDocumentText(selectedDocId)
      .then((txt) => {
        if (!isCancelled) setSelectedDocText(txt)
      })
      .catch((e) => {
        if (isCancelled) return
        setSelectedDocText('')
        fail(e)
      })
      .finally(() => {
        if (!isCancelled) setLoadingText(false)
      })
    return () => {
      isCancelled = true
    }
  }, [selectedDocId])

  const handleUpload = async (e: FormEvent) => {
    e.preventDefault()
    if (!file) return

    setUploading(true)
    try {
      await uploadDocument(file, title)
      setTitle('')
      setFile(null)
      // reset file input
      const fileInput = document.getElementById('ocr-file-input') as HTMLInputElement
      if (fileInput) fileInput.value = ''
      await loadDocuments()
    } catch (e) {
      fail(e)
    } finally {
      setUploading(false)
    }
  }

  const handleRetry = async (id: number) => {
    try {
      await retryOcr(id)
    } catch (e) {
      return fail(e)
    }
    await loadDocuments()
  }

  const handleEmbed = async (id: number) => {
    try {
      await embedDocument(id)
    } catch (e) {
      return fail(e)
    }
    await loadDocuments()
  }

  const handleCopyText = () => {
    if (!selectedDocText) return
    navigator.clipboard.writeText(selectedDocText)
    setCopySuccess(true)
    setTimeout(() => setCopySuccess(false), 2000)
  }

  const selectedDoc = documents.find((d) => d.id === selectedDocId)

  const getStatusBadge = (status: OcrStatus) => {
    switch (status) {
      case 'COMPLETED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            Đã OCR xong
          </span>
        )
      case 'PROCESSING':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            Đang quét OCR...
          </span>
        )
      case 'PENDING':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-yellow-500/10 text-yellow-400 border border-yellow-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-yellow-400" />
            Hàng đợi
          </span>
        )
      case 'FAILED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
            Lỗi xử lý
          </span>
        )
    }
  }

  return (
    <div className="flex-1 flex flex-col h-full bg-[#0e0e0e] text-white overflow-hidden">
      {/* Top Bar / Header */}
      <div className="h-14 px-6 border-b border-white/10 flex items-center justify-between bg-black flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#55b9ff]/10 text-[#55b9ff] flex items-center justify-center border border-[#55b9ff]/20 font-bold text-sm">
            OCR
          </div>
          <div>
            <h1 className="text-sm font-semibold text-white tracking-wide">
              Quản lý Tài liệu & Bóc tách Chữ (OCR Demo)
            </h1>
            <p className="text-[11px] text-[#8e8e8e]">
              MinIO Storage &bull; RabbitMQ Pipeline &bull; Tesseract OCR &bull; Qdrant Vector
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={loadDocuments}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-[#b4b4b4] hover:text-white border border-white/10 transition-colors"
        >
          <svg
            className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          <span>Làm mới danh sách</span>
        </button>
      </div>

      {error && (
        <div className="mx-4 lg:mx-6 mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex-shrink-0">
          {error}
        </div>
      )}

      {/* Main Grid: 2 columns */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 p-4 lg:p-6 overflow-hidden min-h-0">
        {/* Left Column (5 cols): Upload form & Documents List */}
        <div className="lg:col-span-5 flex flex-col gap-4 overflow-hidden min-h-0">
          {/* Card: Upload New File */}
          <div className="bg-[#181818] border border-white/10 rounded-xl p-4 flex-shrink-0">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#b4b4b4] mb-3 flex items-center gap-2">
              <svg className="w-4 h-4 text-[#55b9ff]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              Tải tài liệu mới lên MinIO
            </h2>

            <form onSubmit={handleUpload} className="space-y-3">
              <div>
                <input
                  type="text"
                  placeholder="Tiêu đề tài liệu (VD: Quy chế thi đua 2026)"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  className="w-full bg-[#212121] border border-white/10 rounded-lg px-3 py-2 text-xs text-white placeholder-[#666] focus:outline-none focus:border-[#55b9ff]"
                />
              </div>

              <div>
                <input
                  id="ocr-file-input"
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg"
                  required
                  onChange={(e) => setFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-[#8e8e8e] file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-white/10 file:text-white hover:file:bg-white/20 cursor-pointer"
                />
              </div>

              <button
                type="submit"
                disabled={uploading || !file}
                className="w-full py-2 px-3 bg-[#55b9ff] hover:bg-[#43a5e8] text-black font-semibold rounded-lg text-xs transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {uploading ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-black border-t-transparent rounded-full animate-spin" />
                    <span>Đang nạp lên MinIO & Hàng đợi...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>Tải lên & Khởi chạy Tesseract OCR</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* List of Documents */}
          <div className="flex-1 bg-[#181818] border border-white/10 rounded-xl p-4 flex flex-col overflow-hidden min-h-0">
            <div className="flex items-center justify-between mb-3 flex-shrink-0">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-[#b4b4b4]">
                Tài liệu đã lưu trữ ({documents.length})
              </h2>
              <span className="text-[11px] text-[#666]">Bấm để xem chữ bóc tách</span>
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 pr-1 no-scrollbar">
              {documents.map((doc) => {
                const isSelected = doc.id === selectedDocId
                return (
                  <div
                    key={doc.id}
                    onClick={() => setSelectedDocId(doc.id)}
                    className={`p-3 rounded-lg border transition-all cursor-pointer ${
                      isSelected
                        ? 'bg-[#262626] border-[#55b9ff]/60 shadow-sm shadow-[#55b9ff]/10'
                        : 'bg-[#212121] border-white/5 hover:border-white/15 hover:bg-[#252525]'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2 mb-1.5">
                      <h3 className="text-xs font-medium text-white truncate flex-1">
                        {doc.title}
                      </h3>
                      {getStatusBadge(doc.ocrStatus)}
                    </div>

                    <div className="flex items-center justify-between text-[11px] text-[#8e8e8e]">
                      <span className="truncate">{doc.fileName || 'file.pdf'}</span>
                      <span>{doc.fileSize || '1 MB'}</span>
                    </div>

                    {/* Action Bar inside item */}
                    <div className="mt-2.5 pt-2 border-t border-white/5 flex items-center justify-between">
                      <span className="text-[10px] text-[#666]">{doc.createdAt}</span>

                      <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
                        {doc.ocrStatus === 'COMPLETED' && (
                          <button
                            type="button"
                            onClick={() => handleEmbed(doc.id)}
                            className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                              doc.isEmbedded
                                ? 'bg-[#55b9ff]/20 text-[#55b9ff] border border-[#55b9ff]/30'
                                : 'bg-white/10 hover:bg-white/20 text-white'
                            }`}
                          >
                            {doc.isEmbedded ? '✓ Đã nạp AI' : '+ Nạp Vector AI'}
                          </button>
                        )}

                        {doc.ocrStatus === 'FAILED' && (
                          <button
                            type="button"
                            onClick={() => handleRetry(doc.id)}
                            className="px-2 py-0.5 rounded text-[10px] bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
                          >
                            Thử lại
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Right Column (7 cols): Extracted Text Preview Reader */}
        <div className="lg:col-span-7 bg-[#181818] border border-white/10 rounded-xl p-5 flex flex-col overflow-hidden min-h-0">
          {/* Header of reader */}
          <div className="flex items-center justify-between pb-3 border-b border-white/10 flex-shrink-0">
            <div>
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-[#55b9ff]">
                  Văn bản trích xuất OCR
                </span>
                {selectedDoc && getStatusBadge(selectedDoc.ocrStatus)}
              </div>
              <h2 className="text-sm font-medium text-white truncate max-w-md mt-0.5">
                {selectedDoc?.title || 'Chưa chọn tài liệu'}
              </h2>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleCopyText}
                disabled={!selectedDocText}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-white border border-white/10 transition-colors disabled:opacity-40"
              >
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span>{copySuccess ? 'Đã sao chép!' : 'Sao chép chữ'}</span>
              </button>
            </div>
          </div>

          {/* Text Content Area */}
          <div className="flex-1 mt-4 bg-[#121212] border border-white/5 rounded-lg p-4 overflow-y-auto font-mono text-xs leading-relaxed text-[#d4d4d4] selection:bg-[#55b9ff]/30">
            {loadingText ? (
              <div className="h-full flex items-center justify-center text-[#666] gap-2">
                <span className="w-4 h-4 border-2 border-[#55b9ff] border-t-transparent rounded-full animate-spin" />
                <span>Đang đọc dữ liệu chữ OCR...</span>
              </div>
            ) : selectedDocText ? (
              <pre className="whitespace-pre-wrap font-sans text-sm text-[#e0e0e0] leading-relaxed">
                {selectedDocText}
              </pre>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-[#666]">
                <p>Chưa có dữ liệu trích xuất cho tài liệu này.</p>
              </div>
            )}
          </div>

          {/* Footer note for Demo */}
          <div className="pt-3 mt-3 border-t border-white/5 flex items-center justify-between text-[11px] text-[#666] flex-shrink-0">
            <span>Độ phân giải: 300 DPI &bull; Tesseract v5.x</span>
            <span>Sau khi OCR hoàn tất, văn bản sẵn sàng để AI RAG tìm kiếm và trích dẫn</span>
          </div>
        </div>
      </div>
    </div>
  )
}
