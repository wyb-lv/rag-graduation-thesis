package com.example.fileservice.services;

import com.example.fileservice.entities.Document;
import com.example.fileservice.entities.OcrStatus;
import com.example.fileservice.messaging.EmbedJobPublisher;
import com.example.fileservice.repositories.DocumentRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.Optional;

/**
 * Ghi nhận kết quả OCR do ai-service báo về. file-service không còn tự OCR: nó chỉ lưu file, đưa job vào hàng đợi,
 * ai-service lấy job, tải file qua link tải của file-service, OCR rồi gọi lại để lưu trạng thái + văn bản.
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class OcrResultService {

    public enum Outcome { RECORDED, NOT_FOUND, DUPLICATE }

    private final DocumentRepository documentRepository;
    private final EmbedJobPublisher embedJobPublisher;

    /** ai-service nhận job. RabbitMQ giao ít nhất một lần nên job trùng của tài liệu đã xong trả DUPLICATE để ai-service bỏ qua. */
    public Outcome start(Long documentId) {
        Optional<Document> found = documentRepository.findById(documentId);
        if (found.isEmpty()) return Outcome.NOT_FOUND;
        Document doc = found.get();
        if (doc.getOcrStatus() == OcrStatus.COMPLETED || doc.getOcrStatus() == OcrStatus.FAILED) {
            log.info("Bỏ qua job OCR trùng của tài liệu {} (đã {})", documentId, doc.getOcrStatus());
            return Outcome.DUPLICATE;
        }
        doc.setOcrStatus(OcrStatus.PROCESSING);
        doc.setOcrError(null);
        documentRepository.save(doc);
        return Outcome.RECORDED;
    }

    /** OCR xong: lưu văn bản, rồi báo ai-service embed vào Qdrant (nếu có chữ). */
    public Outcome complete(Long documentId, String text) {
        // Tải lại vì tài liệu có thể đã bị xóa trong lúc OCR
        Optional<Document> found = documentRepository.findById(documentId);
        if (found.isEmpty()) return Outcome.NOT_FOUND;
        Document doc = found.get();
        doc.setOcrStatus(OcrStatus.COMPLETED);
        doc.setExtractedText(text);
        doc.setOcrError(null);
        doc.setOcrCompletedAt(LocalDateTime.now());
        documentRepository.save(doc);
        if (text != null && !text.isBlank()) {
            embedJobPublisher.publish(doc.getId(), doc.getTitle());
        }
        return Outcome.RECORDED;
    }

    /** OCR lỗi vĩnh viễn (file hỏng...) hoặc job bị chuyển vào dead-letter queue. Không ghi đè tài liệu đã COMPLETED. */
    public Outcome fail(Long documentId, String reason) {
        Optional<Document> found = documentRepository.findById(documentId);
        if (found.isEmpty()) return Outcome.NOT_FOUND;
        Document doc = found.get();
        if (doc.getOcrStatus() == OcrStatus.COMPLETED) return Outcome.RECORDED;
        doc.setOcrStatus(OcrStatus.FAILED);
        doc.setOcrError(reason);
        doc.setOcrCompletedAt(LocalDateTime.now());
        documentRepository.save(doc);
        return Outcome.RECORDED;
    }
}
