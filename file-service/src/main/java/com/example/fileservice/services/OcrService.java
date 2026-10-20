package com.example.fileservice.services;

import com.example.fileservice.entities.Document;
import com.example.fileservice.entities.OcrStatus;
import com.example.fileservice.messaging.EmbedJobPublisher;
import com.example.fileservice.repositories.DocumentRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;

@Slf4j
@Service
@RequiredArgsConstructor
public class OcrService {

    private final DocumentRepository documentRepository;
    private final FileStorageService fileStorage;
    private final TextExtractionService textExtractionService;
    private final EmbedJobPublisher embedJobPublisher;

    public void process(Long documentId) {
        Document doc = documentRepository.findById(documentId).orElse(null);
        if (doc == null) return;
        // RabbitMQ giao ít nhất một lần nên có thể nhận trùng job đã xong
        if (doc.getOcrStatus() == OcrStatus.COMPLETED || doc.getOcrStatus() == OcrStatus.FAILED) {
            log.info("Bỏ qua job OCR trùng của tài liệu {} (đã {})", documentId, doc.getOcrStatus());
            return;
        }
        doc.setOcrStatus(OcrStatus.PROCESSING);
        doc.setOcrError(null);
        documentRepository.save(doc);

        long start = System.currentTimeMillis();
        String text = null;
        String error = null;
        Path tempFile = null;
        try {
            // Tesseract/PDFBox cần file thật trên disk; giữ đuôi file để nhận dạng định dạng
            tempFile = Files.createTempFile("ocr-" + documentId + "-", extensionOf(doc.getStoredFileName()));
            fileStorage.download(doc.getStoredFileName(), tempFile);
            text = textExtractionService.extract(tempFile);
        } catch (StorageException e) {
            // Lỗi tạm thời của MinIO: để listener retry thay vì đánh dấu FAILED
            throw e;
        } catch (Throwable e) {
            // Bắt cả Error: lỗi native của Tesseract qua JNA (UnsatisfiedLinkError, "Invalid memory access")
            // vẫn phải đánh dấu FAILED thay vì kẹt PROCESSING
            error = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();
            log.error("OCR thất bại cho tài liệu {}", documentId, e);
        } finally {
            deleteQuietly(tempFile);
        }

        // Tải lại vì tài liệu có thể đã bị xóa trong lúc OCR
        String finalText = text;
        String finalError = error;
        documentRepository.findById(documentId).ifPresent(d -> {
            d.setOcrStatus(finalError == null ? OcrStatus.COMPLETED : OcrStatus.FAILED);
            d.setExtractedText(finalText);
            d.setOcrError(finalError);
            d.setOcrCompletedAt(LocalDateTime.now());
            documentRepository.save(d);

            // Sau khi OCR thành công → publish embed job sang ai-service qua RabbitMQ
            if (finalError == null && finalText != null && !finalText.isBlank()) {
                embedJobPublisher.publish(d.getId(), d.getTitle(), null);
            }
        });
        log.info("OCR tài liệu {} xong trong {} ms", documentId, System.currentTimeMillis() - start);
    }

    public void markFailed(Long documentId, String reason) {
        documentRepository.findById(documentId).ifPresent(d -> {
            if (d.getOcrStatus() == OcrStatus.COMPLETED) return;
            d.setOcrStatus(OcrStatus.FAILED);
            d.setOcrError(reason);
            d.setOcrCompletedAt(LocalDateTime.now());
            documentRepository.save(d);
        });
    }

    private static String extensionOf(String fileName) {
        int dot = fileName.lastIndexOf('.');
        return dot < 0 ? "" : fileName.substring(dot);
    }

    private static void deleteQuietly(Path file) {
        if (file == null) return;
        try {
            Files.deleteIfExists(file);
        } catch (IOException e) {
            log.warn("Không xóa được file tạm {}", file, e);
        }
    }
}
