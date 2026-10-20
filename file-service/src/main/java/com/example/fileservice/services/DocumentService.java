package com.example.fileservice.services;

import com.example.fileservice.entities.Document;
import com.example.fileservice.entities.OcrStatus;
import com.example.fileservice.messaging.EmbedJobPublisher;
import com.example.fileservice.messaging.OcrJobPublisher;
import com.example.fileservice.repositories.DocumentRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

@Slf4j
@Service
public class DocumentService {

    private final DocumentRepository documentRepository;
    private final OcrJobPublisher ocrJobPublisher;
    private final FileStorageService fileStorage;
    private final EmbedJobPublisher embedJobPublisher;

    public DocumentService(DocumentRepository documentRepository, OcrJobPublisher ocrJobPublisher,
                           FileStorageService fileStorage, EmbedJobPublisher embedJobPublisher) {
        this.documentRepository = documentRepository;
        this.ocrJobPublisher = ocrJobPublisher;
        this.fileStorage = fileStorage;
        this.embedJobPublisher = embedJobPublisher;
    }

    public Document storeFile(MultipartFile file, String title) throws Exception {
        String originalFileName = StringUtils.cleanPath(Objects.requireNonNull(file.getOriginalFilename()));

        if (!TextExtractionService.isSupported(originalFileName)) {
            throw new Exception("Chỉ hỗ trợ định dạng PDF, DOCX và ảnh (PNG, JPG, TIFF, BMP)!");
        }

        String storedFileName = UUID.randomUUID().toString() + "_" + originalFileName;
        try (InputStream in = file.getInputStream()) {
            fileStorage.store(storedFileName, in, file.getSize(), file.getContentType());
        }

        Document doc = new Document();
        doc.setTitle(title.isEmpty() ? originalFileName : title);
        doc.setOriginalFileName(originalFileName);
        doc.setStoredFileName(storedFileName);
        doc.setContentType(file.getContentType());
        doc.setFileSize(file.getSize());
        doc.setUploadDate(LocalDateTime.now());
        doc.setOcrStatus(OcrStatus.PENDING);
        try {
            documentRepository.save(doc);
        } catch (RuntimeException e) {
            // Tránh object mồ côi trên MinIO
            try {
                fileStorage.delete(storedFileName);
            } catch (RuntimeException cleanupError) {
                log.warn("Không dọn được object mồ côi {}", storedFileName, cleanupError);
            }
            throw e;
        }

        // RabbitMQ lỗi thì tài liệu vẫn PENDING và OcrJobRecovery gửi lại sau, upload vẫn thành công
        ocrJobPublisher.publish(doc.getId());
        return doc;
    }

    public void reprocessOcr(Long id) throws Exception {
        Document doc = getDocumentById(id);
        if (doc.getOcrStatus() == OcrStatus.PENDING || doc.getOcrStatus() == OcrStatus.PROCESSING) {
            throw new Exception("Tài liệu đang được xử lý OCR");
        }
        doc.setOcrStatus(OcrStatus.PENDING);
        doc.setOcrQueuedAt(null);
        documentRepository.save(doc);
        ocrJobPublisher.publish(doc.getId());
    }

    public List<Document> getAllDocuments() {
        return documentRepository.findAll();
    }

    public Resource loadFileAsResource(String storedFileName) throws Exception {
        return new InputStreamResource(fileStorage.open(storedFileName));
    }

    public void deleteDocument(Long id) throws Exception {
        Document doc = documentRepository.findById(id).orElseThrow(() -> new Exception("Tài liệu không tồn tại"));
        // Xóa object trước: xóa lặp lại an toàn nên bước sau lỗi thì gọi lại được
        fileStorage.delete(doc.getStoredFileName());
        documentRepository.delete(doc);
        // Báo cho ai-service xóa vector tương ứng trong Qdrant
        embedJobPublisher.publishDelete(id);
    }

    public Document getDocumentById(Long id) throws Exception {
        return documentRepository.findById(id).orElseThrow(() -> new Exception("Tài liệu không tồn tại"));
    }
}
