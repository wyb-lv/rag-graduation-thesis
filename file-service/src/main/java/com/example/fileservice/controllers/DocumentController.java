package com.example.fileservice.controllers;

import com.example.fileservice.entities.Document;
import com.example.fileservice.entities.OcrStatus;
import com.example.fileservice.messaging.EmbedJobPublisher;
import com.example.fileservice.services.DocumentService;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.Resource;
import org.springframework.http.ContentDisposition;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequiredArgsConstructor
public class DocumentController {

    private final DocumentService documentService;
    private final EmbedJobPublisher embedJobPublisher;

    private static Map<String, String> success(String message) {
        Map<String, String> response = new LinkedHashMap<>();
        response.put("status", "success");
        response.put("message", message);
        return response;
    }

    private static ResponseEntity<Map<String, String>> failure(String error) {
        Map<String, String> response = new LinkedHashMap<>();
        response.put("status", "error");
        response.put("error", error);
        return ResponseEntity.badRequest().body(response);
    }

    private static ResponseEntity<Map<String, String>> failure(Exception e) {
        return failure("Lỗi: " + e.getMessage());
    }

    @GetMapping("/api/documents")
    public ResponseEntity<List<Document>> getAllDocuments() {
        return ResponseEntity.ok(documentService.getAllDocuments());
    }

    @PostMapping("/api/documents/upload")
    public ResponseEntity<Map<String, String>> uploadFile(@RequestParam("file") MultipartFile file,
                                                          @RequestParam("title") String title) {
        try {
            Document doc = documentService.storeFile(file, title);
            Map<String, String> response = success("Upload file thành công!");
            response.put("id", String.valueOf(doc.getId()));
            response.put("ocrStatus", doc.getOcrStatus().name());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return failure(e);
        }
    }

    @DeleteMapping("/api/documents/{id}")
    public ResponseEntity<Map<String, String>> deleteFile(@PathVariable Long id) {
        try {
            documentService.deleteDocument(id);
            return ResponseEntity.ok(success("Xóa file thành công!"));
        } catch (Exception e) {
            return failure(e);
        }
    }

    // Cũng là "link" để ai-service tải file về OCR
    @GetMapping("/api/documents/download/{id}")
    public ResponseEntity<Resource> downloadFile(@PathVariable Long id) {
        try {
            Document doc = documentService.getDocumentById(id);
            Resource resource = documentService.loadFileAsResource(doc.getStoredFileName());
            // Tên có dấu tiếng Việt phải mã hóa (RFC 5987), nếu không Tomcat bỏ mất cả header; content_type NULL (tài liệu cũ) thì dùng octet-stream
            MediaType type = doc.getContentType() == null ? MediaType.APPLICATION_OCTET_STREAM : MediaType.parseMediaType(doc.getContentType());
            return ResponseEntity.ok()
                    .contentType(type)
                    .contentLength(doc.getFileSize())
                    .header(HttpHeaders.CONTENT_DISPOSITION,
                            ContentDisposition.attachment().filename(doc.getOriginalFileName(), StandardCharsets.UTF_8).build().toString())
                    .body(resource);
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }

    @GetMapping("/api/documents/{id}/text")
    public ResponseEntity<Map<String, Object>> getDocumentText(@PathVariable Long id) {
        try {
            Document doc = documentService.getDocumentById(id);
            Map<String, Object> response = new LinkedHashMap<>();
            response.put("id", doc.getId());
            response.put("ocrStatus", doc.getOcrStatus());
            response.put("ocrError", doc.getOcrError());
            response.put("ocrCompletedAt", doc.getOcrCompletedAt());
            response.put("text", doc.getExtractedText());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.notFound().build();
        }
    }

    @PostMapping("/api/documents/{id}/ocr")
    public ResponseEntity<Map<String, String>> reprocessOcr(@PathVariable Long id) {
        try {
            documentService.reprocessOcr(id);
            return ResponseEntity.accepted().body(success("Đã đưa tài liệu vào hàng chờ OCR"));
        } catch (Exception e) {
            return failure(e);
        }
    }

    /**
     * Trigger embed thủ công cho một tài liệu đã OCR xong.
     * Dùng khi cần re-index vào Qdrant mà không cần chạy lại OCR.
     */
    @PostMapping("/api/documents/{id}/embed")
    public ResponseEntity<Map<String, String>> triggerEmbed(@PathVariable Long id) {
        try {
            Document doc = documentService.getDocumentById(id);
            if (doc.getOcrStatus() != OcrStatus.COMPLETED) {
                return failure("Tài liệu chưa hoàn thành OCR (trạng thái: " + doc.getOcrStatus() + ")");
            }
            embedJobPublisher.publish(doc.getId(), doc.getTitle());
            return ResponseEntity.accepted().body(success("Đã gửi yêu cầu embed tài liệu vào hàng chờ."));
        } catch (Exception e) {
            return failure(e);
        }
    }
}
