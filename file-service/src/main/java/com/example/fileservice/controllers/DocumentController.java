package com.example.fileservice.controllers;

import com.example.fileservice.entities.Document;
import com.example.fileservice.entities.OcrStatus;
import com.example.fileservice.messaging.EmbedJobPublisher;
import com.example.fileservice.services.DocumentService;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequiredArgsConstructor
public class DocumentController {

    private final DocumentService documentService;
    private final EmbedJobPublisher embedJobPublisher;

    @GetMapping("/api/documents")
    public ResponseEntity<List<Document>> getAllDocuments() {
        return ResponseEntity.ok(documentService.getAllDocuments());
    }

    @PostMapping("/api/documents/upload")
    public ResponseEntity<Map<String, String>> uploadFile(@RequestParam("file") MultipartFile file,
                                                          @RequestParam("title") String title) {
        Map<String, String> response = new LinkedHashMap<>();
        try {
            Document doc = documentService.storeFile(file, title);
            response.put("status", "success");
            response.put("message", "Upload file thành công!");
            response.put("id", String.valueOf(doc.getId()));
            response.put("ocrStatus", doc.getOcrStatus().name());
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            response.put("status", "error");
            response.put("error", "Lỗi: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }

    @DeleteMapping("/api/documents/{id}")
    public ResponseEntity<Map<String, String>> deleteFile(@PathVariable Long id) {
        Map<String, String> response = new LinkedHashMap<>();
        try {
            documentService.deleteDocument(id);
            response.put("status", "success");
            response.put("message", "Xóa file thành công!");
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            response.put("status", "error");
            response.put("error", "Lỗi: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }

    @GetMapping("/api/documents/download/{id}")
    public ResponseEntity<Resource> downloadFile(@PathVariable Long id) {
        try {
            Document doc = documentService.getDocumentById(id);
            Resource resource = documentService.loadFileAsResource(doc.getStoredFileName());
            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(doc.getContentType()))
                    .contentLength(doc.getFileSize())
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + doc.getOriginalFileName() + "\"")
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
        Map<String, String> response = new LinkedHashMap<>();
        try {
            documentService.reprocessOcr(id);
            response.put("status", "success");
            response.put("message", "Đã đưa tài liệu vào hàng chờ OCR");
            return ResponseEntity.accepted().body(response);
        } catch (Exception e) {
            response.put("status", "error");
            response.put("error", "Lỗi: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Trigger embed thủ công cho một tài liệu đã OCR xong.
     * Dùng khi cần re-index vào Qdrant mà không cần chạy lại OCR.
     */
    @PostMapping("/api/documents/{id}/embed")
    public ResponseEntity<Map<String, String>> triggerEmbed(@PathVariable Long id) {
        Map<String, String> response = new LinkedHashMap<>();
        try {
            Document doc = documentService.getDocumentById(id);
            if (doc.getOcrStatus() != OcrStatus.COMPLETED) {
                response.put("status", "error");
                response.put("error", "Tài liệu chưa hoàn thành OCR (trạng thái: " + doc.getOcrStatus() + ")");
                return ResponseEntity.badRequest().body(response);
            }
            embedJobPublisher.publish(doc.getId(), doc.getTitle(), null);
            response.put("status", "success");
            response.put("message", "Đã gửi yêu cầu embed tài liệu vào hàng chờ.");
            return ResponseEntity.accepted().body(response);
        } catch (Exception e) {
            response.put("status", "error");
            response.put("error", "Lỗi: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
}