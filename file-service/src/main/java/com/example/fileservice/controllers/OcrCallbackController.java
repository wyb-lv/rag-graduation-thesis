package com.example.fileservice.controllers;

import com.example.fileservice.entities.OcrStatus;
import com.example.fileservice.services.OcrResultService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * ai-service báo trạng thái / kết quả OCR về đây. Đường dẫn /internal/** không nằm trong route nào của api-gateway
 * (gateway chỉ chuyển /api/documents/**), chỉ gọi được từ trong mạng nội bộ.
 */
@RestController
@RequiredArgsConstructor
public class OcrCallbackController {

    public record OcrResult(OcrStatus status, String text, String error) {
    }

    private final OcrResultService ocrResultService;

    @PutMapping("/internal/documents/{id}/ocr")
    public ResponseEntity<Void> record(@PathVariable Long id, @RequestBody OcrResult result) {
        if (result.status() == null || result.status() == OcrStatus.PENDING) return ResponseEntity.badRequest().build();
        OcrResultService.Outcome outcome = switch (result.status()) {
            case PROCESSING -> ocrResultService.start(id);
            case COMPLETED -> ocrResultService.complete(id, result.text());
            default -> ocrResultService.fail(id, result.error());
        };
        return switch (outcome) {
            case RECORDED -> ResponseEntity.noContent().build();
            case NOT_FOUND -> ResponseEntity.notFound().build();
            case DUPLICATE -> ResponseEntity.status(HttpStatus.CONFLICT).build();
        };
    }
}
