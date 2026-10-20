package com.example.fileservice.messaging;

// Chỉ mang id, không đẩy nội dung file qua broker
public record OcrJobMessage(Long documentId) {
}
