package com.example.fileservice.entities;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.*;
import lombok.Data;

import java.time.LocalDateTime;

@Entity
@Table(name = "Documents")
@Data
public class Document {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(length = 255)
    private String title;
    private String originalFileName;
    private String storedFileName;
    private String contentType;
    private Long fileSize;
    private LocalDateTime uploadDate;

    @Enumerated(EnumType.STRING)
    @Column(length = 20)
    private OcrStatus ocrStatus;
    // Không trả về trong danh sách tài liệu, lấy qua GET /api/documents/{id}/text
    @JsonIgnore
    @Column(columnDefinition = "TEXT")
    private String extractedText;
    @Column(columnDefinition = "TEXT")
    private String ocrError;
    private LocalDateTime ocrCompletedAt;
    // null = job OCR chưa được gửi vào RabbitMQ
    @JsonIgnore
    private LocalDateTime ocrQueuedAt;
}
