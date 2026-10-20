package com.example.fileservice.messaging;

/**
 * Message gửi sang ai-service để trigger embedding vào Qdrant.
 *
 * @param documentId   ID tài liệu trong file-service
 * @param source       Tên hiển thị của tài liệu (title)
 * @param kho          Nhãn kho (có thể null)
 * @param hieuLucTu    Ngày hiệu lực từ ISO-8601 (có thể null)
 * @param hieuLucDen   Ngày hiệu lực đến ISO-8601 (có thể null)
 * @param action       Hành động: "UPSERT" hoặc "DELETE"
 */
public record EmbedJobMessage(
        Long documentId,
        String source,
        String kho,
        String hieuLucTu,
        String hieuLucDen,
        String action
) {
    public EmbedJobMessage(Long documentId, String source, String kho, String hieuLucTu, String hieuLucDen) {
        this(documentId, source, kho, hieuLucTu, hieuLucDen, "UPSERT");
    }
}
