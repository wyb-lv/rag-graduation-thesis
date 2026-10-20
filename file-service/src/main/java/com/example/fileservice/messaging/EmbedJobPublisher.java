package com.example.fileservice.messaging;

import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import static com.example.fileservice.config.RabbitMQConfig.EMBED_EXCHANGE;
import static com.example.fileservice.config.RabbitMQConfig.EMBED_ROUTING_KEY;

/**
 * Gửi job embed sang ai-service qua RabbitMQ sau khi OCR hoàn tất.
 * <p>
 * Lỗi publish không ảnh hưởng đến trạng thái OCR: văn bản vẫn được lưu trong DB.
 * Admin có thể trigger lại bằng cách gọi POST /api/documents/{id}/embed.
 */
@Slf4j
@Component
public class EmbedJobPublisher {

    private final RabbitTemplate rabbitTemplate;

    @Value("${embed.enabled:true}")
    private boolean embedEnabled;

    public EmbedJobPublisher(RabbitTemplate rabbitTemplate) {
        this.rabbitTemplate = rabbitTemplate;
    }

    /**
     * Publish job embed sau khi OCR xong.
     *
     * @param documentId ID tài liệu
     * @param source     Tên tài liệu (title)
     * @param kho        Nhãn kho (có thể null)
     */
    public void publish(Long documentId, String source, String kho) {
        if (!embedEnabled) {
            log.debug("Embed bị tắt (embed.enabled=false), bỏ qua tài liệu {}", documentId);
            return;
        }
        try {
            EmbedJobMessage msg = new EmbedJobMessage(documentId, source, kho, null, null);
            rabbitTemplate.convertAndSend(EMBED_EXCHANGE, EMBED_ROUTING_KEY, msg);
            log.info("Đã gửi embed job cho tài liệu {}", documentId);
        } catch (Exception e) {
            log.warn("Không gửi được embed job cho tài liệu {}: {}. Có thể embed thủ công qua POST /api/documents/{}/embed",
                    documentId, e.getMessage(), documentId);
        }
    }

    /**
     * Publish thông báo xóa vector tài liệu sang ai-service khi tài liệu bị xóa.
     *
     * @param documentId ID tài liệu cần xóa vector
     */
    public void publishDelete(Long documentId) {
        if (!embedEnabled) {
            return;
        }
        try {
            EmbedJobMessage msg = new EmbedJobMessage(documentId, null, null, null, null, "DELETE");
            rabbitTemplate.convertAndSend(EMBED_EXCHANGE, EMBED_ROUTING_KEY, msg);
            log.info("Đã gửi delete embed job cho tài liệu {}", documentId);
        } catch (Exception e) {
            log.warn("Không gửi được delete embed job cho tài liệu {}: {}", documentId, e.getMessage());
        }
    }
}
