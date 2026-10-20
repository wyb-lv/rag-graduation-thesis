package com.example.fileservice.messaging;

import com.example.fileservice.repositories.DocumentRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.connection.CorrelationData;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.concurrent.TimeUnit;

import static com.example.fileservice.config.RabbitMQConfig.OCR_EXCHANGE;
import static com.example.fileservice.config.RabbitMQConfig.OCR_ROUTING_KEY;

@Slf4j
@Component
public class OcrJobPublisher {

    private final RabbitTemplate rabbitTemplate;
    private final DocumentRepository documentRepository;
    private final Duration confirmTimeout;

    public OcrJobPublisher(RabbitTemplate rabbitTemplate, DocumentRepository documentRepository,
                           @Value("${ocr.publish-confirm-timeout}") Duration confirmTimeout) {
        this.rabbitTemplate = rabbitTemplate;
        this.documentRepository = documentRepository;
        this.confirmTimeout = confirmTimeout;
    }

    /**
     * Gửi job và chờ broker xác nhận. Đánh dấu ocrQueuedAt trước khi gửi để mỗi lần xếp hàng chỉ một nơi gửi
     * (kể cả nhiều instance); gửi lỗi thì bỏ đánh dấu để {@link OcrJobRecovery} gửi lại.
     *
     * @return false nếu broker lỗi; true nếu đã gửi hoặc nơi khác đã gửi
     */
    public boolean publish(Long documentId) {
        if (documentRepository.markQueued(documentId, LocalDateTime.now()) == 0) {
            return true;
        }
        try {
            CorrelationData correlation = new CorrelationData(String.valueOf(documentId));
            rabbitTemplate.convertAndSend(OCR_EXCHANGE, OCR_ROUTING_KEY, new OcrJobMessage(documentId), correlation);

            CorrelationData.Confirm confirm = correlation.getFuture()
                    .get(confirmTimeout.toMillis(), TimeUnit.MILLISECONDS);
            if (!confirm.isAck()) {
                throw new IllegalStateException("Broker từ chối message: " + confirm.reason());
            }
            if (correlation.getReturned() != null) {
                throw new IllegalStateException("Message không được route vào queue nào");
            }
            return true;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            documentRepository.clearQueued(documentId);
            return false;
        } catch (Exception e) {
            log.warn("Chưa gửi được tài liệu {} vào hàng đợi OCR, sẽ thử lại sau: {}", documentId, e.toString());
            documentRepository.clearQueued(documentId);
            return false;
        }
    }
}
