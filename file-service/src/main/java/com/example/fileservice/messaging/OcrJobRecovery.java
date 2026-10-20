package com.example.fileservice.messaging;

import com.example.fileservice.entities.OcrStatus;
import com.example.fileservice.repositories.DocumentRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.List;

/** Gửi lại tài liệu chưa vào được hàng đợi OCR (RabbitMQ lỗi lúc upload, hoặc bản ghi cũ trước khi có RabbitMQ). */
@Slf4j
@Component
@EnableScheduling
@RequiredArgsConstructor
public class OcrJobRecovery {

    private final DocumentRepository documentRepository;
    private final OcrJobPublisher publisher;

    @Scheduled(initialDelayString = "${ocr.recovery-initial-delay}", fixedDelayString = "${ocr.recovery-interval}")
    public void republishUnqueued() {
        List<Long> ids = documentRepository.findIdsNotQueued(List.of(OcrStatus.PENDING, OcrStatus.PROCESSING));
        for (Long id : ids) {
            // Broker còn lỗi thì dừng, đợi lượt sau
            if (!publisher.publish(id)) return;
            log.info("Đã đưa lại tài liệu {} vào hàng đợi OCR", id);
        }
    }
}
