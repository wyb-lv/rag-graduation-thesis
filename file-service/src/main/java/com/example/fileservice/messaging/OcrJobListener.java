package com.example.fileservice.messaging;

import com.example.fileservice.services.OcrService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

import static com.example.fileservice.config.RabbitMQConfig.OCR_DEAD_QUEUE;
import static com.example.fileservice.config.RabbitMQConfig.OCR_QUEUE;

@Slf4j
@Component
@RequiredArgsConstructor
public class OcrJobListener {

    private final OcrService ocrService;

    @RabbitListener(queues = OCR_QUEUE, containerFactory = "ocrListenerContainerFactory")
    public void onJob(OcrJobMessage job) {
        ocrService.process(job.documentId());
    }

    // Job bị bỏ cuộc: đánh dấu FAILED để không kẹt PENDING/PROCESSING, người dùng chạy lại qua POST /ocr
    @RabbitListener(queues = OCR_DEAD_QUEUE, containerFactory = "deadLetterListenerContainerFactory")
    public void onDeadJob(OcrJobMessage job) {
        log.error("Job OCR của tài liệu {} bị chuyển vào dead-letter queue", job.documentId());
        ocrService.markFailed(job.documentId(), "Không xử lý được sau nhiều lần thử, vui lòng chạy lại OCR");
    }
}
