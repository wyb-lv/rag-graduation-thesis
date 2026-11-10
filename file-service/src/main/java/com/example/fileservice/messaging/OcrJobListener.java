package com.example.fileservice.messaging;

import com.example.fileservice.services.OcrResultService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

import static com.example.fileservice.config.RabbitMQConfig.OCR_DEAD_QUEUE;

/**
 * Job OCR được ai-service lấy từ hàng đợi {@code file-service.ocr.jobs} (file-service chỉ khai báo hàng đợi và gửi job).
 * Ở đây chỉ còn xử lý job bị bỏ cuộc sau nhiều lần giao (ai-service lỗi liên tục).
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class OcrJobListener {

    private final OcrResultService ocrResultService;

    // Job bị bỏ cuộc: đánh dấu FAILED để không kẹt PENDING/PROCESSING, người dùng chạy lại qua POST /ocr
    @RabbitListener(queues = OCR_DEAD_QUEUE, containerFactory = "deadLetterListenerContainerFactory")
    public void onDeadJob(OcrJobMessage job) {
        log.error("Job OCR của tài liệu {} bị chuyển vào dead-letter queue", job.documentId());
        ocrResultService.fail(job.documentId(), "Không xử lý được sau nhiều lần thử, vui lòng chạy lại OCR");
    }
}
