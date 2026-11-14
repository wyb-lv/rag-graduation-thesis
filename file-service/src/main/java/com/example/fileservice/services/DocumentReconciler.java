package com.example.fileservice.services;

import com.example.fileservice.messaging.EmbedJobPublisher;
import com.example.fileservice.repositories.DocumentRepository;
import com.example.fileservice.repositories.DocumentRepository.StoredFile;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Ràng buộc: file trên MinIO là nguồn gốc, văn bản OCR (cột extracted_text), chunk và vector Qdrant chỉ là dữ liệu dẫn xuất,
 * không được sống lâu hơn file. Xóa qua API đã dọn ngay (DocumentService.deleteDocument); lớp này lo các đường còn lại:
 * file bị xóa thẳng trên MinIO (console, mc), và job DELETE gửi sang ai-service bị mất.
 * Mỗi lượt: tài liệu mất object thì xóa như xóa qua API, rồi gửi PRUNE để ai-service xóa vector không còn tài liệu nào.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class DocumentReconciler {

    private final DocumentRepository documents;
    private final FileStorageService storage;
    private final DocumentService documentService;
    private final EmbedJobPublisher embedJobs;

    @Scheduled(initialDelayString = "${reconcile.initial-delay}", fixedDelayString = "${reconcile.interval}")
    public void reconcile() {
        // Đọc DB trước rồi mới liệt kê MinIO: upload ghi object trước, dòng DB sau, nên mọi dòng đã đọc đều có object trong danh sách.
        // Làm ngược lại thì tài liệu vừa upload bị tưởng là mất file và bị xóa.
        List<StoredFile> all = documents.findAllBy();
        Set<String> inStorage;
        try {
            inStorage = storage.listObjectNames();
        } catch (RuntimeException e) {
            log.warn("Bỏ qua lượt đối chiếu vì không liệt kê được MinIO: {}", e.getMessage());
            return;
        }

        List<StoredFile> gone = all.stream().filter(d -> !inStorage.contains(d.getStoredFileName())).toList();
        if (all.size() > 1 && gone.size() == all.size()) {
            // Mất hết cùng lúc là dấu hiệu MinIO bị reset (volume mới, sai bucket), không phải người dùng xóa; xóa theo là mất sạch văn bản.
            log.error("Mọi {} tài liệu đều mất object trên MinIO: nghi MinIO bị reset, không xóa gì cả.", all.size());
            return;
        }

        Set<Long> removed = new HashSet<>();
        for (StoredFile d : gone) {
            try {
                documentService.deleteDocument(d.getId());
                removed.add(d.getId());
                log.info("Tài liệu {} ({}) không còn trên MinIO: đã xóa văn bản OCR và vector theo", d.getId(), d.getStoredFileName());
            } catch (Exception e) {
                log.warn("Chưa xóa được tài liệu {} đã mất file, lượt sau thử lại: {}", d.getId(), e.getMessage());
            }
        }

        embedJobs.publishPrune(all.stream().map(StoredFile::getId).filter(id -> !removed.contains(id)).toList());
    }
}
