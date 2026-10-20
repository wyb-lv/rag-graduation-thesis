package com.example.fileservice.repositories;

import com.example.fileservice.entities.Document;
import com.example.fileservice.entities.OcrStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Collection;
import java.util.List;

public interface DocumentRepository extends JpaRepository<Document, Long> {

    @Query("select d.id from Document d where d.ocrStatus in :statuses and d.ocrQueuedAt is null order by d.id")
    List<Long> findIdsNotQueued(@Param("statuses") Collection<OcrStatus> statuses);

    // Update đúng một cột, không save cả entity, để không ghi đè trạng thái OCR do consumer đang sửa
    @Transactional
    @Modifying
    @Query("update Document d set d.ocrQueuedAt = :at where d.id = :id and d.ocrQueuedAt is null")
    int markQueued(@Param("id") Long id, @Param("at") LocalDateTime at);

    @Transactional
    @Modifying
    @Query("update Document d set d.ocrQueuedAt = null where d.id = :id")
    int clearQueued(@Param("id") Long id);
}
