package com.example.fileservice.services;

import com.example.fileservice.entities.Document;
import com.example.fileservice.entities.OcrStatus;
import com.example.fileservice.messaging.EmbedJobPublisher;
import com.example.fileservice.repositories.DocumentRepository;
import com.example.fileservice.services.OcrResultService.Outcome;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class OcrResultServiceTest {

    private final DocumentRepository repository = mock(DocumentRepository.class);
    private final EmbedJobPublisher embed = mock(EmbedJobPublisher.class);
    private final OcrResultService service = new OcrResultService(repository, embed);

    private Document document(OcrStatus status) {
        Document doc = new Document();
        doc.setId(7L);
        doc.setTitle("Quy chế");
        doc.setOcrStatus(status);
        doc.setOcrError("lỗi cũ");
        when(repository.findById(7L)).thenReturn(Optional.of(doc));
        return doc;
    }

    @Test
    void startMarksProcessingAndClearsOldError() {
        Document doc = document(OcrStatus.PENDING);
        assertEquals(Outcome.RECORDED, service.start(7L));
        assertEquals(OcrStatus.PROCESSING, doc.getOcrStatus());
        assertNull(doc.getOcrError());
        verify(repository).save(doc);
    }

    @Test
    void startOfAFinishedDocumentIsADuplicateJob() {
        for (OcrStatus done : new OcrStatus[]{OcrStatus.COMPLETED, OcrStatus.FAILED}) {
            Document doc = document(done);
            assertEquals(Outcome.DUPLICATE, service.start(7L));
            assertEquals(done, doc.getOcrStatus());
        }
        verify(repository, never()).save(any());
    }

    @Test
    void unknownDocumentIsNotFoundForEveryResult() {
        when(repository.findById(anyLong())).thenReturn(Optional.empty());
        assertEquals(Outcome.NOT_FOUND, service.start(1L));
        assertEquals(Outcome.NOT_FOUND, service.complete(1L, "x"));
        assertEquals(Outcome.NOT_FOUND, service.fail(1L, "x"));
    }

    @Test
    void completeStoresTextAndQueuesEmbedOnlyWhenThereIsText() {
        Document doc = document(OcrStatus.PROCESSING);
        assertEquals(Outcome.RECORDED, service.complete(7L, "Điều 1. Phạm vi"));
        assertEquals(OcrStatus.COMPLETED, doc.getOcrStatus());
        assertEquals("Điều 1. Phạm vi", doc.getExtractedText());
        assertNull(doc.getOcrError());
        assertNotNull(doc.getOcrCompletedAt());
        verify(embed).publish(7L, "Quy chế");

        document(OcrStatus.PROCESSING);
        service.complete(7L, "  ");
        verify(embed, times(1)).publish(eq(7L), any());
    }

    @Test
    void failRecordsReasonButNeverOverwritesACompletedDocument() {
        Document doc = document(OcrStatus.PROCESSING);
        service.fail(7L, "File hỏng");
        assertEquals(OcrStatus.FAILED, doc.getOcrStatus());
        assertEquals("File hỏng", doc.getOcrError());

        Document done = document(OcrStatus.COMPLETED);
        done.setOcrError(null);
        service.fail(7L, "job trễ");
        assertEquals(OcrStatus.COMPLETED, done.getOcrStatus());
        assertNull(done.getOcrError());
    }
}
