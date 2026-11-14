package com.example.fileservice.services;

import com.example.fileservice.messaging.EmbedJobPublisher;
import com.example.fileservice.repositories.DocumentRepository;
import com.example.fileservice.repositories.DocumentRepository.StoredFile;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DocumentReconcilerTest {

    private final DocumentRepository documents = mock(DocumentRepository.class);
    private final FileStorageService storage = mock(FileStorageService.class);
    private final DocumentService documentService = mock(DocumentService.class);
    private final EmbedJobPublisher embedJobs = mock(EmbedJobPublisher.class);
    private final DocumentReconciler reconciler = new DocumentReconciler(documents, storage, documentService, embedJobs);

    private static StoredFile doc(long id, String objectName) {
        return new StoredFile() {
            public Long getId() { return id; }

            public String getStoredFileName() { return objectName; }
        };
    }

    @Test
    void deletesOnlyDocumentsWhoseObjectIsGoneAndPrunesTheRest() throws Exception {
        when(documents.findAllBy()).thenReturn(List.of(doc(1, "a.pdf"), doc(2, "b.pdf"), doc(3, "c.pdf")));
        when(storage.listObjectNames()).thenReturn(Set.of("a.pdf", "c.pdf"));

        reconciler.reconcile();

        verify(documentService).deleteDocument(2L);
        verify(documentService, never()).deleteDocument(1L);
        verify(documentService, never()).deleteDocument(3L);
        verify(embedJobs).publishPrune(List.of(1L, 3L));
    }

    @Test
    void aDocumentWhoseDeletionFailedStaysInThePruneKeepList() throws Exception {
        when(documents.findAllBy()).thenReturn(List.of(doc(1, "a.pdf"), doc(2, "b.pdf")));
        when(storage.listObjectNames()).thenReturn(Set.of("a.pdf"));
        doThrow(new StorageException("minio down", null)).when(documentService).deleteDocument(2L);

        reconciler.reconcile();

        verify(embedJobs).publishPrune(List.of(1L, 2L));
    }

    @Test
    void doesNothingWhenMinioCannotBeListed() throws Exception {
        when(documents.findAllBy()).thenReturn(List.of(doc(1, "a.pdf"), doc(2, "b.pdf")));
        when(storage.listObjectNames()).thenThrow(new StorageException("minio down", null));

        reconciler.reconcile();

        verify(documentService, never()).deleteDocument(anyLong());
        verify(embedJobs, never()).publishPrune(any());
    }

    @Test
    void doesNotWipeEverythingWhenEveryObjectIsMissing() throws Exception {
        when(documents.findAllBy()).thenReturn(List.of(doc(1, "a.pdf"), doc(2, "b.pdf")));
        when(storage.listObjectNames()).thenReturn(Set.of());

        reconciler.reconcile();

        verify(documentService, never()).deleteDocument(anyLong());
        verify(embedJobs, never()).publishPrune(any());
    }

    @Test
    void aSingleDocumentWhoseObjectIsGoneIsStillDeleted() throws Exception {
        when(documents.findAllBy()).thenReturn(List.of(doc(1, "a.pdf")));
        when(storage.listObjectNames()).thenReturn(Set.of());

        reconciler.reconcile();

        verify(documentService).deleteDocument(1L);
    }
}
