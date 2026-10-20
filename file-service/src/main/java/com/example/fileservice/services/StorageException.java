package com.example.fileservice.services;

/** Lỗi tạm thời của MinIO (khác NoSuchFileException): job OCR gặp lỗi này được thử lại, không đánh dấu FAILED. */
public class StorageException extends RuntimeException {

    public StorageException(String message, Throwable cause) {
        super(message, cause);
    }
}
