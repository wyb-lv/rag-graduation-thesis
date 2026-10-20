package com.example.fileservice.services;

import io.minio.BucketExistsArgs;
import io.minio.GetObjectArgs;
import io.minio.MakeBucketArgs;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import io.minio.RemoveObjectArgs;
import io.minio.errors.ErrorResponseException;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Set;

/** Lưu file upload trên MinIO; tên object là Document.storedFileName. */
@Slf4j
@Service
public class FileStorageService {

    private static final Set<String> NOT_FOUND_CODES = Set.of("NoSuchKey", "NoSuchObject");

    private final MinioClient minio;
    private final String bucket;
    private volatile boolean bucketReady;

    public FileStorageService(MinioClient minio, @Value("${minio.bucket}") String bucket) {
        this.minio = minio;
        this.bucket = bucket;
    }

    // Không bắt buộc MinIO phải sẵn sàng lúc khởi động; store() sẽ thử lại việc tạo bucket
    @EventListener(ApplicationReadyEvent.class)
    public void prepareBucketOnStartup() {
        try {
            ensureBucket();
        } catch (Exception e) {
            log.warn("Chưa kết nối được MinIO để chuẩn bị bucket '{}', sẽ thử lại khi upload: {}", bucket, e.toString());
        }
    }

    public void store(String objectName, InputStream in, long size, String contentType) {
        try {
            ensureBucket();
            minio.putObject(PutObjectArgs.builder()
                    .bucket(bucket)
                    .object(objectName)
                    .stream(in, size, -1L)
                    .contentType(contentType == null || contentType.isBlank() ? "application/octet-stream" : contentType)
                    .build());
        } catch (Exception e) {
            throw new StorageException("Không lưu được file lên MinIO: " + objectName, e);
        }
    }

    public InputStream open(String objectName) throws NoSuchFileException {
        try {
            return minio.getObject(GetObjectArgs.builder().bucket(bucket).object(objectName).build());
        } catch (ErrorResponseException e) {
            if (NOT_FOUND_CODES.contains(e.errorResponse().code())) {
                throw new NoSuchFileException(objectName, null, "không còn trên MinIO");
            }
            throw new StorageException("Không đọc được file từ MinIO: " + objectName, e);
        } catch (Exception e) {
            throw new StorageException("Không đọc được file từ MinIO: " + objectName, e);
        }
    }

    public void download(String objectName, Path target) throws NoSuchFileException {
        try (InputStream in = open(objectName)) {
            Files.copy(in, target, StandardCopyOption.REPLACE_EXISTING);
        } catch (NoSuchFileException e) {
            throw e;
        } catch (IOException e) {
            throw new StorageException("Không tải được file từ MinIO: " + objectName, e);
        }
    }

    // Object không tồn tại vẫn coi là thành công
    public void delete(String objectName) {
        try {
            minio.removeObject(RemoveObjectArgs.builder().bucket(bucket).object(objectName).build());
        } catch (Exception e) {
            throw new StorageException("Không xóa được file trên MinIO: " + objectName, e);
        }
    }

    private void ensureBucket() throws Exception {
        if (bucketReady) return;
        synchronized (this) {
            if (bucketReady) return;
            if (!minio.bucketExists(BucketExistsArgs.builder().bucket(bucket).build())) {
                minio.makeBucket(MakeBucketArgs.builder().bucket(bucket).build());
                log.info("Đã tạo bucket MinIO '{}'", bucket);
            }
            bucketReady = true;
        }
    }
}
