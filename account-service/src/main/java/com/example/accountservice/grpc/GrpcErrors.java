package com.example.accountservice.grpc;

import com.example.accountservice.service.AccountNotFoundException;
import io.grpc.Status;
import net.devh.boot.grpc.server.advice.GrpcAdvice;
import net.devh.boot.grpc.server.advice.GrpcExceptionHandler;

/** Lỗi khác không khai báo ở đây thì gRPC trả UNKNOWN, auth-service coi là account-service không dùng được (503). */
@GrpcAdvice
public class GrpcErrors {
    @GrpcExceptionHandler(AccountNotFoundException.class)
    public Status notFound(AccountNotFoundException e) {
        return Status.NOT_FOUND.withDescription(e.getReason());
    }
}
