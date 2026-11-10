package com.example.accountservice.service;

import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

/** REST trả 404; adapter gRPC ánh xạ theo kiểu này sang NOT_FOUND (xem GrpcErrors). */
public class AccountNotFoundException extends ResponseStatusException {
    public AccountNotFoundException() {
        super(HttpStatus.NOT_FOUND, "Account not found");
    }
}
