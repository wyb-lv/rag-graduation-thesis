package com.example.fileservice.config;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.HandlerInterceptor;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/**
 * /internal/** là API dành cho service khác (ai-service báo kết quả OCR), không dành cho người dùng nên không đi qua JWT của gateway.
 * Chỉ nhận request mang header X-Internal-Token đúng bằng INTERNAL_TOKEN dùng chung. Chưa cấu hình token thì từ chối tất cả
 * (đóng mặc định), nên quên đặt biến này không làm lộ API. So sánh theo thời gian không đổi để không dò được token.
 * Không dùng Spring Security: theo ranh giới của hệ thống, chỉ auth-service có nó.
 */
public class InternalAccessInterceptor implements HandlerInterceptor {

    public static final String HEADER = "X-Internal-Token";

    private final byte[] expected;

    public InternalAccessInterceptor(String token) {
        this.expected = token == null ? new byte[0] : token.getBytes(StandardCharsets.UTF_8);
    }

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        String given = request.getHeader(HEADER);
        if (expected.length == 0 || given == null
                || !MessageDigest.isEqual(expected, given.getBytes(StandardCharsets.UTF_8))) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Invalid or missing internal token");
        }
        return true;
    }
}
