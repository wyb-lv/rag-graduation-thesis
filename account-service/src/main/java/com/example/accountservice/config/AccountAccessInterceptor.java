package com.example.accountservice.config;

import com.example.accountservice.controller.SelfService;
import com.example.accountservice.entity.UserRole;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.HttpStatus;
import org.springframework.web.method.HandlerMethod;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.servlet.HandlerInterceptor;

/**
 * /api/accounts/** chỉ cho ADMIN, trừ các API đánh dấu @SelfService (mọi người dùng đã đăng nhập).
 * Danh tính lấy từ header X-User-Role do api-gateway gắn sau khi auth-service xác thực token (đúng is_active + token_version),
 * nên service này PHẢI chỉ nhận request qua gateway: không publish cổng ra ngoài. auth-service không đi qua đây mà gọi gRPC.
 * Quyết định theo handler Spring đã khớp (không tự so khớp URL nên không lệch với routing); đường dẫn lạ (404) không có
 * @SelfService nên cũng bị chặn mặc định.
 */
public class AccountAccessInterceptor implements HandlerInterceptor {

    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        String role = request.getHeader("X-User-Role");
        if (role == null || role.isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Missing identity (call through the gateway)");
        }
        boolean selfService = handler instanceof HandlerMethod method && method.hasMethodAnnotation(SelfService.class);
        if (!selfService && !UserRole.ADMIN.name().equals(role)) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Admin only");
        }
        return true;
    }
}
