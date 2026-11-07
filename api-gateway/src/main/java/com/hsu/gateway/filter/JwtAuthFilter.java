package com.hsu.gateway.filter;

import com.example.authservice.grpc.TokenValidationRequest;
import com.example.authservice.grpc.TokenValidationResponse;
import com.example.authservice.grpc.TokenValidationServiceGrpc.TokenValidationServiceBlockingStub;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletRequestWrapper;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.*;

@Slf4j
@Component
@Order(3) // Chạy sau RequestLoggingFilter (@Order(2)) và RateLimitFilter (@Order(1))
public class JwtAuthFilter extends OncePerRequestFilter {

    @GrpcClient("auth-service")
    private TokenValidationServiceBlockingStub tokenValidationStub;

    // Danh sách các URL công khai không cần kiểm tra Token
    private static final List<String> PUBLIC_PATHS = List.of(
            "/api/auth/login",
            "/api/auth/refresh",
            "/api/auth/logout",
            "/api/ai/health",
            "/health",
            "/actuator"
    );

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        String path = request.getRequestURI();

        // Bỏ qua các path công khai hoặc request OPTIONS (preflight CORS)
        if ("OPTIONS".equalsIgnoreCase(request.getMethod()) ||
                PUBLIC_PATHS.stream().anyMatch(path::startsWith)) {
            chain.doFilter(request, response);
            return;
        }

        // Kiểm tra Header Authorization
        String authHeader = request.getHeader("Authorization");
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            rejectUnauthorized(response, "Miss or wrong Authorization Header (Bearer <token>) Header format");
            return;
        }

        String token = authHeader.substring(7).trim();

        // Gọi gRPC sang auth-service
        try {
            TokenValidationResponse validationResponse = tokenValidationStub.validateToken(
                    TokenValidationRequest.newBuilder().setToken(token).build()
            );

            if (!validationResponse.getIsValid()) {
                rejectUnauthorized(response, validationResponse.getErrorMessage());
                return;
            }

            // Token hợp lệ -> Chèn thêm Header danh tính (X-User-*) để các service phía sau sử dụng
            HeaderMapRequestWrapper requestWrapper = new HeaderMapRequestWrapper(request);
            requestWrapper.addHeader("X-User-Id", String.valueOf(validationResponse.getUserId()));
            requestWrapper.addHeader("X-User-Name", validationResponse.getUsername());
            requestWrapper.addHeader("X-User-Role", validationResponse.getRole());
            requestWrapper.addHeader("X-Department-Id", String.valueOf(validationResponse.getDepartmentId()));

            chain.doFilter(requestWrapper, response);

        } catch (Exception e) {
            log.error("Lỗi khi kết nối gRPC tới auth-service: {}", e.getMessage());
            response.setStatus(HttpServletResponse.SC_SERVICE_UNAVAILABLE);
            response.setContentType("application/json;charset=UTF-8");
            response.getWriter().write("{\"error\":\"Service Unavailable\",\"message\":\"Dịch vụ xác thực không khả dụng, vui lòng thử lại sau.\"}");
        }
    }

    private void rejectUnauthorized(HttpServletResponse response, String message) throws IOException {
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        response.setContentType("application/json;charset=UTF-8");
        String safeMsg = escapeJson(message != null ? message : "Unauthorized");
        response.getWriter().write(String.format("{\"error\":\"Unauthorized\",\"message\":\"%s\"}", safeMsg));
    }

    private static String escapeJson(String input) {
        if (input == null) return "";
        StringBuilder sb = new StringBuilder();
        for (char c : input.toCharArray()) {
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < ' ') {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        return sb.toString();
    }

    // Lớp bọc request để thêm Header tùy biến chuyển tiếp cho downstream service
    private static class HeaderMapRequestWrapper extends HttpServletRequestWrapper {
        // Dùng CASE_INSENSITIVE_ORDER để không phân biệt hoa/thường và chống client giả mạo header X-User-*
        private final Map<String, String> customHeaders = new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
        public HeaderMapRequestWrapper(HttpServletRequest request) {
            super(request);
        }
        public void addHeader(String name, String value) {
            customHeaders.put(name, value);
        }
        @Override
        public String getHeader(String name) {
            String val = customHeaders.get(name);
            return (val != null) ? val : super.getHeader(name);
        }
        @Override
        public Enumeration<String> getHeaders(String name) {
            String val = customHeaders.get(name);
            if (val != null) {
                return Collections.enumeration(List.of(val));
            }
            return super.getHeaders(name);
        }
        @Override
        public Enumeration<String> getHeaderNames() {
            Set<String> set = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
            Enumeration<String> e = super.getHeaderNames();
            while (e.hasMoreElements()) {
                set.add(e.nextElement());
            }
            set.addAll(customHeaders.keySet());
            return Collections.enumeration(set);
        }
    }
}