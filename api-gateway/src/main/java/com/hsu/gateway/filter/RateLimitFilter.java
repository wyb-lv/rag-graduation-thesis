package com.hsu.gateway.filter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * In-memory token bucket rate limiter per IP address.
 * <p>
 * Áp dụng giới hạn khác nhau cho các nhóm route:
 * - /api/ai/** : rate limit thấp hơn (tốn tài nguyên GPU)
 * - /api/documents/** : rate limit cao hơn
 */
@Slf4j
@Component
@Order(1)
public class RateLimitFilter extends OncePerRequestFilter {

    @Value("${gateway.rate-limit.ai-service-rps:10}")
    private int aiRps;

    @Value("${gateway.rate-limit.file-service-rps:20}")
    private int fileRps;

    @Value("${gateway.rate-limit.auth-service-rps:5}")
    private int authRps;

    @Value("${gateway.rate-limit.ai-service-burst:20}")
    private int aiBurst;

    @Value("${gateway.rate-limit.file-service-burst:40}")
    private int fileBurst;

    @Value("${gateway.rate-limit.auth-service-burst:5}")
    private int authBurst;

    // key: "ip:route-group" → token bucket
    private final Map<String, TokenBucket> buckets = new ConcurrentHashMap<>();

    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        String ip = getClientIp(request);
        String path = request.getRequestURI();

//        boolean isAiRoute = path.startsWith("/api/ai/");
//        int rps = isAiRoute ? aiRps : fileRps;
//        int burst = isAiRoute ? aiBurst : fileBurst;
//        String group = isAiRoute ? "ai" : "file";

        // Phân loại nhóm route và áp dụng ngưỡng tương ứng
        int rps;
        int burst;
        String group;
        if (path.startsWith("/api/auth/")) {
            rps = authRps;
            burst = authBurst;
            group = "auth";
        } else if (path.startsWith("/api/ai/")) {
            rps = aiRps;
            burst = aiBurst;
            group = "ai";
        } else {
            rps = fileRps;
            burst = fileBurst;
            group = "file";
        }
        // Tạo hoặc lấy bucket độc lập cho từng cặp (IP + Nhóm route)
        TokenBucket bucket = buckets.computeIfAbsent(ip + ":" + group,
                k -> new TokenBucket(burst, rps));

        if (!bucket.tryAcquire()) {
            log.warn("Rate limit exceeded for IP={} path={}", ip, path);
            response.setStatus(429);
            response.setContentType("application/json;charset=UTF-8");
            response.getWriter().write("{\"error\":\"Too Many Requests\",\"message\":\"Vui lòng thử lại sau.\"}");
            return;
        }

        response.setHeader("X-RateLimit-Remaining", String.valueOf((int) bucket.tokens));
        chain.doFilter(request, response);
    }

    private String getClientIp(HttpServletRequest request) {
        String forwarded = request.getHeader("X-Forwarded-For");
        if (forwarded != null && !forwarded.isBlank()) {
            return forwarded.split(",")[0].trim();
        }
        return request.getRemoteAddr();
    }

    /** Đơn giản token bucket: refill tuyến tính theo thời gian. */
    static class TokenBucket {
        private final int capacity;
        private final double refillRate;   // tokens / ms
        private volatile double tokens;
        private volatile long lastRefillMs;

        TokenBucket(int capacity, int rps) {
            this.capacity = capacity;
            this.refillRate = rps / 1000.0;
            this.tokens = capacity;
            this.lastRefillMs = Instant.now().toEpochMilli();
        }

        synchronized boolean tryAcquire() {
            long now = Instant.now().toEpochMilli();
            long elapsed = now - lastRefillMs;
            tokens = Math.min(capacity, tokens + elapsed * refillRate);
            lastRefillMs = now;
            if (tokens >= 1) {
                tokens--;
                return true;
            }
            return false;
        }
    }
}
