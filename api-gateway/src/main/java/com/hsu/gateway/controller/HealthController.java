package com.hsu.gateway.controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Health check tổng hợp: kiểm tra cả gateway lẫn trạng thái của các downstream service.
 */
@Slf4j
@RestController
public class HealthController {

    private final RestClient restClient = RestClient.builder().build();

    @Value("${AUTH_SERVICE_URL:http://localhost:8081}")
    private String authServiceUrl;

    @Value("${FILE_SERVICE_URL:http://localhost:8080}")
    private String fileServiceUrl;

    @Value("${AI_SERVICE_URL:http://localhost:8000}")
    private String aiServiceUrl;

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> status = new LinkedHashMap<>();
        status.put("gateway", "up");
        status.put("fileService", checkService(fileServiceUrl + "/actuator/health"));
        status.put("authService", checkService(authServiceUrl + "/actuator/health"));
        status.put("aiService", checkService(aiServiceUrl + "/health"));
        return ResponseEntity.ok(status);
    }

    private String checkService(String url) {
        try {
            restClient.get().uri(url).retrieve().toBodilessEntity();
            return "up";
        } catch (Exception e) {
            log.warn("Health check thất bại cho {}: {}", url, e.getMessage());
            return "down";
        }
    }
}
