package com.hsu.gateway.docs;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClient;

import java.net.URI;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Cấp dữ liệu cho Swagger UI của gateway (/swagger-ui.html): danh sách tài liệu API và spec đã chỉnh của từng service.
 * Đặt gateway.docs.enabled=false (GATEWAY_DOCS_ENABLED) để tắt khi triển khai thật.
 */
@Slf4j
@RestController
@EnableConfigurationProperties(DocsProperties.class)
@ConditionalOnProperty(name = "gateway.docs.enabled", havingValue = "true", matchIfMissing = true)
public class OpenApiController {

    private static final ParameterizedTypeReference<LinkedHashMap<String, Object>> JSON_OBJECT = new ParameterizedTypeReference<>() {
    };

    private final DocsProperties properties;
    private final RestClient client;

    public OpenApiController(DocsProperties properties) {
        this.properties = properties;
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(Duration.ofSeconds(2));
        factory.setReadTimeout(Duration.ofSeconds(5));
        this.client = RestClient.builder().requestFactory(factory).build();
    }

    /** Địa chỉ quen thuộc của springdoc (/swagger-ui/index.html...) chuyển về trang của gateway. */
    @GetMapping({"/swagger-ui", "/swagger-ui/", "/swagger-ui/index.html"})
    public ResponseEntity<Void> swaggerUiAlias() {
        return ResponseEntity.status(HttpStatus.FOUND).location(URI.create("/swagger-ui.html")).build();
    }

    /** Swagger UI đọc cấu hình này (configUrl) nên thêm service mới chỉ cần khai báo trong application.yaml. */
    @GetMapping("/v3/api-docs/swagger-config")
    public Map<String, Object> swaggerConfig() {
        List<Map<String, String>> urls = properties.sources().stream()
                .map(s -> Map.of("name", s.title(), "url", "/v3/api-docs/" + s.name()))
                .toList();
        Map<String, Object> config = new LinkedHashMap<>();
        config.put("urls", urls);
        config.put("persistAuthorization", true); // nhập Bearer token một lần, dùng cho cả 4 tài liệu
        config.put("displayRequestDuration", true);
        config.put("docExpansion", "none");
        config.put("filter", true);
        return config;
    }

    @GetMapping("/v3/api-docs/{name}")
    public ResponseEntity<Map<String, Object>> spec(@PathVariable String name) {
        DocsProperties.Source source = properties.sources().stream().filter(s -> s.name().equals(name)).findFirst().orElse(null);
        if (source == null) return ResponseEntity.notFound().build();
        try {
            Map<String, Object> raw = client.get().uri(source.url()).retrieve().body(JSON_OBJECT);
            return ResponseEntity.ok(OpenApiAdapter.adapt(raw == null ? Map.of() : raw, source));
        } catch (RuntimeException e) {
            // Không đưa URL nội bộ ra ngoài; nguyên nhân chi tiết nằm trong log
            log.warn("Không lấy được tài liệu API của '{}' từ {}: {}", name, source.url(), e.toString());
            return ResponseEntity.status(502).body(Map.of(
                    "error", "Bad Gateway",
                    "message", "Không lấy được tài liệu API của '" + source.title() + "' (service chưa chạy hoặc chưa bật OpenAPI)."));
        }
    }
}
