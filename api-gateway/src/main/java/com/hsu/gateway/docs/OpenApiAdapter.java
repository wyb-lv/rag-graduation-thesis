package com.hsu.gateway.docs;

import com.hsu.gateway.filter.JwtAuthFilter;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chỉnh spec OpenAPI gốc của một service cho khớp với những gì client thấy qua gateway, để "Try it out" trong Swagger UI chạy được:
 * server là chính gateway (URL tương đối), path có đúng tiền tố /api/..., bỏ path không có route (vd /internal/**), có Bearer token
 * và các path công khai của gateway (login, refresh...) không đòi token.
 */
final class OpenApiAdapter {

    private static final String BEARER = "bearerAuth";
    private static final Set<String> HTTP_METHODS = Set.of("get", "put", "post", "delete", "options", "head", "patch", "trace");

    private OpenApiAdapter() {
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> adapt(Map<String, Object> spec, DocsProperties.Source source) {
        Map<String, Object> out = new LinkedHashMap<>(spec);
        out.put("servers", List.of(Map.of("url", "/", "description", "api-gateway")));

        Map<String, Object> paths = new LinkedHashMap<>();
        Map<String, Object> original = (Map<String, Object>) spec.getOrDefault("paths", Map.of());
        original.forEach((path, item) -> {
            String full = source.pathPrefix() + path;
            if (source.include().stream().anyMatch(full::startsWith)) {
                paths.put(full, JwtAuthFilter.isPublic(full) ? withoutSecurity((Map<String, Object>) item) : item);
            }
        });
        out.put("paths", paths);

        Map<String, Object> components = new LinkedHashMap<>((Map<String, Object>) spec.getOrDefault("components", Map.of()));
        Map<String, Object> schemes = new LinkedHashMap<>((Map<String, Object>) components.getOrDefault("securitySchemes", Map.of()));
        schemes.putIfAbsent(BEARER, Map.of("type", "http", "scheme", "bearer", "bearerFormat", "JWT"));
        components.put("securitySchemes", schemes);
        out.put("components", components);
        out.putIfAbsent("security", List.of(Map.of(BEARER, List.of())));
        return out;
    }

    /** Path công khai: bỏ yêu cầu Bearer ở từng thao tác (security rỗng ghi đè security toàn cục). */
    private static Map<String, Object> withoutSecurity(Map<String, Object> pathItem) {
        Map<String, Object> copy = new LinkedHashMap<>(pathItem);
        copy.replaceAll((key, value) -> HTTP_METHODS.contains(key) && value instanceof Map<?, ?> op ? withEmptySecurity(op) : value);
        return copy;
    }

    private static Map<String, Object> withEmptySecurity(Map<?, ?> operation) {
        Map<String, Object> op = new LinkedHashMap<>();
        operation.forEach((k, v) -> op.put(String.valueOf(k), v));
        op.put("security", List.of());
        return op;
    }
}
