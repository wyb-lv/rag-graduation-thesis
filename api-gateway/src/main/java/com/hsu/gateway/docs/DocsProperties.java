package com.hsu.gateway.docs;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.List;

/**
 * Các service có tài liệu OpenAPI được gom vào Swagger UI của gateway (xem gateway.docs.sources trong application.yaml).
 */
@ConfigurationProperties(prefix = "gateway.docs")
public record DocsProperties(List<Source> sources) {

    public DocsProperties {
        sources = sources == null ? List.of() : sources;
    }

    /**
     * @param name       tên trong URL: /v3/api-docs/{name}
     * @param title      tên hiển thị trong ô chọn của Swagger UI
     * @param url        nơi lấy spec gốc của service (đi thẳng tới service, không qua route của gateway)
     * @param pathPrefix tiền tố thêm vào mọi path của spec gốc (service nào được gateway StripPrefix thì spec không có tiền tố)
     * @param include    chỉ giữ các path (sau khi thêm tiền tố) bắt đầu bằng một trong các giá trị này, tức các path gateway có route
     */
    public record Source(String name, String title, String url, String pathPrefix, List<String> include) {

        public Source {
            pathPrefix = pathPrefix == null ? "" : pathPrefix;
            include = include == null ? List.of() : include;
        }
    }
}
