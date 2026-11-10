package com.hsu.gateway.docs;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.json.JsonParserFactory;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Chạy cả gateway thật (đủ filter) với các service giả trả spec giống thật: springdoc (có servers, security, path nội bộ) và FastAPI
 * (không có servers, không có security, path không có tiền tố). Mọi request không kèm token.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class SwaggerUiTest {

    static final AtomicBoolean AI_UP = new AtomicBoolean(true);
    static final HttpServer AUTH = stub("/v3/api-docs", """
            {"openapi":"3.1.0","info":{"title":"Auth","version":"1"},"paths":{
              "/api/auth/login":{"post":{"summary":"login","responses":{"200":{"description":"ok"}}}},
              "/api/auth/refresh":{"post":{"responses":{"200":{"description":"ok"}}}}}}""", null);
    static final HttpServer ACCOUNTS = stub("/v3/api-docs", """
            {"openapi":"3.1.0","info":{"title":"Account","version":"1"},"servers":[{"url":"http://localhost:9090"}],
             "security":[{"bearerAuth":[]}],
             "paths":{"/api/accounts":{"get":{"responses":{"200":{"description":"ok"}}}}},
             "components":{"securitySchemes":{"bearerAuth":{"type":"http","scheme":"bearer","bearerFormat":"JWT"}}}}""", null);
    static final HttpServer DOCUMENTS = stub("/v3/api-docs", """
            {"openapi":"3.1.0","info":{"title":"File","version":"1"},"paths":{
              "/api/documents":{"get":{"responses":{"200":{"description":"ok"}}}},
              "/internal/documents/{id}/ocr":{"put":{"responses":{"200":{"description":"ok"}}}}}}""", null);
    static final HttpServer AI = stub("/openapi.json", """
            {"openapi":"3.1.0","info":{"title":"RAG QA API","version":"0.1.0"},"paths":{
              "/ask":{"post":{"responses":{"200":{"description":"ok"}}}},
              "/health":{"get":{"responses":{"200":{"description":"ok"}}}}}}""", AI_UP);

    @DynamicPropertySource
    static void services(DynamicPropertyRegistry registry) {
        registry.add("AUTH_SERVICE_URL", () -> url(AUTH));
        registry.add("ACCOUNT_SERVICE_URL", () -> url(ACCOUNTS));
        registry.add("FILE_SERVICE_URL", () -> url(DOCUMENTS));
        registry.add("AI_SERVICE_URL", () -> url(AI));
    }

    @Value("${local.server.port}") int port;
    final HttpClient http = HttpClient.newHttpClient();

    @AfterAll
    static void stop() {
        for (HttpServer s : List.of(AUTH, ACCOUNTS, DOCUMENTS, AI)) s.stop(0);
    }

    // ── Không bị chặn khi chưa có token ───────────────────────────────────────────────

    @Test
    void theUiPageAndItsAssetsAreReachableWithoutAToken() throws Exception {
        for (String path : List.of("/swagger-ui.html", "/webjars/swagger-ui/swagger-ui-bundle.js", "/webjars/swagger-ui/swagger-ui.css",
                "/webjars/swagger-ui/swagger-ui-standalone-preset.js")) {
            assertEquals(200, get(path).statusCode(), path);
        }
        assertTrue(get("/swagger-ui.html").body().contains("/v3/api-docs/swagger-config"));
    }

    @Test
    void familiarSwaggerUiAddressesRedirectToTheGatewayPage() throws Exception {
        for (String path : List.of("/swagger-ui", "/swagger-ui/", "/swagger-ui/index.html")) {
            HttpResponse<String> res = get(path);
            assertEquals(302, res.statusCode(), path);
            assertEquals("/swagger-ui.html", res.headers().firstValue("Location").orElse(""), path);
        }
    }

    @Test
    void swaggerConfigListsEveryService() throws Exception {
        HttpResponse<String> res = get("/v3/api-docs/swagger-config");
        assertEquals(200, res.statusCode());
        List<?> urls = (List<?>) json(res).get("urls");
        assertEquals(List.of("/v3/api-docs/auth", "/v3/api-docs/accounts", "/v3/api-docs/documents", "/v3/api-docs/ai"),
                urls.stream().map(u -> ((Map<?, ?>) u).get("url")).toList());
    }

    @Test
    void protectedApisStillRequireAToken() throws Exception {
        assertEquals(401, get("/api/accounts").statusCode());
        assertEquals(401, get("/api/documents").statusCode());
    }

    // ── Spec đã chỉnh để Try it out chạy qua gateway ─────────────────────────────────

    @Test
    void aiSpecGetsTheGatewayPrefixServerAndBearerAuth() throws Exception {
        Map<String, Object> spec = json(get("/v3/api-docs/ai"));
        Map<?, ?> paths = (Map<?, ?>) spec.get("paths");
        assertEquals(List.of("/api/ai/ask", "/api/ai/health"), List.copyOf(paths.keySet()));
        assertEquals("/", ((Map<?, ?>) ((List<?>) spec.get("servers")).get(0)).get("url"));
        Map<?, ?> schemes = (Map<?, ?>) ((Map<?, ?>) spec.get("components")).get("securitySchemes");
        assertTrue(schemes.containsKey("bearerAuth"));
        assertEquals(List.of(Map.of("bearerAuth", List.of())), spec.get("security"));
    }

    @Test
    void publicOperationsDoNotAskForAToken() throws Exception {
        Map<?, ?> ai = (Map<?, ?>) json(get("/v3/api-docs/ai")).get("paths");
        assertEquals(List.of(), ((Map<?, ?>) ((Map<?, ?>) ai.get("/api/ai/health")).get("get")).get("security"));
        assertFalse(((Map<?, ?>) ((Map<?, ?>) ai.get("/api/ai/ask")).get("post")).containsKey("security"));

        Map<?, ?> auth = (Map<?, ?>) json(get("/v3/api-docs/auth")).get("paths");
        assertEquals(List.of(), ((Map<?, ?>) ((Map<?, ?>) auth.get("/api/auth/login")).get("post")).get("security"));
    }

    @Test
    void springdocServerPointingAtLocalhostIsReplacedByTheGatewayItself() throws Exception {
        Map<String, Object> spec = json(get("/v3/api-docs/accounts"));
        assertEquals("/", ((Map<?, ?>) ((List<?>) spec.get("servers")).get(0)).get("url"));
        assertEquals(List.of("/api/accounts"), List.copyOf(((Map<?, ?>) spec.get("paths")).keySet()));
    }

    @Test
    void pathsWithoutAGatewayRouteAreDropped() throws Exception {
        Map<?, ?> paths = (Map<?, ?>) json(get("/v3/api-docs/documents")).get("paths");
        assertEquals(List.of("/api/documents"), List.copyOf(paths.keySet()));
    }

    // ── Lỗi ───────────────────────────────────────────────────────────────────────

    @Test
    void unknownDocumentIs404() throws Exception {
        assertEquals(404, get("/v3/api-docs/nope").statusCode());
    }

    @Test
    void aServiceThatIsDownGives502ForItsOwnDocumentOnly() throws Exception {
        AI_UP.set(false);
        try {
            HttpResponse<String> res = get("/v3/api-docs/ai");
            assertEquals(502, res.statusCode());
            assertFalse(res.body().contains("localhost"), "không lộ URL nội bộ");
            assertEquals(200, get("/v3/api-docs/auth").statusCode());
        } finally {
            AI_UP.set(true);
        }
    }

    // ── helper ────────────────────────────────────────────────────────────────────

    private HttpResponse<String> get(String path) throws Exception {
        return http.send(HttpRequest.newBuilder(URI.create("http://localhost:" + port + path)).GET().build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private static Map<String, Object> json(HttpResponse<String> res) {
        return JsonParserFactory.getJsonParser().parseMap(res.body());
    }

    private static String url(HttpServer server) {
        return "http://localhost:" + server.getAddress().getPort();
    }

    /** Service giả: trả `body` ở `path` (503 khi `up` là false). */
    private static HttpServer stub(String path, String body, AtomicBoolean up) {
        try {
            HttpServer server = HttpServer.create(new InetSocketAddress("localhost", 0), 0);
            server.createContext(path, exchange -> {
                boolean ok = up == null || up.get();
                byte[] bytes = (ok ? body : "{}").getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().add("Content-Type", "application/json");
                exchange.sendResponseHeaders(ok ? 200 : 503, bytes.length);
                exchange.getResponseBody().write(bytes);
                exchange.close();
            });
            server.start();
            return server;
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }
}
