package com.hsu.gateway.filter;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RoleAccessPolicyTest {

    @ParameterizedTest
    @ValueSource(strings = {
            "/api/documents", "/api/documents/upload", "/api/documents/7/ocr", "/api/documents/download/7",
            "/api/ai/embed", "/api/ai/embed/7", "/api/ai/ask/stream",
            "/api/accounts", "/api/accounts/7", "/api/accounts/7/reset-password", "/api/accounts/me"})
    void adminCanUseEverything(String path) {
        assertTrue(RoleAccessPolicy.permits("ADMIN", path));
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "/api/ai/ask", "/api/ai/ask/stream", "/api/ai/ask/voice/stream", "/api/ai/stt", "/api/ai/tts",
            "/api/ai/call/greeting", "/api/ai/call/turn",
            "/api/ai/conversations", "/api/ai/conversations/3f2a9c1e-77b0-4d1e-9a61-0a1b2c3d4e5f",
            "/api/accounts/me", "/api/accounts/me/change-password"})
    void userCanAskTheAssistantAndManageOwnAccount(String path) {
        assertTrue(RoleAccessPolicy.permits("USER", path), path);
    }

    @ParameterizedTest
    @ValueSource(strings = {
            // quản lý tài liệu và embed
            "/api/documents", "/api/documents/upload", "/api/documents/7", "/api/documents/7/ocr", "/api/documents/download/7",
            "/api/ai/embed", "/api/ai/embed/7",
            // quản lý tài khoản của người khác
            "/api/accounts", "/api/accounts/7", "/api/accounts/7/status", "/api/accounts/7/reset-password",
            // đường dẫn lạ
            "/api/unknown", "/internal/documents/7/ocr", "/actuator/gateway/routes", "/",
            // lách đường dẫn: dot-segment, mã hóa %, tham số ;, // thừa, viết hoa, dấu / cuối
            "/api/ai/ask/../embed", "/api/ai/ask/%2e%2e/embed", "/api/ai/ask%2f..%2fembed", "/api/ai/ask;x=1",
            "//api/documents", "/api/ai/conversations/../../documents", "/API/DOCUMENTS", "/api/ai/ask/",
            "/api/accounts/me/", "/api/accounts/me/../7", "/api/ai/call/greeting/x"})
    void userIsDeniedEverythingElse(String path) {
        assertFalse(RoleAccessPolicy.permits("USER", path), path);
    }

    @Test
    void unknownOrMissingRoleIsTreatedLikeUserNotAdmin() {
        for (String role : new String[]{null, "", " ", "admin", "Admin", "ADMIN ", "MANAGER", "ROOT"}) {
            assertFalse(RoleAccessPolicy.permits(role, "/api/documents"), String.valueOf(role));
            assertTrue(RoleAccessPolicy.permits(role, "/api/ai/ask/stream"), String.valueOf(role));
        }
    }

    @Test
    void missingPathIsDeniedForNonAdmin() {
        assertFalse(RoleAccessPolicy.permits("USER", null));
    }
}
