package com.hsu.gateway.filter;

import java.util.List;
import java.util.regex.Pattern;

/**
 * Phân quyền (RBAC) tại gateway. ADMIN dùng mọi API. Role khác (USER) chỉ được hỏi đáp AI (chat, giọng nói, lịch sử hội thoại)
 * và tự xem / đổi thông tin, mật khẩu của chính mình; mọi đường dẫn khác bị từ chối mặc định (quản lý tài liệu, embed, quản lý tài khoản...).
 * Chỉ nhận ký tự [A-Za-z0-9_/-] trong đường dẫn nên không lách được bằng "..", "%2e", ";"...
 * Thêm endpoint hỏi đáp mới ở ai-service thì phải thêm vào USER_PATHS, nếu không USER sẽ nhận 403.
 */
final class RoleAccessPolicy {

    static final String ADMIN = "ADMIN";

    private static final List<Pattern> USER_PATHS = List.of(
            Pattern.compile("/api/ai/(ask|ask/stream|ask/voice/stream|stt|tts|conversations)"),
            Pattern.compile("/api/ai/conversations/[A-Za-z0-9_-]+"),
            Pattern.compile("/api/ai/call/[a-z]+"),
            // account-service tự khóa các endpoint này về đúng tài khoản trong token
            Pattern.compile("/api/accounts/me(/[a-z-]+)?")
    );

    private RoleAccessPolicy() {
    }

    static boolean permits(String role, String path) {
        if (ADMIN.equals(role)) return true;
        return path != null && USER_PATHS.stream().anyMatch(p -> p.matcher(path).matches());
    }
}
