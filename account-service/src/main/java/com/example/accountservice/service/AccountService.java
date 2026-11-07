package com.example.accountservice.service;

import com.example.accountservice.client.PasswordClient;
import com.example.accountservice.dto.*;
import com.example.accountservice.entity.User;
import com.example.accountservice.entity.UserRole;
import com.example.accountservice.repository.UserRepository;
import jakarta.persistence.criteria.Predicate;
import lombok.RequiredArgsConstructor;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class AccountService {
    private final UserRepository users;
    private final PasswordClient passwords; // băm/so khớp mật khẩu do auth-service làm (gọi qua gRPC)

    // Các hàm gọi `passwords` KHÔNG để @Transactional: gọi auth-service trước, ghi DB sau (mỗi lệnh repository tự có transaction ngắn),
    // nên không giữ kết nối DB trong lúc chờ mạng (tối đa vài giây mỗi lần gọi).

    // ── Admin ──────────────────────────────────────────────────────────────────

    @Transactional(readOnly = true)
    public PageResponse<AccountResponse> list(Long departmentId, UserRole role, Boolean active, int page, int size) {
        Specification<User> spec = (root, query, cb) -> {
            List<Predicate> filters = new ArrayList<>();
            if (departmentId != null) filters.add(cb.equal(root.get("departmentId"), departmentId));
            if (role != null) filters.add(cb.equal(root.get("role"), role));
            if (active != null) filters.add(cb.equal(root.get("isActive"), active));
            return cb.and(filters.toArray(Predicate[]::new));
        };
        return PageResponse.of(users.findAll(spec, PageRequest.of(page, size, Sort.by("id"))).map(AccountResponse::from));
    }

    public AccountResponse create(CreateAccountRequest req) {
        if (users.existsByUsername(req.username())) throw conflict("Username already exists");
        if (users.existsByEmailIgnoreCase(req.email())) throw conflict("Email already exists");
        User user = User.builder()
                .username(req.username())
                .email(req.email())
                .passwordHash(passwords.encode(req.tempPassword()))
                .departmentId(req.departmentId())
                .position(req.position())
                .role(req.role())
                .mustChangePassword(true)
                .build();
        return AccountResponse.from(save(user));
    }

    @Transactional
    public AccountResponse update(long id, UpdateAccountRequest req) {
        User user = find(id);
        // Email không đổi thì không thể trùng ai (create đã chặn trùng không phân biệt hoa/thường)
        if (!req.email().equalsIgnoreCase(user.getEmail()) && users.existsByEmailIgnoreCaseAndIdNot(req.email(), id)) {
            throw conflict("Email already exists");
        }
        user.setEmail(req.email());
        user.setDepartmentId(req.departmentId());
        user.setPosition(req.position());
        user.setRole(req.role());
        return AccountResponse.from(save(user));
    }

    /** Bật/tắt tài khoản; cả hai chiều đều hủy phiên đang có (tăng token_version). */
    @Transactional
    public AccountResponse setActive(long id, boolean active, long callerId) {
        // Admin tự khóa mình có thể làm hệ thống mất tài khoản quản trị cuối cùng
        if (!active && id == callerId) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "You cannot deactivate your own account");
        }
        User user = find(id);
        user.setActive(active);
        user.setTokenVersion(user.getTokenVersion() + 1);
        return AccountResponse.from(save(user));
    }

    public void resetPassword(long id, String tempPassword) {
        String hash = passwords.encode(tempPassword);
        if (users.updatePassword(id, hash, true, 1) == 0) throw new AccountNotFoundException();
    }

    // ── Người dùng tự phục vụ ──────────────────────────────────────────────────

    public AccountResponse me(long id) {
        return AccountResponse.from(find(id));
    }

    public void changePassword(long id, ChangePasswordRequest req) {
        User user = find(id);
        if (!passwords.matches(req.currentPassword(), user.getPasswordHash())) {
            // 400 (không phải 401): FE coi 401 là hết phiên và đăng xuất
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Current password is incorrect");
        }
        users.updatePassword(id, passwords.encode(req.newPassword()), false, 0);
    }

    // ── Cho auth-service (qua gRPC, xem AccountGrpcService) ────────────────────

    /** Tài khoản kèm mật khẩu đã băm để auth-service tự so khớp. Tài khoản bị khóa vẫn được trả về. */
    public User credentials(String username) {
        return users.findByUsername(username).orElseThrow(AccountNotFoundException::new);
    }

    public User find(long id) {
        return users.findById(id).orElseThrow(AccountNotFoundException::new);
    }

    public void bumpTokenVersion(long id) {
        if (users.bumpTokenVersion(id) == 0) throw new AccountNotFoundException();
    }

    // ───────────────────────────────────────────────────────────────────────────

    // Hai request cùng lúc có thể cùng lọt qua existsBy…; unique constraint của DB là chốt chặn cuối
    private User save(User user) {
        try {
            return users.saveAndFlush(user);
        } catch (DataIntegrityViolationException e) {
            throw conflict("Username or email already exists");
        }
    }

    private static ResponseStatusException conflict(String message) {
        return new ResponseStatusException(HttpStatus.CONFLICT, message);
    }
}
