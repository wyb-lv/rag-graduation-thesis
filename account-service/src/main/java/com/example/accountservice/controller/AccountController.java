package com.example.accountservice.controller;

import com.example.accountservice.dto.*;
import com.example.accountservice.entity.UserRole;
import com.example.accountservice.service.AccountService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

@Tag(name = "Accounts", description = "Quản lý tài khoản. Chỉ ADMIN gọi được (401 chưa đăng nhập, 403 không phải ADMIN), riêng /me dành cho mọi người dùng đã đăng nhập.")
@RestController
@RequestMapping("/api/accounts")
@RequiredArgsConstructor
public class AccountController {
    private final AccountService accounts;

    @Operation(summary = "Danh sách tài khoản (phân trang, sắp xếp theo id), lọc theo đơn vị / role / trạng thái")
    @GetMapping
    public PageResponse<AccountResponse> list(
            @RequestParam(required = false) Long departmentId,
            @RequestParam(required = false) UserRole role,
            @RequestParam(required = false) Boolean active,
            @RequestParam(defaultValue = "0") @Min(0) int page,
            @RequestParam(defaultValue = "20") @Min(1) @Max(100) int size) {
        return accounts.list(departmentId, role, active, page, size);
    }

    @Operation(summary = "Tạo tài khoản với mật khẩu tạm; người dùng bị buộc đổi mật khẩu ở lần đăng nhập đầu")
    @ApiResponse(responseCode = "409", description = "Trùng username hoặc email", content = @Content)
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public AccountResponse create(@Valid @RequestBody CreateAccountRequest req) {
        return accounts.create(req);
    }

    @Operation(summary = "Sửa email, đơn vị, chức vụ, role (username không đổi được)")
    @ApiResponse(responseCode = "404", description = "Không có tài khoản", content = @Content)
    @ApiResponse(responseCode = "409", description = "Trùng email", content = @Content)
    @PutMapping("/{id}")
    public AccountResponse update(@PathVariable long id, @Valid @RequestBody UpdateAccountRequest req) {
        return accounts.update(id, req);
    }

    @Operation(summary = "Vô hiệu hóa (active=false) hoặc kích hoạt lại (active=true); tăng token_version nên mọi access token và refresh token đã cấp đều bị từ chối")
    @ApiResponse(responseCode = "400", description = "Admin tự vô hiệu hóa chính mình", content = @Content)
    @ApiResponse(responseCode = "404", description = "Không có tài khoản", content = @Content)
    @PatchMapping("/{id}/status")
    public AccountResponse setStatus(@PathVariable long id, @Valid @RequestBody StatusRequest req,
                                     @Parameter(hidden = true) @RequestHeader("X-User-Id") long callerId) {
        return accounts.setActive(id, req.active(), callerId);
    }

    @Operation(summary = "Đặt lại mật khẩu: mật khẩu tạm, buộc đổi ở lần đăng nhập sau, hủy mọi phiên (access và refresh token) đang đăng nhập")
    @ApiResponse(responseCode = "404", description = "Không có tài khoản", content = @Content)
    @PostMapping("/{id}/reset-password")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void resetPassword(@PathVariable long id, @Valid @RequestBody ResetPasswordRequest req) {
        accounts.resetPassword(id, req.tempPassword());
    }

    @SelfService
    @Operation(summary = "Thông tin tài khoản của chính người gọi (mọi role)")
    @GetMapping("/me")
    public AccountResponse me(@Parameter(hidden = true) @RequestHeader("X-User-Id") long callerId) {
        return accounts.me(callerId);
    }

    @SelfService
    @Operation(summary = "Tự đổi mật khẩu (mọi role); xóa cờ mustChangePassword")
    @ApiResponse(responseCode = "400", description = "Sai mật khẩu hiện tại hoặc mật khẩu mới không hợp lệ", content = @Content)
    @PostMapping("/me/change-password")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void changePassword(@Valid @RequestBody ChangePasswordRequest req,
                               @Parameter(hidden = true) @RequestHeader("X-User-Id") long callerId) {
        accounts.changePassword(callerId, req);
    }
}
