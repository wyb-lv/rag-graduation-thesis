package com.example.accountservice.dto;

import com.example.accountservice.entity.UserRole;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public record CreateAccountRequest(
        @NotBlank @Size(max = 50) @Pattern(regexp = "\\S+", message = "must not contain whitespace") String username,
        @NotBlank @Email @Size(max = 255) String email,
        @NotNull Long departmentId,
        @Size(max = 100) String position,
        @NotNull UserRole role,
        // Mật khẩu tạm: người dùng bị buộc đổi ở lần đăng nhập đầu. Tối đa 72 vì BCrypt chỉ đọc 72 byte đầu.
        @NotBlank @Size(min = 8, max = 72) String tempPassword
) {}
