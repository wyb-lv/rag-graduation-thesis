package com.example.accountservice.dto;

import com.example.accountservice.entity.UserRole;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

/** Username không đổi được vì là subject của JWT. */
public record UpdateAccountRequest(
        @NotBlank @Email @Size(max = 255) String email,
        @NotNull Long departmentId,
        @Size(max = 100) String position,
        @NotNull UserRole role
) {}
