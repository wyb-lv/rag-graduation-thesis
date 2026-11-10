package com.example.accountservice.dto;

import com.example.accountservice.entity.User;
import com.example.accountservice.entity.UserRole;

/** Không bao giờ chứa passwordHash. */
public record AccountResponse(
        long id,
        String username,
        String email,
        Long departmentId,
        String position,
        UserRole role,
        boolean active,
        boolean mustChangePassword
) {
    public static AccountResponse from(User u) {
        return new AccountResponse(u.getId(), u.getUsername(), u.getEmail(), u.getDepartmentId(),
                u.getPosition(), u.getRole(), u.isActive(), u.isMustChangePassword());
    }
}
