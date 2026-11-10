package com.example.accountservice.entity;

import jakarta.persistence.*;
import lombok.*;

/** Nguồn dữ liệu duy nhất về tài khoản (DB riêng của account-service). auth-service chỉ truy cập qua gRPC (AccountGrpcService). */
@Entity
@Table(name = "users")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class User {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private long id;

    @Column(nullable = false, unique = true, length = 50)
    private String username;

    @Column(nullable = false, unique = true)
    private String email;

    @Column(name = "password_hash", nullable = false)
    private String passwordHash;

    @Column(name = "department_id", nullable = false)
    private Long departmentId;

    @Column(length = 100)
    private String position;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private UserRole role;

    /** Tăng lên để vô hiệu hóa mọi access token và refresh token đã cấp (auth-service so khớp với giá trị này). */
    @Column(name = "token_version", nullable = false)
    @Builder.Default
    private int tokenVersion = 0;

    @Column(name = "is_active", nullable = false)
    @Builder.Default
    private boolean isActive = true;

    @Column(name = "must_change_password", nullable = false)
    @Builder.Default
    private boolean mustChangePassword = false;
}
