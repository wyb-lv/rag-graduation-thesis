package com.example.accountservice.config;

import com.example.accountservice.entity.User;
import com.example.accountservice.entity.UserRole;
import com.example.accountservice.repository.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Seed tài khoản mẫu cho môi trường dev. Chỉ chạy khi app.seed.enabled=true (bật qua ACCOUNT_SEED_ENABLED trong docker-compose).
 * Là cách duy nhất tạo Admin đầu tiên. Bỏ qua username/email đã tồn tại nên chạy lại nhiều lần vẫn an toàn.
 * Mật khẩu để sẵn dạng BCrypt (strength 10, cùng thuật toán auth-service dùng để so khớp) vì account-service không băm mật khẩu
 * và lúc khởi động auth-service có thể chưa chạy. Mật khẩu gốc ghi ở comment mỗi dòng (chỉ dùng cho dev).
 */
@Slf4j
@Component
@ConditionalOnProperty(name = "app.seed.enabled", havingValue = "true")
public class DataSeeder implements CommandLineRunner {

    private record Seed(String username, String email, String passwordHash, long departmentId, UserRole role) {}

    private static final String EMPLOYEE_PW = "$2a$10$Yh1mSw4kR.8dpu51dNTnquIvj3Vr5yRR0OqeESnXiqtx.tSM615oW"; // Employee@123

    private static final List<Seed> SEEDS = List.of(
            new Seed("admin", "admin@hsu.edu.vn", "$2a$10$iunL1xAsNIxjUeafFWTbq.sNcI.Ntb9dq4g66NollJDEBoyaqz.gK", 1, UserRole.ADMIN), // Admin@123
            new Seed("manager", "manager@hsu.edu.vn", "$2a$10$E52qrR0.FMbvE4rea4JhuuxP/kGgC.uCl/gNhvDARmjV.ff.x8SRC", 1, UserRole.MANAGER), // Manager@123
            new Seed("employee1", "employee1@hsu.edu.vn", EMPLOYEE_PW, 1, UserRole.EMPLOYEE),
            new Seed("employee2", "employee2@hsu.edu.vn", EMPLOYEE_PW, 2, UserRole.EMPLOYEE),
            // Tài khoản của nút "Đăng nhập nhanh" trên FE (AuthScreen)
            new Seed("demo_hsu", "demo@hsu.edu.vn", "$2a$10$G2t6KL18M68HJcACTjFxWOW8ISeyQo/zbFcdIbqT90llo.abXIwvO", 1, UserRole.EMPLOYEE) // Password@123
    );

    private final UserRepository userRepository;

    public DataSeeder(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    @Override
    public void run(String... args) {
        for (Seed s : SEEDS) {
            if (userRepository.existsByUsername(s.username()) || userRepository.existsByEmailIgnoreCase(s.email())) continue;
            userRepository.save(User.builder()
                    .username(s.username())
                    .email(s.email())
                    .passwordHash(s.passwordHash())
                    .departmentId(s.departmentId())
                    .role(s.role())
                    .build());
            log.info("Seeded user '{}' ({})", s.username(), s.role());
        }
    }
}
