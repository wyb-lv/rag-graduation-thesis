package com.example.authservice.config;

import com.example.authservice.entity.User;
import com.example.authservice.entity.UserRole;
import com.example.authservice.repository.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Seed tài khoản mẫu cho môi trường dev. Chỉ chạy khi app.seed.enabled=true (bật trong docker-compose gốc).
 * Bỏ qua username đã tồn tại nên chạy lại nhiều lần vẫn an toàn.
 */
@Slf4j
@Component
@ConditionalOnProperty(name = "app.seed.enabled", havingValue = "true")
public class DataSeeder implements CommandLineRunner {

    private record Seed(String username, String email, String password, long departmentId, UserRole role) {}

    private static final List<Seed> SEEDS = List.of(
            new Seed("admin", "admin@hsu.edu.vn", "Admin@123", 1, UserRole.ADMIN),
            new Seed("manager", "manager@hsu.edu.vn", "Manager@123", 1, UserRole.MANAGER),
            new Seed("employee1", "employee1@hsu.edu.vn", "Employee@123", 1, UserRole.EMPLOYEE),
            new Seed("employee2", "employee2@hsu.edu.vn", "Employee@123", 2, UserRole.EMPLOYEE),
            // Tài khoản của nút "Đăng nhập nhanh" trên FE (AuthScreen)
            new Seed("demo_hsu", "demo@hsu.edu.vn", "Password@123", 1, UserRole.EMPLOYEE)
    );

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public DataSeeder(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    @Override
    public void run(String... args) {
        for (Seed s : SEEDS) {
            if (userRepository.existsByUsername(s.username()) || userRepository.existsByEmail(s.email())) continue;
            userRepository.save(User.builder()
                    .username(s.username())
                    .email(s.email())
                    .passwordHash(passwordEncoder.encode(s.password()))
                    .departmentId(s.departmentId())
                    .role(s.role())
                    .build());
            log.info("Seeded user '{}' ({})", s.username(), s.role());
        }
    }
}
