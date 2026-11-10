package com.example.accountservice.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.core.annotation.Order;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

/**
 * Chỉ còn hai role ADMIN và USER. DB tạo từ bản cũ còn các hàng MANAGER / EMPLOYEE và check constraint cũ trên cột role
 * (ddl-auto=update không sửa constraint): bỏ constraint rồi đổi các hàng đó sang USER, nếu không đọc / ghi tài khoản sẽ lỗi.
 * Chạy trước DataSeeder; chạy lại nhiều lần vẫn an toàn.
 */
@Slf4j
@Component
@Order(0)
public class LegacyRoleMigration implements CommandLineRunner {

    private final JdbcTemplate jdbc;

    public LegacyRoleMigration(JdbcTemplate jdbc) {
        this.jdbc = jdbc;
    }

    @Override
    public void run(String... args) {
        jdbc.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check");
        int migrated = jdbc.update("UPDATE users SET role = 'USER' WHERE role NOT IN ('ADMIN', 'USER')");
        if (migrated > 0) log.info("Đã đổi {} tài khoản có role cũ (MANAGER / EMPLOYEE) sang USER", migrated);
    }
}
