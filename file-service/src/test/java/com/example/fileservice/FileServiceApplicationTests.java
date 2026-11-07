package com.example.fileservice;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

/** Nạp toàn bộ context mà không cần Postgres / RabbitMQ / MinIO thật: H2 thay Postgres, listener RabbitMQ không tự chạy (nên không mở kết nối). */
@SpringBootTest(properties = {
        "spring.datasource.url=jdbc:h2:mem:file_service;DB_CLOSE_DELAY=-1",
        "spring.datasource.username=sa",
        "spring.datasource.password=",
        "spring.rabbitmq.listener.simple.auto-startup=false"
})
class FileServiceApplicationTests {

    @Test
    void contextLoads() {
    }

}
