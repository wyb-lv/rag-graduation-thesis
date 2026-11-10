package com.example.authservice;

import com.example.authservice.client.AccountClient;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

/** api-gateway lấy /v3/api-docs không kèm token để gom vào Swagger UI, nên Spring Security không được chặn đường dẫn này. */
@SpringBootTest
@AutoConfigureMockMvc
class OpenApiDocsTest {

    @Autowired MockMvc mvc;
    @MockitoBean AccountClient accounts;

    @Test
    void apiDocsAreServedWithoutAToken() throws Exception {
        mvc.perform(get("/v3/api-docs"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.paths['/api/auth/login']").exists())
                .andExpect(jsonPath("$.paths['/api/auth/refresh']").exists());
    }

    @Test
    void otherPathsStillNeedAuthentication() throws Exception {
        mvc.perform(get("/api/auth/anything-else")).andExpect(status().is4xxClientError());
    }
}
