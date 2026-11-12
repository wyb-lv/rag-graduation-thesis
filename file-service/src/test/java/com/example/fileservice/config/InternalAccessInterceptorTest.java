package com.example.fileservice.config;

import com.example.fileservice.controllers.OcrCallbackController;
import com.example.fileservice.services.OcrResultService;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

class InternalAccessInterceptorTest {

    private static final String BODY = "{\"status\":\"PROCESSING\"}";

    private final OcrResultService service = mock(OcrResultService.class);

    private MockMvc mvc(String configuredToken) {
        return MockMvcBuilders.standaloneSetup(new OcrCallbackController(service))
                .addMappedInterceptors(new String[]{"/internal/**"}, new InternalAccessInterceptor(configuredToken))
                .build();
    }

    private org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder call(String token) {
        var request = put("/internal/documents/7/ocr").contentType(MediaType.APPLICATION_JSON).content(BODY);
        return token == null ? request : request.header(InternalAccessInterceptor.HEADER, token);
    }

    @Test
    void rejectsRequestWithoutToken() throws Exception {
        mvc("s3cret").perform(call(null)).andExpect(status().isUnauthorized());
        verify(service, never()).start(anyLong());
    }

    @Test
    void rejectsWrongToken() throws Exception {
        mvc("s3cret").perform(call("s3cret-but-longer")).andExpect(status().isUnauthorized());
        mvc("s3cret").perform(call("wrong!")).andExpect(status().isUnauthorized());
        verify(service, never()).start(anyLong());
    }

    @Test
    void acceptsCorrectToken() throws Exception {
        when(service.start(7L)).thenReturn(OcrResultService.Outcome.RECORDED);
        mvc("s3cret").perform(call("s3cret")).andExpect(status().isNoContent());
        verify(service).start(7L);
    }

    @Test
    void failsClosedWhenNoTokenIsConfigured() throws Exception {
        mvc("").perform(call("")).andExpect(status().isUnauthorized());
        mvc(null).perform(call("anything")).andExpect(status().isUnauthorized());
        verify(service, never()).complete(anyLong(), any());
        verify(service, never()).start(anyLong());
    }
}
