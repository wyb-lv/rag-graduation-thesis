package com.hsu.gateway.filter;

import com.example.authservice.grpc.TokenValidationRequest;
import com.example.authservice.grpc.TokenValidationResponse;
import com.example.authservice.grpc.TokenValidationServiceGrpc.TokenValidationServiceBlockingStub;
import jakarta.servlet.http.HttpServletRequest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class JwtAuthFilterRbacTest {

    final TokenValidationServiceBlockingStub stub = mock(TokenValidationServiceBlockingStub.class);
    final JwtAuthFilter filter = new JwtAuthFilter();

    @BeforeEach
    void wireStub() {
        ReflectionTestUtils.setField(filter, "tokenValidationStub", stub);
    }

    private void tokenOf(String role) {
        when(stub.validateToken(any(TokenValidationRequest.class))).thenReturn(TokenValidationResponse.newBuilder()
                .setIsValid(true).setUserId(7).setUsername("bob").setRole(role).setDepartmentId(1).build());
    }

    private MockHttpServletResponse call(String method, String path, MockFilterChain chain) throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest(method, path);
        request.addHeader("Authorization", "Bearer t");
        MockHttpServletResponse response = new MockHttpServletResponse();
        filter.doFilter(request, response, chain);
        return response;
    }

    @Test
    void userAskingTheAssistantPassesWithIdentityHeaders() throws Exception {
        tokenOf("USER");
        MockFilterChain chain = new MockFilterChain();

        MockHttpServletResponse res = call("POST", "/api/ai/ask/stream", chain);

        assertEquals(200, res.getStatus());
        HttpServletRequest forwarded = (HttpServletRequest) chain.getRequest();
        assertNotNull(forwarded);
        assertEquals("USER", forwarded.getHeader("X-User-Role"));
        assertEquals("7", forwarded.getHeader("X-User-Id"));
    }

    @Test
    void userOnDocumentsGets403AndNeverReachesTheService() throws Exception {
        tokenOf("USER");
        MockFilterChain chain = new MockFilterChain();

        MockHttpServletResponse res = call("POST", "/api/documents/upload", chain);

        assertEquals(403, res.getStatus());
        assertNull(chain.getRequest());
        assertEquals("{\"error\":\"Forbidden\",\"message\":\"Tài khoản của bạn không có quyền sử dụng chức năng này.\"}", res.getContentAsString());
    }

    @Test
    void userCannotEmbedOrManageOtherAccounts() throws Exception {
        tokenOf("USER");
        for (String path : new String[]{"/api/ai/embed", "/api/ai/embed/7", "/api/accounts", "/api/accounts/7/status"}) {
            MockFilterChain chain = new MockFilterChain();
            assertEquals(403, call("POST", path, chain).getStatus(), path);
            assertNull(chain.getRequest(), path);
        }
    }

    @Test
    void adminReachesEverything() throws Exception {
        tokenOf("ADMIN");
        for (String path : new String[]{"/api/documents/upload", "/api/ai/embed", "/api/accounts", "/api/ai/ask/stream"}) {
            MockFilterChain chain = new MockFilterChain();
            assertEquals(200, call("POST", path, chain).getStatus(), path);
            assertNotNull(chain.getRequest(), path);
        }
    }

    @Test
    void invalidTokenIsStill401() throws Exception {
        when(stub.validateToken(any(TokenValidationRequest.class))).thenReturn(TokenValidationResponse.newBuilder()
                .setIsValid(false).setErrorMessage("expired").build());
        MockFilterChain chain = new MockFilterChain();

        assertEquals(401, call("POST", "/api/ai/ask", chain).getStatus());
        assertNull(chain.getRequest());
    }
}
