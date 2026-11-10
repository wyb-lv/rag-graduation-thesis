package com.example.authservice;

import com.example.accountservice.grpc.AccountInfo;
import com.example.accountservice.grpc.Credentials;
import com.example.authservice.client.AccountClient;
import com.example.authservice.grpc.TokenValidationRequest;
import com.example.authservice.grpc.TokenValidationResponse;
import com.example.authservice.repository.RefreshTokenRepository;
import com.example.authservice.service.AuthGrpcService;
import com.jayway.jsonpath.JsonPath;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;
import org.springframework.web.server.ResponseStatusException;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Luồng đăng nhập / refresh / logout và kiểm tra token của gateway, chạy với DB thật (H2) cho refresh_tokens
 * và một account-service giả giữ trạng thái tài khoản (cùng cách account-service tăng token_version khi khóa / đặt lại mật khẩu).
 */
@SpringBootTest
@AutoConfigureMockMvc
class AuthFlowIntegrationTest {
    // account-service trả mật khẩu đã băm; auth-service tự so khớp bằng PasswordEncoder của Spring Security (băm một lần cho cả lớp)
    static final String TEMP_HASH = new BCryptPasswordEncoder().encode("Temp@1234");

    @Autowired MockMvc mvc;
    @Autowired AuthGrpcService grpc;
    @Autowired RefreshTokenRepository refreshTokens;
    @MockitoBean AccountClient accounts;

    /** Trạng thái tài khoản "bob" bên account-service. */
    final AtomicReference<AccountInfo> bob = new AtomicReference<>();

    @BeforeEach
    void setUp() {
        refreshTokens.deleteAll();
        bob.set(AccountInfo.newBuilder().setId(7).setUsername("bob").setRole("EMPLOYEE").setDepartmentId(2)
                .setActive(true).setMustChangePassword(true).build());
        when(accounts.credentials(any())).thenReturn(Optional.empty());
        when(accounts.credentials("bob")).thenAnswer(i ->
                Optional.of(Credentials.newBuilder().setAccount(bob.get()).setPasswordHash(TEMP_HASH).build()));
        when(accounts.find(7L)).thenAnswer(i -> Optional.of(bob.get()));
        doAnswer(i -> {
            bumpVersion();
            return null;
        }).when(accounts).bumpTokenVersion(7L);
    }

    private void bumpVersion() {
        bob.updateAndGet(a -> a.toBuilder().setTokenVersion(a.getTokenVersion() + 1).build());
    }

    // ── Helpers ────────────────────────────────────────────────────────────────

    private ResultActions login(String username, String password) throws Exception {
        return mvc.perform(post("/api/auth/login").contentType(MediaType.APPLICATION_JSON)
                .content("{\"username\":\"%s\",\"password\":\"%s\"}".formatted(username, password)));
    }

    private ResultActions refresh(String refreshToken) throws Exception {
        return mvc.perform(post("/api/auth/refresh").contentType(MediaType.APPLICATION_JSON)
                .content("{\"refreshToken\":\"%s\"}".formatted(refreshToken)));
    }

    private static String field(ResultActions r, String name) throws Exception {
        return JsonPath.read(r.andReturn().getResponse().getContentAsString(), "$." + name);
    }

    private String[] loginAsBob() throws Exception {
        ResultActions r = login("bob", "Temp@1234").andExpect(status().isOk());
        return new String[]{field(r, "accessToken"), field(r, "refreshToken")};
    }

    private TokenValidationResponse validate(String token) {
        AtomicReference<TokenValidationResponse> result = new AtomicReference<>();
        grpc.validateToken(TokenValidationRequest.newBuilder().setToken(token).build(), new StreamObserver<>() {
            public void onNext(TokenValidationResponse v) { result.set(v); }
            public void onError(Throwable t) { throw new AssertionError(t); }
            public void onCompleted() {}
        });
        return result.get();
    }

    // ── Đăng nhập ──────────────────────────────────────────────────────────────

    @Test
    void login_ok_returnsTokensAndMustChangeFlag_andGatewayAcceptsAccessToken() throws Exception {
        ResultActions r = login("bob", "Temp@1234").andExpect(status().isOk())
                .andExpect(jsonPath("$.tokenType").value("Bearer"))
                .andExpect(jsonPath("$.mustChangePassword").value(true));

        TokenValidationResponse v = validate(field(r, "accessToken"));
        assertThat(v.getIsValid()).isTrue();
        assertThat(v.getUserId()).isEqualTo(7);
        assertThat(v.getUsername()).isEqualTo("bob");
        assertThat(v.getRole()).isEqualTo("EMPLOYEE");
        assertThat(v.getDepartmentId()).isEqualTo(2);
        assertThat(refreshTokens.count()).isEqualTo(1);
    }

    @Test
    void login_wrongPassword_401_andNoRefreshTokenIssued() throws Exception {
        login("bob", "wrong").andExpect(status().isUnauthorized());
        assertThat(refreshTokens.count()).isZero();
    }

    @Test
    void login_missingFields_401() throws Exception {
        mvc.perform(post("/api/auth/login").contentType(MediaType.APPLICATION_JSON).content("{}"))
                .andExpect(status().isUnauthorized());
    }

    @Test
    void login_deactivatedAccount_403() throws Exception {
        bob.updateAndGet(a -> a.toBuilder().setActive(false).build());

        login("bob", "Temp@1234").andExpect(status().isForbidden());
        assertThat(refreshTokens.count()).isZero();
    }

    // ── Vô hiệu hóa / đặt lại mật khẩu: token cũ bị từ chối ────────────────────

    @Test
    void afterDeactivate_oldAccessTokenRejected_oldRefreshTokenRejected() throws Exception {
        String[] t = loginAsBob();

        bob.updateAndGet(a -> a.toBuilder().setActive(false).setTokenVersion(a.getTokenVersion() + 1).build());

        assertThat(validate(t[0]).getIsValid()).isFalse();
        refresh(t[1]).andExpect(status().isForbidden());
    }

    @Test
    void afterReactivate_tokensIssuedBeforeStayRejected_butFreshLoginWorks() throws Exception {
        String[] old = loginAsBob();
        bob.updateAndGet(a -> a.toBuilder().setActive(false).setTokenVersion(a.getTokenVersion() + 1).build());
        bob.updateAndGet(a -> a.toBuilder().setActive(true).setTokenVersion(a.getTokenVersion() + 1).build());

        assertThat(validate(old[0]).getIsValid()).isFalse();
        refresh(old[1]).andExpect(status().isUnauthorized());
        assertThat(validate(loginAsBob()[0]).getIsValid()).isTrue();
    }

    @Test
    void afterPasswordReset_oldAccessAndRefreshTokensRejected_loginWithNewPasswordWorks() throws Exception {
        String[] t = loginAsBob();

        bumpVersion(); // account-service: reset-password

        assertThat(validate(t[0]).getIsValid()).isFalse();
        refresh(t[1]).andExpect(status().isUnauthorized());
        assertThat(validate(loginAsBob()[0]).getIsValid()).isTrue();
    }

    // ── Refresh / logout ───────────────────────────────────────────────────────

    @Test
    void refresh_rotatesToken_andReusingTheOldOneRevokesAllSessions() throws Exception {
        String[] t = loginAsBob();

        ResultActions r = refresh(t[1]).andExpect(status().isOk());
        String newAccess = field(r, "accessToken");
        assertThat(validate(newAccess).getIsValid()).isTrue();

        refresh(t[1]).andExpect(status().isUnauthorized());
        verify(accounts).bumpTokenVersion(7L);
        assertThat(validate(newAccess).getIsValid()).isFalse();
    }

    @Test
    void logout_revokesRefreshToken_andInvalidatesAccessToken() throws Exception {
        String[] t = loginAsBob();

        mvc.perform(post("/api/auth/logout").contentType(MediaType.APPLICATION_JSON)
                .content("{\"refreshToken\":\"%s\"}".formatted(t[1]))).andExpect(status().isOk());

        verify(accounts).bumpTokenVersion(7L);
        assertThat(validate(t[0]).getIsValid()).isFalse();
        refresh(t[1]).andExpect(status().isUnauthorized());
    }

    @Test
    void refresh_unknownOrBlankToken() throws Exception {
        refresh("not-a-real-token").andExpect(status().isUnauthorized());
        refresh("").andExpect(status().isBadRequest());
    }

    // ── Kiểm tra token của gateway ─────────────────────────────────────────────

    @Test
    void validate_garbageToken_isInvalid_withoutCallingAccountService() {
        assertThat(validate("not.a.jwt").getIsValid()).isFalse();
        verify(accounts, never()).find(anyLong());
    }

    @Test
    void validate_tokenOfDeletedAccount_isInvalid() throws Exception {
        String[] t = loginAsBob();
        when(accounts.find(7L)).thenReturn(Optional.empty());

        assertThat(validate(t[0]).getIsValid()).isFalse();
    }

    // ── account-service không trả lời ──────────────────────────────────────────

    @Test
    void accountServiceDown_loginAndRefresh_503() throws Exception {
        String[] t = loginAsBob();
        var down = new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Account service unavailable");
        when(accounts.credentials(any())).thenThrow(down);
        when(accounts.find(anyLong())).thenThrow(down);

        login("bob", "Temp@1234").andExpect(status().isServiceUnavailable());
        refresh(t[1]).andExpect(status().isServiceUnavailable());
    }

    @Test
    void accountServiceDown_gatewayValidationGetsUNAVAILABLE_notInvalidToken() throws Exception {
        String[] t = loginAsBob();
        when(accounts.find(anyLong())).thenThrow(new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "down"));

        AtomicReference<Throwable> error = new AtomicReference<>();
        grpc.validateToken(TokenValidationRequest.newBuilder().setToken(t[0]).build(), new StreamObserver<>() {
            public void onNext(TokenValidationResponse v) { throw new AssertionError("must not answer valid/invalid"); }
            public void onError(Throwable e) { error.set(e); }
            public void onCompleted() {}
        });

        assertThat(Status.fromThrowable(error.get()).getCode()).isEqualTo(Status.Code.UNAVAILABLE);
    }
}
