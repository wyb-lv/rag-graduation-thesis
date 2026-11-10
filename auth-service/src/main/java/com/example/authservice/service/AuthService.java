package com.example.authservice.service;

import com.example.accountservice.grpc.AccountInfo;
import com.example.accountservice.grpc.Credentials;
import com.example.authservice.client.AccountClient;
import com.example.authservice.dto.*;
import com.example.authservice.entity.RefreshToken;
import com.example.authservice.repository.RefreshTokenRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.server.ResponseStatusException;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.HexFormat;
import java.util.UUID;

/** Chỉ lo xác thực và cấp token. Dữ liệu tài khoản (mật khẩu, trạng thái, role) thuộc account-service. */
@Service
public class AuthService {
    private final RefreshTokenRepository refreshTokenRepository;
    private final AccountClient accounts;
    private final JwtService jwtService;
    private final PasswordEncoder passwordEncoder;

    @Value("${application.jwt.refresh-token-expiration}")
    private long refreshTokenExpiration;

    public AuthService(RefreshTokenRepository refreshTokenRepository, AccountClient accounts, JwtService jwtService,
                       PasswordEncoder passwordEncoder) {
        this.refreshTokenRepository = refreshTokenRepository;
        this.accounts = accounts;
        this.jwtService = jwtService;
        this.passwordEncoder = passwordEncoder;
    }

    public AuthResponse login(LoginRequest req) {
        if (req.username() == null || req.password() == null) throw badCredentials();
        Credentials credentials = accounts.credentials(req.username()).orElseThrow(AuthService::badCredentials);
        if (!passwordEncoder.matches(req.password(), credentials.getPasswordHash())) throw badCredentials();

        AccountInfo user = credentials.getAccount();
        if (!user.getActive()) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Account has been unactivated!");
        }

        String rawRefreshToken = UUID.randomUUID().toString();
        saveRefreshToken(user, rawRefreshToken);
        return respond(user, rawRefreshToken);
    }

    @Transactional
    public void logout(RefreshTokenRequest req) {
        String tokenHash = hashSha256(req.refreshToken());

        refreshTokenRepository.findByTokenHash(tokenHash).ifPresent(token -> {
            token.setRevokedAt(Instant.now());
            refreshTokenRepository.save(token);
            accounts.bumpTokenVersion(token.getUserId());
        });
    }

    @Transactional
    public AuthResponse refreshToken(RefreshTokenRequest req) {
        if (req.refreshToken() == null || req.refreshToken().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Refresh token must not be blank!");
        }

        String tokenHash = hashSha256(req.refreshToken());
        RefreshToken oldToken = refreshTokenRepository.findByTokenHash(tokenHash)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Refresh token is invalid!"));
        if (oldToken.isRevoked() || oldToken.getReplacedBy() != null) {
            accounts.bumpTokenVersion(oldToken.getUserId());
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED,
                    "Security warning: Token has been used. All login session has been revoked!");
        }

        if (oldToken.isExpired()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Refresh token has expired, please login again!");
        }

        AccountInfo user = accounts.find(oldToken.getUserId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "User does not exist!"));

        if (!user.getActive()) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Account has been unactivated!");
        }

        // Account-service tăng token_version khi vô hiệu hóa/kích hoạt lại, đặt lại mật khẩu: refresh token cấp trước đó coi như bị thu hồi
        if (oldToken.getTokenVersion() != user.getTokenVersion()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Refresh token is invalid!");
        }

        String newRawRefreshToken = UUID.randomUUID().toString();
        RefreshToken newToken = saveRefreshToken(user, newRawRefreshToken);
        oldToken.setRevokedAt(Instant.now());
        oldToken.setReplacedBy(newToken.getId());
        refreshTokenRepository.save(oldToken);

        return respond(user, newRawRefreshToken);
    }

    // Sai username và sai mật khẩu phải cùng một thông báo để không lộ username nào tồn tại
    private static ResponseStatusException badCredentials() {
        return new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Wrong Username or Password!");
    }

    private RefreshToken saveRefreshToken(AccountInfo user, String rawRefreshToken) {
        return refreshTokenRepository.save(RefreshToken.builder()
                .userId(user.getId())
                .tokenHash(hashSha256(rawRefreshToken))
                .tokenVersion(user.getTokenVersion())
                .expiresAt(Instant.now().plusMillis(refreshTokenExpiration))
                .build());
    }

    private AuthResponse respond(AccountInfo user, String rawRefreshToken) {
        return AuthResponse.of(jwtService.generateAccessToken(user), rawRefreshToken, user.getMustChangePassword());
    }

    private String hashSha256(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] encodedhash = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(encodedhash);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("Error encrypt algorithm", e);
        }
    }
}
