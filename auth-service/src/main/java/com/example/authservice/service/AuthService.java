package com.example.authservice.service;

import com.example.authservice.dto.*;
import com.example.authservice.entity.*;
import com.example.authservice.repository.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

import javax.security.auth.RefreshFailedException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.HexFormat;
import java.util.UUID;

@Service
public class AuthService {
    private final UserRepository userRepository;
    private final RefreshTokenRepository refreshTokenRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtService jwtService;

    @Value("${application.jwt.refresh-token-expiration}")
    private long refreshTokenExpiration;

    public AuthService(UserRepository userRepository,
                       RefreshTokenRepository refreshTokenRepository,
                       PasswordEncoder passwordEncoder,
                       JwtService jwtService) {
        this.userRepository = userRepository;
        this.refreshTokenRepository = refreshTokenRepository;
        this.passwordEncoder = passwordEncoder;
        this.jwtService = jwtService;
    }

    @Transactional
    public AuthResponse register(RegisterRequest req) {
        if(userRepository.existsByUsername(req.username())) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Username has been exist !");
        }
        if(userRepository.existsByEmail(req.email())) {
            throw new ResponseStatusException(HttpStatus.CONFLICT, "Email has been exist !");
        }
        User user = User.builder()
                .username(req.username())
                .email(req.email())
                .passwordHash(passwordEncoder.encode(req.password()))
                .departmentId(req.departmentId())
                .role(UserRole.EMPLOYEE)
                .build();

        userRepository.save(user);
        return generateTokensAndSave(user);
    }

    @Transactional
    public AuthResponse login(LoginRequest req) {
        User user = userRepository.findByUsername(req.username())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Wrong Username or Password!"));

        if(!passwordEncoder.matches(req.password(), user.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Wrong Username or Password!");
        }

        if (!user.isActive()) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Account has been unactivated!");
        }

        return generateTokensAndSave(user);
    }

    @Transactional
    public void logout(RefreshTokenRequest req) {
        String tokenHash = hashSha256(req.refreshToken());

        refreshTokenRepository.findByTokenHash(tokenHash).ifPresent(token -> {
            token.setRevokedAt(Instant.now());
            refreshTokenRepository.save(token);

            userRepository.findById(token.getUserId()).ifPresent(user -> {
                user.setTokenVersion(user.getTokenVersion() + 1);
                userRepository.save(user);
            });
        });
    }

    @Transactional(noRollbackFor = ResponseStatusException.class)
    public AuthResponse refreshToken(RefreshTokenRequest req) {
        if (req.refreshToken() == null || req.refreshToken().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Refresh token must not be blank!");
        }

        String tokenHash = hashSha256(req.refreshToken());
        RefreshToken oldToken = refreshTokenRepository.findByTokenHash(tokenHash)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Refresh token is invalid!"));
        if (oldToken.isRevoked() || oldToken.getReplacedBy() != null) {
            userRepository.findById(oldToken.getUserId()).ifPresent(user -> {
                user.setTokenVersion(user.getTokenVersion() + 1);
                userRepository.save(user);
            });
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED,
                    "Security warning: Token has been used. All login session has been revoked!");
        }

        if (oldToken.isExpired()) {
            throw new ResponseStatusException(HttpStatus.UNAUTHORIZED, "Refresh token has expired, please login again!");
        }

        User user = userRepository.findById(oldToken.getUserId())
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.UNAUTHORIZED, "User does not exist!"));

        if (!user.isActive()) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Account has been unactivated!");
        }

        String newAccessToken = jwtService.generateAccessToken(user);
        String newRawRefreshToken = UUID.randomUUID().toString();
        String newRefreshTokenHash = hashSha256(newRawRefreshToken);

        RefreshToken newToken = RefreshToken.builder()
                .userId(user.getId())
                .tokenHash(newRefreshTokenHash)
                .expiresAt(Instant.now().plusMillis(refreshTokenExpiration))
                .build();
        refreshTokenRepository.save(newToken);

        oldToken.setRevokedAt(Instant.now());
        oldToken.setReplacedBy(newToken.getId());
        refreshTokenRepository.save(oldToken);

        return AuthResponse.of(newAccessToken, newRawRefreshToken);
    }

    private AuthResponse generateTokensAndSave(User user) {
        String accessToken = jwtService.generateAccessToken(user);
        String rawRefreshToken = UUID.randomUUID().toString();
        RefreshToken refreshToken = RefreshToken.builder()
                .userId(user.getId())
                .tokenHash(hashSha256(rawRefreshToken))
                .expiresAt(Instant.now().plusMillis(refreshTokenExpiration))
                .build();
        refreshTokenRepository.save(refreshToken);
        return AuthResponse.of(accessToken, rawRefreshToken);
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
