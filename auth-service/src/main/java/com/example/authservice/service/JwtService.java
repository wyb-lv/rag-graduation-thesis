package com.example.authservice.service;

import com.example.accountservice.grpc.AccountInfo;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtParser;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;

@Service
public class JwtService {
    private final SecretKey key;
    private final JwtParser parser;
    private final long accessTokenExpiration;

    public JwtService(@Value("${application.jwt.secret-key}") String secretKey,
                      @Value("${application.jwt.access-token-expiration}") long accessTokenExpiration) {
        this.key = Keys.hmacShaKeyFor(secretKey.getBytes(StandardCharsets.UTF_8));
        this.parser = Jwts.parser().verifyWith(key).build();
        this.accessTokenExpiration = accessTokenExpiration;
    }

    public String generateAccessToken(AccountInfo user) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("userId", user.getId());
        claims.put("departmentId", user.getDepartmentId());
        claims.put("role", user.getRole());
        claims.put("tokenVersion", user.getTokenVersion());

        Date now = new Date();
        Date expiryDate = new Date(now.getTime() + accessTokenExpiration);

        return Jwts.builder()
                .subject(user.getUsername())
                .claims(claims)
                .issuedAt(now)
                .expiration(expiryDate)
                .signWith(key)
                .compact();
    }

    /** Kiểm tra chữ ký và hạn một lần; ném JwtException nếu token sai hoặc hết hạn. */
    public Claims parse(String token) {
        return parser.parseSignedClaims(token).getPayload();
    }

    /** Token còn giá trị nếu đúng người dùng và tokenVersion chưa bị account-service tăng lên. */
    public boolean isTokenValid(Claims claims, AccountInfo user) {
        Integer tokenVersion = claims.get("tokenVersion", Integer.class);
        return user.getUsername().equals(claims.getSubject())
                && tokenVersion != null
                && tokenVersion == user.getTokenVersion();
    }
}
