package com.example.authservice.service;

import com.example.authservice.entity.User;
import io.jsonwebtoken.Claims;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.crypto.SecretKey;
import java.nio.charset.StandardCharsets;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

@Service
public class JwtService {
    @Value("${application.jwt.secret-key}")
    private String secretKey;

    @Value("${application.jwt.access-token-expiration}")
    private long accessTokenExpiration;

    private SecretKey getSigningKey() {
        byte[] keyBytes = secretKey.getBytes(StandardCharsets.UTF_8);
        return Keys.hmacShaKeyFor(keyBytes);
    }

    public String generateAccessToken(User user) {
        Map<String, Object> claims = new HashMap<>();
        claims.put("userId", user.getId());
        claims.put("departmentId", user.getDepartmentId());
        claims.put("role", user.getRole().name());
        claims.put("tokenVersion", user.getTokenVersion());

        Date now = new Date();
        Date expiryDate = new Date(now.getTime() + accessTokenExpiration);

        return Jwts.builder()
                .subject(user.getUsername())
                .claims(claims)
                .issuedAt(now)
                .expiration(expiryDate)
                .signWith(getSigningKey())
                .compact();
    }

    public String extractUsername(String token) {
        return extractClaim(token, Claims::getSubject);
    }

    public Integer extractTokenVersion(String token) {
        return extractClaim(token, claims -> claims.get("tokenVersion", Integer.class));
    }

    public <T> T extractClaim(String token, Function<Claims, T> claimsResolver) {
        final Claims claims = extractAllClaims(token);
        return claimsResolver.apply(claims);
    }

    public Claims extractAllClaims(String token) {
        return Jwts.parser()
                .verifyWith(getSigningKey())
                .build()
                .parseSignedClaims(token)
                .getPayload();
    }

    public boolean isTokenValid(String token, User user) {
        try {
            Claims claims = extractAllClaims(token);
            String username = claims.getSubject();
            Integer tokenVersion = claims.get("tokenVersion", Integer.class);

            return username.equals(user.getUsername())
                    && tokenVersion != null
                    && tokenVersion == user.getTokenVersion();
        } catch (JwtException | IllegalArgumentException e) {
            return false;
        }
    }
}
