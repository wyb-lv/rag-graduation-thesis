package com.example.authservice.dto;

public record AuthResponse (
        String accessToken,
        String refreshToken,
        String tokenType,
        boolean mustChangePassword
){
    public static AuthResponse of(String accessToken, String refreshToken, boolean mustChangePassword) {
        return new AuthResponse(accessToken, refreshToken, "Bearer", mustChangePassword);
    }
}
