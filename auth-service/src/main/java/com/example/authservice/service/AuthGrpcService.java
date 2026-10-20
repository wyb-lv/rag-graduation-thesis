package com.example.authservice.service;

import com.example.authservice.entity.User;
import com.example.authservice.grpc.TokenValidationRequest;
import com.example.authservice.grpc.TokenValidationResponse;
import com.example.authservice.grpc.TokenValidationServiceGrpc.TokenValidationServiceImplBase;
import com.example.authservice.repository.UserRepository;
import io.grpc.stub.StreamObserver;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.devh.boot.grpc.server.service.GrpcService;

import java.util.Optional;

@Slf4j
@GrpcService
@RequiredArgsConstructor
public class AuthGrpcService extends TokenValidationServiceImplBase {

    private final JwtService jwtService;
    private final UserRepository userRepository;

    @Override
    public void validateToken(TokenValidationRequest request, StreamObserver<TokenValidationResponse> responseObserver) {
        String token = request.getToken();
        TokenValidationResponse.Builder responseBuilder = TokenValidationResponse.newBuilder();

        try {
            String username = jwtService.extractUsername(token);
            Optional<User> userOpt = userRepository.findByUsername(username);

            if (userOpt.isEmpty()) {
                responseBuilder.setIsValid(false).setErrorMessage("Người dùng không tồn tại");
            } else {
                User user = userOpt.get();

                if (!user.isActive()) {
                    responseBuilder.setIsValid(false).setErrorMessage("Tài khoản đã bị vô hiệu hóa");
                } else if (!jwtService.isTokenValid(token, user)) {
                    responseBuilder.setIsValid(false).setErrorMessage("Token không hợp lệ hoặc đã hết hạn");
                } else {
                    responseBuilder.setIsValid(true)
                            .setUserId(user.getId())
                            .setUsername(user.getUsername())
                            .setRole(user.getRole().name())
                            .setDepartmentId(user.getDepartmentId() != null ? user.getDepartmentId() : 0);
                }
            }
        } catch (Exception e) {
            log.warn("Lỗi khi xác thực gRPC token: {}", e.getMessage());
            responseBuilder.setIsValid(false).setErrorMessage("Lỗi xác thực: " + e.getMessage());
        }

        responseObserver.onNext(responseBuilder.build());
        responseObserver.onCompleted();
    }
}