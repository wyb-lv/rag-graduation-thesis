package com.example.authservice.service;

import com.example.accountservice.grpc.AccountInfo;
import com.example.authservice.client.AccountClient;
import com.example.authservice.grpc.TokenValidationRequest;
import com.example.authservice.grpc.TokenValidationResponse;
import com.example.authservice.grpc.TokenValidationServiceGrpc.TokenValidationServiceImplBase;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import io.jsonwebtoken.Claims;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.devh.boot.grpc.server.service.GrpcService;
import org.springframework.web.server.ResponseStatusException;

import java.util.Optional;

@Slf4j
@GrpcService
@RequiredArgsConstructor
public class AuthGrpcService extends TokenValidationServiceImplBase {

    private final JwtService jwtService;
    private final AccountClient accounts;

    @Override
    public void validateToken(TokenValidationRequest request, StreamObserver<TokenValidationResponse> responseObserver) {
        String token = request.getToken();
        TokenValidationResponse.Builder responseBuilder = TokenValidationResponse.newBuilder();

        try {
            Claims claims = jwtService.parse(token);
            Long userId = claims.get("userId", Long.class);
            Optional<AccountInfo> userOpt = userId == null ? Optional.empty() : accounts.find(userId);

            if (userOpt.isEmpty()) {
                responseBuilder.setIsValid(false).setErrorMessage("Người dùng không tồn tại");
            } else {
                AccountInfo user = userOpt.get();

                if (!user.getActive()) {
                    responseBuilder.setIsValid(false).setErrorMessage("Tài khoản đã bị vô hiệu hóa");
                } else if (!jwtService.isTokenValid(claims, user)) {
                    responseBuilder.setIsValid(false).setErrorMessage("Token không hợp lệ hoặc đã hết hạn");
                } else {
                    responseBuilder.setIsValid(true)
                            .setUserId(user.getId())
                            .setUsername(user.getUsername())
                            .setRole(user.getRole())
                            .setDepartmentId(user.getDepartmentId());
                }
            }
        } catch (ResponseStatusException e) {
            // Account-service không trả lời: báo UNAVAILABLE để gateway trả 503, không phải 401 (FE sẽ đăng xuất người dùng)
            log.warn("Không gọi được account-service để xác thực token: {}", e.getReason());
            responseObserver.onError(Status.UNAVAILABLE.withDescription(e.getReason()).asRuntimeException());
            return;
        } catch (Exception e) {
            log.warn("Lỗi khi xác thực gRPC token: {}", e.getMessage());
            responseBuilder.setIsValid(false).setErrorMessage("Lỗi xác thực: " + e.getMessage());
        }

        responseObserver.onNext(responseBuilder.build());
        responseObserver.onCompleted();
    }
}
