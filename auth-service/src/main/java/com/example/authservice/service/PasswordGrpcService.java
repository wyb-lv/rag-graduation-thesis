package com.example.authservice.service;

import com.example.authservice.grpc.EncodeRequest;
import com.example.authservice.grpc.EncodeResponse;
import com.example.authservice.grpc.MatchesRequest;
import com.example.authservice.grpc.MatchesResponse;
import com.example.authservice.grpc.PasswordServiceGrpc.PasswordServiceImplBase;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import lombok.RequiredArgsConstructor;
import net.devh.boot.grpc.server.service.GrpcService;
import org.springframework.security.crypto.password.PasswordEncoder;

/** Cho account-service băm và so khớp mật khẩu bằng PasswordEncoder của Spring Security. Chỉ gọi từ mạng nội bộ. */
@GrpcService
@RequiredArgsConstructor
public class PasswordGrpcService extends PasswordServiceImplBase {
    private final PasswordEncoder passwordEncoder;

    @Override
    public void encode(EncodeRequest request, StreamObserver<EncodeResponse> out) {
        try {
            out.onNext(EncodeResponse.newBuilder().setPasswordHash(passwordEncoder.encode(request.getRawPassword())).build());
            out.onCompleted();
        } catch (IllegalArgumentException e) { // BCrypt từ chối mật khẩu quá 72 byte
            out.onError(Status.INVALID_ARGUMENT.withDescription(e.getMessage()).asRuntimeException());
        }
    }

    @Override
    public void matches(MatchesRequest request, StreamObserver<MatchesResponse> out) {
        out.onNext(MatchesResponse.newBuilder()
                .setMatches(passwordEncoder.matches(request.getRawPassword(), request.getPasswordHash()))
                .build());
        out.onCompleted();
    }
}
