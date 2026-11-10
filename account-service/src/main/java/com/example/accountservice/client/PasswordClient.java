package com.example.accountservice.client;

import com.example.authservice.grpc.EncodeRequest;
import com.example.authservice.grpc.MatchesRequest;
import com.example.authservice.grpc.PasswordServiceGrpc.PasswordServiceBlockingStub;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * Băm và so khớp mật khẩu là việc của auth-service (Spring Security), gọi qua gRPC.
 * Auth-service không trả lời được thì ném 503 (thao tác đổi mật khẩu bị hủy, không lưu gì).
 */
@Component
public class PasswordClient {
    private static final long DEADLINE_SECONDS = 3;

    private final PasswordServiceBlockingStub stub;

    /** Địa chỉ cấu hình ở grpc.client.auth-service.* (application.properties). */
    public PasswordClient(@GrpcClient("auth-service") PasswordServiceBlockingStub stub) {
        this.stub = stub;
    }

    public String encode(String rawPassword) {
        var req = EncodeRequest.newBuilder().setRawPassword(rawPassword).build();
        return call(s -> s.encode(req)).getPasswordHash();
    }

    public boolean matches(String rawPassword, String passwordHash) {
        var req = MatchesRequest.newBuilder().setRawPassword(rawPassword).setPasswordHash(passwordHash).build();
        return call(s -> s.matches(req)).getMatches();
    }

    private <T> T call(Function<PasswordServiceBlockingStub, T> rpc) {
        try {
            return rpc.apply(stub.withDeadlineAfter(DEADLINE_SECONDS, TimeUnit.SECONDS));
        } catch (StatusRuntimeException e) {
            if (e.getStatus().getCode() == Status.Code.INVALID_ARGUMENT) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Password cannot be used (too long?)", e);
            }
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Auth service unavailable", e);
        }
    }
}
