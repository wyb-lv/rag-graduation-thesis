package com.example.accountservice.grpc;

import com.example.accountservice.entity.User;
import com.example.accountservice.service.AccountService;
import io.grpc.stub.StreamObserver;
import lombok.RequiredArgsConstructor;
import net.devh.boot.grpc.server.service.GrpcService;

/**
 * Cổng gRPC cho auth-service. Không có xác thực riêng: ranh giới là mạng nội bộ (không publish cổng gRPC ra ngoài).
 * Tài khoản không có thì AccountService ném AccountNotFoundException, GrpcErrors đổi thành NOT_FOUND.
 */
@GrpcService
@RequiredArgsConstructor
public class AccountGrpcService extends AccountInternalServiceGrpc.AccountInternalServiceImplBase {
    private final AccountService accounts;

    @Override
    public void getCredentials(GetCredentialsRequest req, StreamObserver<Credentials> out) {
        User user = accounts.credentials(req.getUsername());
        reply(out, Credentials.newBuilder().setAccount(toInfo(user)).setPasswordHash(user.getPasswordHash()).build());
    }

    @Override
    public void getAccount(GetAccountRequest req, StreamObserver<AccountInfo> out) {
        reply(out, toInfo(accounts.find(req.getId())));
    }

    @Override
    public void bumpTokenVersion(BumpTokenVersionRequest req, StreamObserver<BumpTokenVersionResponse> out) {
        accounts.bumpTokenVersion(req.getId());
        reply(out, BumpTokenVersionResponse.getDefaultInstance());
    }

    private static <T> void reply(StreamObserver<T> out, T value) {
        out.onNext(value);
        out.onCompleted();
    }

    private static AccountInfo toInfo(User u) {
        return AccountInfo.newBuilder()
                .setId(u.getId())
                .setUsername(u.getUsername())
                .setRole(u.getRole().name())
                .setDepartmentId(u.getDepartmentId())
                .setTokenVersion(u.getTokenVersion())
                .setActive(u.isActive())
                .setMustChangePassword(u.isMustChangePassword())
                .build();
    }
}
