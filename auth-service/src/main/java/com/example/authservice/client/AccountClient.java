package com.example.authservice.client;

import com.example.accountservice.grpc.*;
import com.example.accountservice.grpc.AccountInternalServiceGrpc.AccountInternalServiceBlockingStub;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import net.devh.boot.grpc.client.inject.GrpcClient;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * Cửa duy nhất sang account-service (gRPC). Tài khoản có/không có thì trả Optional; account-service không trả lời được
 * thì ném 503 để lớp trên không nhầm thành "sai mật khẩu" hay "token hết hạn" (FE sẽ đăng xuất người dùng nếu nhận 401).
 */
@Component
public class AccountClient {
    private static final long DEADLINE_SECONDS = 3;

    private final AccountInternalServiceBlockingStub stub;

    /** Địa chỉ cấu hình ở grpc.client.account-service.* (application.properties). */
    public AccountClient(@GrpcClient("account-service") AccountInternalServiceBlockingStub stub) {
        this.stub = stub;
    }

    /** Tài khoản + mật khẩu đã băm (auth-service tự so khớp). Empty nếu không có username. Tài khoản bị khóa vẫn được trả về. */
    public Optional<Credentials> credentials(String username) {
        var req = GetCredentialsRequest.newBuilder().setUsername(username).build();
        return call(s -> s.getCredentials(req));
    }

    public Optional<AccountInfo> find(long id) {
        var req = GetAccountRequest.newBuilder().setId(id).build();
        return call(s -> s.getAccount(req));
    }

    /** Hủy mọi phiên của người dùng. Tài khoản đã bị xóa thì bỏ qua. */
    public void bumpTokenVersion(long id) {
        var req = BumpTokenVersionRequest.newBuilder().setId(id).build();
        call(s -> s.bumpTokenVersion(req));
    }

    /** Empty khi account-service báo NOT_FOUND; mọi lỗi khác (không kết nối được, quá hạn, lỗi nội bộ) là 503. */
    private <T> Optional<T> call(Function<AccountInternalServiceBlockingStub, T> rpc) {
        try {
            return Optional.of(rpc.apply(stub.withDeadlineAfter(DEADLINE_SECONDS, TimeUnit.SECONDS)));
        } catch (StatusRuntimeException e) {
            if (e.getStatus().getCode() == Status.Code.NOT_FOUND) return Optional.empty();
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Account service unavailable", e);
        }
    }
}
