package com.example.authservice;

import com.example.accountservice.grpc.*;
import com.example.authservice.client.AccountClient;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** AccountClient với một account-service giả chạy trên kênh gRPC in-process: kiểm tra ánh xạ mã lỗi gRPC. */
class AccountClientTest {
    static final AccountInfo BOB = AccountInfo.newBuilder().setId(1).setUsername("bob").setRole("EMPLOYEE")
            .setDepartmentId(2).setTokenVersion(3).setActive(true).setMustChangePassword(true).build();

    static final Credentials BOB_CREDENTIALS = Credentials.newBuilder().setAccount(BOB).setPasswordHash("$2a$hash").build();

    /** username "bob" tồn tại, "boom" gây lỗi INTERNAL; id 1 tồn tại, id 500 gây lỗi INTERNAL; còn lại không có. */
    static class FakeAccountService extends AccountInternalServiceGrpc.AccountInternalServiceImplBase {
        @Override
        public void getCredentials(GetCredentialsRequest r, StreamObserver<Credentials> out) {
            switch (r.getUsername()) {
                case "bob" -> done(out, BOB_CREDENTIALS);
                case "boom" -> out.onError(Status.INTERNAL.asRuntimeException());
                default -> out.onError(Status.NOT_FOUND.asRuntimeException());
            }
        }

        @Override
        public void getAccount(GetAccountRequest r, StreamObserver<AccountInfo> out) {
            if (r.getId() == 1) done(out, BOB);
            else if (r.getId() == 500) out.onError(Status.INTERNAL.asRuntimeException());
            else out.onError(Status.NOT_FOUND.asRuntimeException());
        }

        @Override
        public void bumpTokenVersion(BumpTokenVersionRequest r, StreamObserver<BumpTokenVersionResponse> out) {
            if (r.getId() == 1) done(out, BumpTokenVersionResponse.getDefaultInstance());
            else out.onError(Status.NOT_FOUND.asRuntimeException());
        }

        private static <T> void done(StreamObserver<T> out, T value) {
            out.onNext(value);
            out.onCompleted();
        }
    }

    Server server;
    ManagedChannel channel;
    AccountClient client;

    @BeforeEach
    void setUp() throws Exception {
        String name = "fake-account-" + System.nanoTime();
        server = InProcessServerBuilder.forName(name).directExecutor().addService(new FakeAccountService()).build().start();
        channel = InProcessChannelBuilder.forName(name).directExecutor().build();
        client = new AccountClient(AccountInternalServiceGrpc.newBlockingStub(channel));
    }

    @AfterEach
    void tearDown() {
        channel.shutdownNow();
        server.shutdownNow();
    }

    private static void assertUnavailable(Throwable t) {
        assertThat(t).isInstanceOf(ResponseStatusException.class);
        assertThat(((ResponseStatusException) t).getStatusCode()).isEqualTo(HttpStatus.SERVICE_UNAVAILABLE);
    }

    @Test
    void credentials_ok_returnsAccountAndHash() {
        assertThat(client.credentials("bob")).contains(BOB_CREDENTIALS);
    }

    @Test
    void credentials_unknownUser_isEmpty_notAnError() {
        assertThat(client.credentials("ghost")).isEmpty();
    }

    @Test
    void find_existingAndMissing() {
        assertThat(client.find(1)).contains(BOB);
        assertThat(client.find(2)).isEmpty();
    }

    @Test
    void bumpTokenVersion_ok_andUnknownAccountIsIgnored() {
        client.bumpTokenVersion(1);
        client.bumpTokenVersion(2);
    }

    @Test
    void serverError_becomes503_notWrongPassword() {
        assertThatThrownBy(() -> client.credentials("boom")).satisfies(AccountClientTest::assertUnavailable);
        assertThatThrownBy(() -> client.find(500)).satisfies(AccountClientTest::assertUnavailable);
    }

    @Test
    void accountServiceDown_becomes503() {
        server.shutdownNow();

        assertThatThrownBy(() -> client.credentials("bob")).satisfies(AccountClientTest::assertUnavailable);
        assertThatThrownBy(() -> client.find(1)).satisfies(AccountClientTest::assertUnavailable);
        assertThatThrownBy(() -> client.bumpTokenVersion(1)).satisfies(AccountClientTest::assertUnavailable);
    }
}
