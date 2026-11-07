package com.example.accountservice;

import com.example.accountservice.client.PasswordClient;
import com.example.authservice.grpc.*;
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

/** PasswordClient với một auth-service giả trên kênh gRPC in-process: kiểm tra ánh xạ mã lỗi gRPC. */
class PasswordClientTest {
    /** Encode "toolong" gây INVALID_ARGUMENT, "boom" gây INTERNAL; còn lại hash là "h(" + mật khẩu + ")". */
    static class FakePasswordService extends PasswordServiceGrpc.PasswordServiceImplBase {
        @Override
        public void encode(EncodeRequest r, StreamObserver<EncodeResponse> out) {
            switch (r.getRawPassword()) {
                case "toolong" -> out.onError(Status.INVALID_ARGUMENT.asRuntimeException());
                case "boom" -> out.onError(Status.INTERNAL.asRuntimeException());
                default -> {
                    out.onNext(EncodeResponse.newBuilder().setPasswordHash("h(" + r.getRawPassword() + ")").build());
                    out.onCompleted();
                }
            }
        }

        @Override
        public void matches(MatchesRequest r, StreamObserver<MatchesResponse> out) {
            out.onNext(MatchesResponse.newBuilder().setMatches(("h(" + r.getRawPassword() + ")").equals(r.getPasswordHash())).build());
            out.onCompleted();
        }
    }

    Server server;
    ManagedChannel channel;
    PasswordClient client;

    @BeforeEach
    void setUp() throws Exception {
        String name = "fake-auth-" + System.nanoTime();
        server = InProcessServerBuilder.forName(name).directExecutor().addService(new FakePasswordService()).build().start();
        channel = InProcessChannelBuilder.forName(name).directExecutor().build();
        client = new PasswordClient(PasswordServiceGrpc.newBlockingStub(channel));
    }

    @AfterEach
    void tearDown() {
        channel.shutdownNow();
        server.shutdownNow();
    }

    private static void assertStatus(Throwable t, HttpStatus expected) {
        assertThat(t).isInstanceOf(ResponseStatusException.class);
        assertThat(((ResponseStatusException) t).getStatusCode()).isEqualTo(expected);
    }

    @Test
    void encodeAndMatches_roundTrip() {
        String hash = client.encode("Secret@123");

        assertThat(hash).isEqualTo("h(Secret@123)");
        assertThat(client.matches("Secret@123", hash)).isTrue();
        assertThat(client.matches("other", hash)).isFalse();
    }

    @Test
    void invalidArgument_becomes400() {
        assertThatThrownBy(() -> client.encode("toolong")).satisfies(t -> assertStatus(t, HttpStatus.BAD_REQUEST));
    }

    @Test
    void serverError_becomes503() {
        assertThatThrownBy(() -> client.encode("boom")).satisfies(t -> assertStatus(t, HttpStatus.SERVICE_UNAVAILABLE));
    }

    @Test
    void authServiceDown_becomes503() {
        server.shutdownNow();

        assertThatThrownBy(() -> client.encode("x")).satisfies(t -> assertStatus(t, HttpStatus.SERVICE_UNAVAILABLE));
        assertThatThrownBy(() -> client.matches("x", "h(x)")).satisfies(t -> assertStatus(t, HttpStatus.SERVICE_UNAVAILABLE));
    }
}
