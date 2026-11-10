package com.example.authservice;

import com.example.authservice.grpc.*;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.security.crypto.password.PasswordEncoder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** PasswordService qua kênh gRPC thật (in-process) đúng như account-service sẽ gọi. */
@SpringBootTest
class PasswordGrpcServiceTest {
    @Autowired PasswordEncoder encoder;

    ManagedChannel channel;
    PasswordServiceGrpc.PasswordServiceBlockingStub stub;

    @BeforeEach
    void setUp() {
        channel = InProcessChannelBuilder.forName("auth-test").directExecutor().build();
        stub = PasswordServiceGrpc.newBlockingStub(channel);
    }

    @AfterEach
    void tearDown() {
        channel.shutdownNow();
    }

    private String encode(String raw) {
        return stub.encode(EncodeRequest.newBuilder().setRawPassword(raw).build()).getPasswordHash();
    }

    private boolean matches(String raw, String hash) {
        return stub.matches(MatchesRequest.newBuilder().setRawPassword(raw).setPasswordHash(hash).build()).getMatches();
    }

    @Test
    void encode_producesBCryptHash_thatSpringSecurityAndMatchesAccept() {
        String hash = encode("Secret@123");

        assertThat(hash).startsWith("$2").isNotEqualTo("Secret@123");
        assertThat(encoder.matches("Secret@123", hash)).isTrue(); // đúng hash mà login so khớp
        assertThat(matches("Secret@123", hash)).isTrue();
    }

    @Test
    void matches_wrongPasswordOrGarbageHash_isFalse() {
        String hash = encode("Secret@123");

        assertThat(matches("other", hash)).isFalse();
        assertThat(matches("Secret@123", "not-a-bcrypt-hash")).isFalse();
    }

    @Test
    void encode_sameInputTwice_differentSalts() {
        assertThat(encode("Secret@123")).isNotEqualTo(encode("Secret@123"));
    }

    @Test
    void encode_passwordOver72Bytes_INVALID_ARGUMENT() {
        assertThatThrownBy(() -> encode("x".repeat(80)))
                .satisfies(t -> assertThat(((StatusRuntimeException) t).getStatus().getCode()).isEqualTo(Status.Code.INVALID_ARGUMENT));
    }
}
