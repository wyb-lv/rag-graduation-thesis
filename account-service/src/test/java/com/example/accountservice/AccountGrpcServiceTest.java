package com.example.accountservice;

import com.example.accountservice.entity.User;
import com.example.accountservice.entity.UserRole;
import com.example.accountservice.grpc.*;
import com.example.accountservice.repository.UserRepository;
import io.grpc.ManagedChannel;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.inprocess.InProcessChannelBuilder;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/** Gọi AccountInternalService qua kênh gRPC thật (in-process) đúng như auth-service sẽ gọi. */
@SpringBootTest
class AccountGrpcServiceTest {
    @Autowired UserRepository users;

    ManagedChannel channel;
    AccountInternalServiceGrpc.AccountInternalServiceBlockingStub stub;
    long id;

    @BeforeEach
    void setUp() {
        users.deleteAll();
        id = users.save(User.builder().username("bob").email("bob@hsu.edu.vn").passwordHash("enc(Old@12345)")
                .departmentId(4L).role(UserRole.MANAGER).mustChangePassword(true).build()).getId();
        channel = InProcessChannelBuilder.forName("account-test").directExecutor().build();
        stub = AccountInternalServiceGrpc.newBlockingStub(channel);
    }

    @AfterEach
    void tearDown() {
        channel.shutdownNow();
    }

    private static Status.Code codeOf(Throwable t) {
        return ((StatusRuntimeException) t).getStatus().getCode();
    }

    @Test
    void getCredentials_returnsAccountInfoAndStoredHash() {
        Credentials c = stub.getCredentials(GetCredentialsRequest.newBuilder().setUsername("bob").build());

        assertThat(c.getPasswordHash()).isEqualTo("enc(Old@12345)");
        AccountInfo info = c.getAccount();
        assertThat(info.getId()).isEqualTo(id);
        assertThat(info.getUsername()).isEqualTo("bob");
        assertThat(info.getRole()).isEqualTo("MANAGER");
        assertThat(info.getDepartmentId()).isEqualTo(4);
        assertThat(info.getTokenVersion()).isZero();
        assertThat(info.getActive()).isTrue();
        assertThat(info.getMustChangePassword()).isTrue();
    }

    @Test
    void getCredentials_unknownUser_NOT_FOUND() {
        assertThatThrownBy(() -> stub.getCredentials(GetCredentialsRequest.newBuilder().setUsername("ghost").build()))
                .satisfies(t -> assertThat(codeOf(t)).isEqualTo(Status.Code.NOT_FOUND));
    }

    @Test
    void getCredentials_inactiveAccount_isStillReturned() {
        User u = users.findById(id).orElseThrow();
        u.setActive(false);
        users.save(u);

        assertThat(stub.getCredentials(GetCredentialsRequest.newBuilder().setUsername("bob").build()).getAccount().getActive()).isFalse();
    }

    @Test
    void getAccount_reflectsLatestStateAndUnknownIsNotFound() {
        assertThat(stub.getAccount(GetAccountRequest.newBuilder().setId(id).build()).getUsername()).isEqualTo("bob");

        assertThatThrownBy(() -> stub.getAccount(GetAccountRequest.newBuilder().setId(99999).build()))
                .satisfies(t -> assertThat(codeOf(t)).isEqualTo(Status.Code.NOT_FOUND));
    }

    @Test
    void bumpTokenVersion_incrementsAndIsVisibleToGetAccount() {
        stub.bumpTokenVersion(BumpTokenVersionRequest.newBuilder().setId(id).build());

        assertThat(stub.getAccount(GetAccountRequest.newBuilder().setId(id).build()).getTokenVersion()).isEqualTo(1);
        assertThat(users.findById(id).orElseThrow().getTokenVersion()).isEqualTo(1);
    }

    @Test
    void bumpTokenVersion_unknownAccount_NOT_FOUND() {
        assertThatThrownBy(() -> stub.bumpTokenVersion(BumpTokenVersionRequest.newBuilder().setId(99999).build()))
                .satisfies(t -> assertThat(codeOf(t)).isEqualTo(Status.Code.NOT_FOUND));
    }
}
