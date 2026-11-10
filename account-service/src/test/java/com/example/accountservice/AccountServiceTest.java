package com.example.accountservice;

import com.example.accountservice.client.PasswordClient;
import com.example.accountservice.dto.ChangePasswordRequest;
import com.example.accountservice.dto.CreateAccountRequest;
import com.example.accountservice.dto.UpdateAccountRequest;
import com.example.accountservice.entity.User;
import com.example.accountservice.entity.UserRole;
import com.example.accountservice.repository.UserRepository;
import com.example.accountservice.service.AccountService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class AccountServiceTest {
    @Mock UserRepository users;
    @Mock PasswordClient passwords;
    AccountService service;

    @BeforeEach
    void setUp() {
        service = new AccountService(users, passwords);
    }

    private static CreateAccountRequest createReq() {
        return new CreateAccountRequest("alice", "alice@hsu.edu.vn", 1L, "Giảng viên", UserRole.EMPLOYEE, "Temp@1234");
    }

    private User existing(long id) {
        return User.builder().id(id).username("bob").email("bob@hsu.edu.vn").passwordHash("hash(Old@12345)")
                .departmentId(1L).role(UserRole.EMPLOYEE).tokenVersion(3).build();
    }

    private static void assertStatus(Throwable t, HttpStatus expected) {
        assertThat(t).isInstanceOf(ResponseStatusException.class);
        assertThat(((ResponseStatusException) t).getStatusCode()).isEqualTo(expected);
    }

    // ── Admin ──────────────────────────────────────────────────────────────────

    @Test
    void create_hashesTempPasswordAndForcesChange() {
        when(passwords.encode("Temp@1234")).thenReturn("hash(Temp@1234)");
        when(users.saveAndFlush(any(User.class))).thenAnswer(i -> i.getArgument(0));

        var res = service.create(createReq());

        assertThat(res.username()).isEqualTo("alice");
        assertThat(res.mustChangePassword()).isTrue();
        assertThat(res.active()).isTrue();
        var saved = ArgumentCaptor.forClass(User.class);
        verify(users).saveAndFlush(saved.capture());
        assertThat(saved.getValue().getPasswordHash()).isEqualTo("hash(Temp@1234)"); // hash do auth-service băm, không lưu mật khẩu thô
    }

    @Test
    void create_duplicateUsername_409() {
        when(users.existsByUsername("alice")).thenReturn(true);

        assertThatThrownBy(() -> service.create(createReq())).satisfies(t -> assertStatus(t, HttpStatus.CONFLICT));
        verify(users, never()).saveAndFlush(any());
    }

    @Test
    void create_duplicateEmail_409() {
        when(users.existsByEmailIgnoreCase("alice@hsu.edu.vn")).thenReturn(true);

        assertThatThrownBy(() -> service.create(createReq())).satisfies(t -> assertStatus(t, HttpStatus.CONFLICT));
        verify(users, never()).saveAndFlush(any());
    }

    @Test
    void create_authServiceDown_503_andNothingSaved() {
        when(passwords.encode("Temp@1234")).thenThrow(new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Auth service unavailable"));

        assertThatThrownBy(() -> service.create(createReq())).satisfies(t -> assertStatus(t, HttpStatus.SERVICE_UNAVAILABLE));
        verify(users, never()).saveAndFlush(any());
    }

    @Test
    void create_raceOnUniqueConstraint_409() {
        when(users.saveAndFlush(any(User.class))).thenThrow(new DataIntegrityViolationException("uk"));

        assertThatThrownBy(() -> service.create(createReq())).satisfies(t -> assertStatus(t, HttpStatus.CONFLICT));
    }

    @Test
    void update_changesFieldsButNotUsername() {
        var user = existing(2);
        when(users.findById(2L)).thenReturn(Optional.of(user));
        when(users.saveAndFlush(user)).thenReturn(user);

        var res = service.update(2, new UpdateAccountRequest("new@hsu.edu.vn", 9L, "Trưởng khoa", UserRole.MANAGER));

        assertThat(res.username()).isEqualTo("bob");
        assertThat(res.email()).isEqualTo("new@hsu.edu.vn");
        assertThat(res.departmentId()).isEqualTo(9L);
        assertThat(res.position()).isEqualTo("Trưởng khoa");
        assertThat(res.role()).isEqualTo(UserRole.MANAGER);
    }

    @Test
    void update_emailTakenByAnotherUser_409() {
        when(users.findById(2L)).thenReturn(Optional.of(existing(2)));
        when(users.existsByEmailIgnoreCaseAndIdNot("taken@hsu.edu.vn", 2L)).thenReturn(true);

        assertThatThrownBy(() -> service.update(2, new UpdateAccountRequest("taken@hsu.edu.vn", 1L, null, UserRole.EMPLOYEE)))
                .satisfies(t -> assertStatus(t, HttpStatus.CONFLICT));
    }

    @Test
    void update_sameEmailIgnoringCase_skipsTheUniquenessQuery() {
        var user = existing(2);
        when(users.findById(2L)).thenReturn(Optional.of(user));
        when(users.saveAndFlush(user)).thenReturn(user);

        service.update(2, new UpdateAccountRequest("BOB@hsu.edu.vn", 1L, null, UserRole.EMPLOYEE));

        verify(users, never()).existsByEmailIgnoreCaseAndIdNot(any(), anyLong());
    }

    @Test
    void update_unknownId_404() {
        when(users.findById(99L)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.update(99, new UpdateAccountRequest("a@b.c", 1L, null, UserRole.EMPLOYEE)))
                .satisfies(t -> assertStatus(t, HttpStatus.NOT_FOUND));
    }

    @Test
    void deactivate_bumpsTokenVersion() {
        var user = existing(2);
        when(users.findById(2L)).thenReturn(Optional.of(user));
        when(users.saveAndFlush(user)).thenReturn(user);

        var res = service.setActive(2, false, 1);

        assertThat(res.active()).isFalse();
        assertThat(user.getTokenVersion()).isEqualTo(4);
    }

    @Test
    void reactivate_alsoBumpsTokenVersion() {
        var user = existing(2);
        user.setActive(false);
        when(users.findById(2L)).thenReturn(Optional.of(user));
        when(users.saveAndFlush(user)).thenReturn(user);

        var res = service.setActive(2, true, 1);

        assertThat(res.active()).isTrue();
        assertThat(user.getTokenVersion()).isEqualTo(4);
    }

    @Test
    void deactivate_self_400AndNothingChanges() {
        assertThatThrownBy(() -> service.setActive(1, false, 1)).satisfies(t -> assertStatus(t, HttpStatus.BAD_REQUEST));
        verify(users, never()).saveAndFlush(any());
    }

    @Test
    void resetPassword_writesTheHashFromAuthServiceForcesChangeAndBumpsOnce() {
        when(passwords.encode("NewTemp@99")).thenReturn("hash(NewTemp@99)");
        when(users.updatePassword(2, "hash(NewTemp@99)", true, 1)).thenReturn(1);

        service.resetPassword(2, "NewTemp@99");

        verify(users).updatePassword(2, "hash(NewTemp@99)", true, 1);
    }

    @Test
    void resetPassword_unknownId_404() {
        when(passwords.encode("NewTemp@99")).thenReturn("hash(NewTemp@99)");
        when(users.updatePassword(99, "hash(NewTemp@99)", true, 1)).thenReturn(0);

        assertThatThrownBy(() -> service.resetPassword(99, "NewTemp@99")).satisfies(t -> assertStatus(t, HttpStatus.NOT_FOUND));
    }

    @Test
    void resetPassword_authServiceDown_503_andNothingWritten() {
        when(passwords.encode("NewTemp@99"))
                .thenThrow(new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE, "Auth service unavailable"));

        assertThatThrownBy(() -> service.resetPassword(2, "NewTemp@99")).satisfies(t -> assertStatus(t, HttpStatus.SERVICE_UNAVAILABLE));
        verify(users, never()).updatePassword(anyLong(), any(), anyBoolean(), anyInt());
    }

    // ── Tự phục vụ ─────────────────────────────────────────────────────────────

    @Test
    void changePassword_writesNewHashAndClearsFlagWithoutBumping() {
        when(users.findById(2L)).thenReturn(Optional.of(existing(2)));
        when(passwords.matches("Old@12345", "hash(Old@12345)")).thenReturn(true);
        when(passwords.encode("Brand@New1")).thenReturn("hash(Brand@New1)");

        service.changePassword(2, new ChangePasswordRequest("Old@12345", "Brand@New1"));

        verify(users).updatePassword(2, "hash(Brand@New1)", false, 0);
    }

    @Test
    void changePassword_wrongCurrentPassword_400_notChanged() {
        when(users.findById(2L)).thenReturn(Optional.of(existing(2)));
        when(passwords.matches("wrong", "hash(Old@12345)")).thenReturn(false);

        assertThatThrownBy(() -> service.changePassword(2, new ChangePasswordRequest("wrong", "Brand@New1")))
                .satisfies(t -> assertStatus(t, HttpStatus.BAD_REQUEST));
        verify(users, never()).updatePassword(anyLong(), any(), anyBoolean(), anyInt());
        verify(passwords, never()).encode(any());
    }

    // ── Cho auth-service ───────────────────────────────────────────────────────

    @Test
    void credentials_returnsTheStoredUserWithoutVerifyingPassword_evenWhenInactive() {
        var user = existing(2);
        user.setActive(false); // auth-service tự quyết định (403), nên tài khoản bị khóa vẫn được trả về
        when(users.findByUsername("bob")).thenReturn(Optional.of(user));

        assertThat(service.credentials("bob")).isSameAs(user);
        verifyNoInteractions(passwords);
    }

    @Test
    void credentials_unknownUser_404() {
        when(users.findByUsername("ghost")).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.credentials("ghost")).satisfies(t -> assertStatus(t, HttpStatus.NOT_FOUND));
    }

    @Test
    void find_unknownId_404() {
        when(users.findById(99L)).thenReturn(Optional.empty());

        assertThatThrownBy(() -> service.find(99)).satisfies(t -> assertStatus(t, HttpStatus.NOT_FOUND));
    }

    @Test
    void bumpTokenVersion_isOneAtomicUpdate_andUnknownIdIs404() {
        when(users.bumpTokenVersion(2L)).thenReturn(1);
        when(users.bumpTokenVersion(99L)).thenReturn(0);

        service.bumpTokenVersion(2);

        verify(users).bumpTokenVersion(2L);
        assertThatThrownBy(() -> service.bumpTokenVersion(99)).satisfies(t -> assertStatus(t, HttpStatus.NOT_FOUND));
    }
}
