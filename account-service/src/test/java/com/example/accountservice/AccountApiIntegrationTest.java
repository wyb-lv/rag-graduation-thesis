package com.example.accountservice;

import com.example.accountservice.client.PasswordClient;
import com.example.accountservice.entity.User;
import com.example.accountservice.entity.UserRole;
import com.example.accountservice.repository.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webmvc.test.autoconfigure.AutoConfigureMockMvc;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.request.MockHttpServletRequestBuilder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/** Chạy toàn bộ stack (filter Admin → controller → service → JPA/H2). Header X-User-* giả lập những gì gateway gắn. */
@SpringBootTest
@AutoConfigureMockMvc
class AccountApiIntegrationTest {
    @Autowired MockMvc mvc;
    @Autowired UserRepository users;
    // auth-service (băm mật khẩu) được giả: "enc(x)" là hash của mật khẩu x
    @MockitoBean PasswordClient passwords;

    long adminId;

    @BeforeEach
    void seed() {
        when(passwords.encode(anyString())).thenAnswer(i -> "enc(" + i.getArgument(0) + ")");
        when(passwords.matches(anyString(), anyString())).thenAnswer(i -> ("enc(" + i.getArgument(0) + ")").equals(i.getArgument(1)));
        users.deleteAll();
        adminId = users.save(user("admin", "admin@hsu.edu.vn", 1L, UserRole.ADMIN, true)).getId();
    }

    private User user(String username, String email, long dept, UserRole role, boolean active) {
        return User.builder().username(username).email(email).passwordHash("enc(Old@12345)")
                .departmentId(dept).role(role).isActive(active).build();
    }

    /** Gắn danh tính như gateway sau khi xác thực token. */
    private static MockHttpServletRequestBuilder as(String role, long userId, MockHttpServletRequestBuilder b) {
        return b.header("X-User-Id", userId).header("X-User-Role", role).contentType(MediaType.APPLICATION_JSON);
    }

    private MockHttpServletRequestBuilder asAdmin(MockHttpServletRequestBuilder b) {
        return as("ADMIN", adminId, b);
    }

    private static String createBody(String username, String email) {
        return """
                {"username":"%s","email":"%s","departmentId":2,"position":"Giảng viên","role":"EMPLOYEE","tempPassword":"Temp@1234"}
                """.formatted(username, email);
    }

    @Test
    void openApi_documentsEveryAccountOperation() throws Exception {
        mvc.perform(get("/v3/api-docs"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.paths['/api/accounts'].get").exists())
                .andExpect(jsonPath("$.paths['/api/accounts'].post").exists())
                .andExpect(jsonPath("$.paths['/api/accounts/{id}'].put").exists())
                .andExpect(jsonPath("$.paths['/api/accounts/{id}/status'].patch").exists())
                .andExpect(jsonPath("$.paths['/api/accounts/{id}/reset-password'].post").exists());
    }

    // ── Chỉ Admin ──────────────────────────────────────────────────────────────

    @Test
    void withoutIdentity_401() throws Exception {
        mvc.perform(get("/api/accounts")).andExpect(status().isUnauthorized());
    }

    @Test
    void nonAdminRoles_403_onEveryEndpoint() throws Exception {
        for (String role : new String[]{"MANAGER", "EMPLOYEE"}) {
            for (var req : new MockHttpServletRequestBuilder[]{
                    get("/api/accounts"),
                    post("/api/accounts").content(createBody("x", "x@hsu.edu.vn")),
                    put("/api/accounts/1").content("{}"),
                    patch("/api/accounts/1/status").content("{\"active\":false}"),
                    post("/api/accounts/1/reset-password").content("{\"tempPassword\":\"Temp@1234\"}")}) {
                mvc.perform(as(role, 5, req))
                        .andExpect(status().isForbidden());
            }
        }
    }

    @Test
    void pathVariantsWithoutIdentityNeverReachTheControllers() throws Exception {
        for (String path : new String[]{"/api/accounts/", "/api/accounts;x=1", "//api/accounts", "/api/accounts/me/", "/api/accounts/nope/deeper"}) {
            mvc.perform(get(path)).andExpect(status().is4xxClientError());
        }
        mvc.perform(get("/api/accounts/nope/deeper").header("X-User-Role", "EMPLOYEE")).andExpect(status().isForbidden());
    }

    // ── Tạo tài khoản ──────────────────────────────────────────────────────────

    @Test
    void create_returns201_hidesPassword_andStoresBcryptTempPassword() throws Exception {
        mvc.perform(asAdmin(post("/api/accounts")).content(createBody("alice", "alice@hsu.edu.vn")))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.username").value("alice"))
                .andExpect(jsonPath("$.position").value("Giảng viên"))
                .andExpect(jsonPath("$.role").value("EMPLOYEE"))
                .andExpect(jsonPath("$.active").value(true))
                .andExpect(jsonPath("$.mustChangePassword").value(true))
                .andExpect(jsonPath("$.passwordHash").doesNotExist())
                .andExpect(jsonPath("$.tempPassword").doesNotExist());

        User saved = users.findAll().stream().filter(u -> u.getUsername().equals("alice")).findFirst().orElseThrow();
        assertThat(saved.getPasswordHash()).isEqualTo("enc(Temp@1234)");
        assertThat(saved.isMustChangePassword()).isTrue();
    }

    @Test
    void create_duplicateUsername_409() throws Exception {
        mvc.perform(asAdmin(post("/api/accounts")).content(createBody("admin", "other@hsu.edu.vn")))
                .andExpect(status().isConflict());
    }

    @Test
    void create_duplicateEmail_409_evenWithDifferentCase() throws Exception {
        mvc.perform(asAdmin(post("/api/accounts")).content(createBody("other", "ADMIN@hsu.edu.vn")))
                .andExpect(status().isConflict());
    }

    @Test
    void create_invalidBody_400() throws Exception {
        mvc.perform(asAdmin(post("/api/accounts")).content("""
                {"username":"a b","email":"not-an-email","departmentId":2,"role":"EMPLOYEE","tempPassword":"short"}"""))
                .andExpect(status().isBadRequest());
        mvc.perform(asAdmin(post("/api/accounts")).content("""
                {"username":"bad_role","email":"r@hsu.edu.vn","departmentId":2,"role":"ROOT","tempPassword":"Temp@1234"}"""))
                .andExpect(status().isBadRequest());
    }

    // ── Danh sách ──────────────────────────────────────────────────────────────

    private void seedMany() {
        users.save(user("emp1", "emp1@hsu.edu.vn", 1L, UserRole.EMPLOYEE, true));
        users.save(user("emp2", "emp2@hsu.edu.vn", 2L, UserRole.EMPLOYEE, true));
        users.save(user("emp3", "emp3@hsu.edu.vn", 2L, UserRole.EMPLOYEE, false));
        users.save(user("mgr", "mgr@hsu.edu.vn", 2L, UserRole.MANAGER, true));
    }

    @Test
    void list_paginates() throws Exception {
        seedMany(); // + admin = 5 tài khoản
        mvc.perform(asAdmin(get("/api/accounts")).param("size", "2").param("page", "1"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.content.length()").value(2))
                .andExpect(jsonPath("$.page").value(1))
                .andExpect(jsonPath("$.size").value(2))
                .andExpect(jsonPath("$.totalElements").value(5))
                .andExpect(jsonPath("$.totalPages").value(3))
                .andExpect(jsonPath("$.content[0].passwordHash").doesNotExist());
    }

    @Test
    void list_filtersByDepartmentRoleAndStatus() throws Exception {
        seedMany();
        mvc.perform(asAdmin(get("/api/accounts")).param("departmentId", "2"))
                .andExpect(jsonPath("$.totalElements").value(3));
        mvc.perform(asAdmin(get("/api/accounts")).param("role", "MANAGER"))
                .andExpect(jsonPath("$.totalElements").value(1))
                .andExpect(jsonPath("$.content[0].username").value("mgr"));
        mvc.perform(asAdmin(get("/api/accounts")).param("active", "false"))
                .andExpect(jsonPath("$.totalElements").value(1))
                .andExpect(jsonPath("$.content[0].username").value("emp3"));
        mvc.perform(asAdmin(get("/api/accounts")).param("departmentId", "2").param("role", "EMPLOYEE").param("active", "true"))
                .andExpect(jsonPath("$.totalElements").value(1))
                .andExpect(jsonPath("$.content[0].username").value("emp2"));
    }

    @Test
    void list_rejectsBadPaging_400() throws Exception {
        mvc.perform(asAdmin(get("/api/accounts")).param("size", "101")).andExpect(status().isBadRequest());
        mvc.perform(asAdmin(get("/api/accounts")).param("page", "-1")).andExpect(status().isBadRequest());
    }

    // ── Sửa ────────────────────────────────────────────────────────────────────

    @Test
    void update_changesEditableFields() throws Exception {
        long id = users.save(user("emp1", "emp1@hsu.edu.vn", 1L, UserRole.EMPLOYEE, true)).getId();

        mvc.perform(asAdmin(put("/api/accounts/" + id)).content("""
                {"email":"new@hsu.edu.vn","departmentId":7,"position":"Trưởng khoa","role":"MANAGER"}"""))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.username").value("emp1"))
                .andExpect(jsonPath("$.email").value("new@hsu.edu.vn"))
                .andExpect(jsonPath("$.departmentId").value(7))
                .andExpect(jsonPath("$.position").value("Trưởng khoa"))
                .andExpect(jsonPath("$.role").value("MANAGER"));
    }

    @Test
    void update_keepingOwnEmail_isNotConflict() throws Exception {
        long id = users.save(user("emp1", "emp1@hsu.edu.vn", 1L, UserRole.EMPLOYEE, true)).getId();

        mvc.perform(asAdmin(put("/api/accounts/" + id)).content("""
                {"email":"emp1@hsu.edu.vn","departmentId":3,"role":"EMPLOYEE"}""")).andExpect(status().isOk());
    }

    @Test
    void update_emailOfAnotherUser_409() throws Exception {
        long id = users.save(user("emp1", "emp1@hsu.edu.vn", 1L, UserRole.EMPLOYEE, true)).getId();

        mvc.perform(asAdmin(put("/api/accounts/" + id)).content("""
                {"email":"admin@hsu.edu.vn","departmentId":1,"role":"EMPLOYEE"}""")).andExpect(status().isConflict());
    }

    @Test
    void update_unknownId_404() throws Exception {
        mvc.perform(asAdmin(put("/api/accounts/99999")).content("""
                {"email":"a@hsu.edu.vn","departmentId":1,"role":"EMPLOYEE"}""")).andExpect(status().isNotFound());
    }

    // ── Vô hiệu hóa / kích hoạt ────────────────────────────────────────────────

    private long newEmployee() {
        return users.save(user("emp1", "emp1@hsu.edu.vn", 1L, UserRole.EMPLOYEE, true)).getId();
    }

    @Test
    void deactivate_setsInactive_andBumpsTokenVersion() throws Exception {
        long id = newEmployee();

        mvc.perform(asAdmin(patch("/api/accounts/" + id + "/status")).content("{\"active\":false}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.active").value(false));

        User u = users.findById(id).orElseThrow();
        assertThat(u.isActive()).isFalse();
        assertThat(u.getTokenVersion()).isEqualTo(1);
    }

    @Test
    void reactivate_setsActive_andBumpsTokenVersionAgain() throws Exception {
        long id = newEmployee();
        mvc.perform(asAdmin(patch("/api/accounts/" + id + "/status")).content("{\"active\":false}")).andExpect(status().isOk());

        mvc.perform(asAdmin(patch("/api/accounts/" + id + "/status")).content("{\"active\":true}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.active").value(true));

        assertThat(users.findById(id).orElseThrow().getTokenVersion()).isEqualTo(2);
    }

    @Test
    void deactivate_doesNotTouchOtherUsers() throws Exception {
        long id = newEmployee();
        long otherId = users.save(user("emp2", "emp2@hsu.edu.vn", 1L, UserRole.EMPLOYEE, true)).getId();

        mvc.perform(asAdmin(patch("/api/accounts/" + id + "/status")).content("{\"active\":false}")).andExpect(status().isOk());

        User other = users.findById(otherId).orElseThrow();
        assertThat(other.getTokenVersion()).isZero();
        assertThat(other.isActive()).isTrue();
    }

    @Test
    void deactivate_ownAccount_400_andStaysActive() throws Exception {
        mvc.perform(asAdmin(patch("/api/accounts/" + adminId + "/status")).content("{\"active\":false}"))
                .andExpect(status().isBadRequest());

        assertThat(users.findById(adminId).orElseThrow().isActive()).isTrue();
    }

    @Test
    void status_unknownId_404_andMissingBody_400() throws Exception {
        mvc.perform(asAdmin(patch("/api/accounts/99999/status")).content("{\"active\":false}")).andExpect(status().isNotFound());
        mvc.perform(asAdmin(patch("/api/accounts/" + adminId + "/status")).content("{}")).andExpect(status().isBadRequest());
    }

    // ── Đặt lại mật khẩu ───────────────────────────────────────────────────────

    @Test
    void resetPassword_changesPassword_forcesChange_bumpsTokenVersion() throws Exception {
        long id = newEmployee();

        mvc.perform(asAdmin(post("/api/accounts/" + id + "/reset-password")).content("{\"tempPassword\":\"NewTemp@99\"}"))
                .andExpect(status().isNoContent());

        User u = users.findById(id).orElseThrow();
        assertThat(u.getPasswordHash()).isEqualTo("enc(NewTemp@99)");
        assertThat(u.isMustChangePassword()).isTrue();
        assertThat(u.getTokenVersion()).isEqualTo(1);
    }

    @Test
    void resetPassword_weakOrUnknown() throws Exception {
        mvc.perform(asAdmin(post("/api/accounts/" + adminId + "/reset-password")).content("{\"tempPassword\":\"short\"}"))
                .andExpect(status().isBadRequest());
        mvc.perform(asAdmin(post("/api/accounts/99999/reset-password")).content("{\"tempPassword\":\"NewTemp@99\"}"))
                .andExpect(status().isNotFound());
    }

    // ── Tự phục vụ (mọi role) ──────────────────────────────────────────────────

    private MockHttpServletRequestBuilder asEmployee(MockHttpServletRequestBuilder b, long id) {
        return as("EMPLOYEE", id, b);
    }

    @Test
    void me_isAvailableToNonAdmin_andHidesPassword() throws Exception {
        long id = newEmployee();

        mvc.perform(asEmployee(get("/api/accounts/me"), id))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.username").value("emp1"))
                .andExpect(jsonPath("$.passwordHash").doesNotExist());
    }

    @Test
    void me_butOtherAccountEndpointsStayAdminOnly() throws Exception {
        long id = newEmployee();

        mvc.perform(asEmployee(put("/api/accounts/me"), id).content("{}")).andExpect(status().isForbidden());
        mvc.perform(asEmployee(get("/api/accounts/me/"), id)).andExpect(status().isForbidden());
    }

    @Test
    void changePassword_byNonAdmin_updatesHashAndClearsMustChange() throws Exception {
        User u = user("emp1", "emp1@hsu.edu.vn", 1L, UserRole.EMPLOYEE, true);
        u.setMustChangePassword(true);
        long id = users.save(u).getId();

        mvc.perform(asEmployee(post("/api/accounts/me/change-password"), id)
                        .content("{\"currentPassword\":\"Old@12345\",\"newPassword\":\"Brand@New1\"}"))
                .andExpect(status().isNoContent());

        User saved = users.findById(id).orElseThrow();
        assertThat(saved.getPasswordHash()).isEqualTo("enc(Brand@New1)");
        assertThat(saved.isMustChangePassword()).isFalse();
    }

    @Test
    void changePassword_wrongCurrent_400_weakNew_400() throws Exception {
        long id = newEmployee();

        mvc.perform(asEmployee(post("/api/accounts/me/change-password"), id)
                        .content("{\"currentPassword\":\"wrong\",\"newPassword\":\"Brand@New1\"}"))
                .andExpect(status().isBadRequest());
        mvc.perform(asEmployee(post("/api/accounts/me/change-password"), id)
                        .content("{\"currentPassword\":\"Old@12345\",\"newPassword\":\"short\"}"))
                .andExpect(status().isBadRequest());
        assertThat(users.findById(id).orElseThrow().getPasswordHash()).isEqualTo("enc(Old@12345)");
    }

    // ── auth-service không trả lời ─────────────────────────────────────────────

    private void authServiceDown() {
        var down = new org.springframework.web.server.ResponseStatusException(
                org.springframework.http.HttpStatus.SERVICE_UNAVAILABLE, "Auth service unavailable");
        when(passwords.encode(anyString())).thenThrow(down);
        when(passwords.matches(anyString(), anyString())).thenThrow(down);
    }

    @Test
    void create_authServiceDown_503_andNoAccountCreated() throws Exception {
        authServiceDown();

        mvc.perform(asAdmin(post("/api/accounts")).content(createBody("alice", "alice@hsu.edu.vn")))
                .andExpect(status().isServiceUnavailable());

        assertThat(users.existsByUsername("alice")).isFalse();
    }

    @Test
    void resetAndChangePassword_authServiceDown_503_andHashUnchanged() throws Exception {
        long id = newEmployee();
        authServiceDown();

        mvc.perform(asAdmin(post("/api/accounts/" + id + "/reset-password")).content("{\"tempPassword\":\"NewTemp@99\"}"))
                .andExpect(status().isServiceUnavailable());
        mvc.perform(asEmployee(post("/api/accounts/me/change-password"), id)
                        .content("{\"currentPassword\":\"Old@12345\",\"newPassword\":\"Brand@New1\"}"))
                .andExpect(status().isServiceUnavailable());

        User u = users.findById(id).orElseThrow();
        assertThat(u.getPasswordHash()).isEqualTo("enc(Old@12345)");
        assertThat(u.getTokenVersion()).isZero();
    }
}
