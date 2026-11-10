package com.example.accountservice.repository;

import com.example.accountservice.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long>, JpaSpecificationExecutor<User> {
    Optional<User> findByUsername(String username);

    boolean existsByUsername(String username);

    boolean existsByEmailIgnoreCase(String email);

    boolean existsByEmailIgnoreCaseAndIdNot(String email, long id);

    // Hai lệnh dưới là UPDATE nguyên tử, chỉ đụng đúng các cột này: ghi xong sau khi đã gọi auth-service băm mật khẩu mà không
    // giữ kết nối DB lúc chờ mạng, và không ghi đè thay đổi đồng thời ở cột khác (vd is_active). Trả về số dòng đã cập nhật (0 = không có id).

    /** `bump` = số lần tăng token_version (1 = hủy mọi phiên đang đăng nhập). */
    @Modifying
    @Transactional
    @Query("update User u set u.passwordHash = :hash, u.mustChangePassword = :mustChange, u.tokenVersion = u.tokenVersion + :bump where u.id = :id")
    int updatePassword(@Param("id") long id, @Param("hash") String hash, @Param("mustChange") boolean mustChange, @Param("bump") int bump);

    @Modifying
    @Transactional
    @Query("update User u set u.tokenVersion = u.tokenVersion + 1 where u.id = :id")
    int bumpTokenVersion(@Param("id") long id);
}
