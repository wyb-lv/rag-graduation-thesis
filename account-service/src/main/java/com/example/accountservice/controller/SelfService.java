package com.example.accountservice.controller;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/** API cho mọi người dùng đã đăng nhập (thao tác trên chính tài khoản của người gọi); các API còn lại của /api/accounts chỉ cho ADMIN. */
@Documented
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface SelfService {
}
