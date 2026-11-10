package com.example.accountservice.dto;

import org.springframework.data.domain.Page;

import java.util.List;

/** JSON phân trang ổn định (không serialize PageImpl trực tiếp). */
public record PageResponse<T>(List<T> content, int page, int size, long totalElements, int totalPages) {
    public static <T> PageResponse<T> of(Page<T> p) {
        return new PageResponse<>(p.getContent(), p.getNumber(), p.getSize(), p.getTotalElements(), p.getTotalPages());
    }
}
