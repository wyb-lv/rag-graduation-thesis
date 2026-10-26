package com.example.fileservice.services;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TableLayoutTest {

    private static TableLayout.Box w(String t, int x, int y) {
        return new TableLayout.Box(t, x, y, t.length() * 10, 20);
    }

    @Test
    void rowsWithWideGapsBecomeMarkdownTable() {
        String text = TableLayout.layout(List.of(
                w("Chứng", 0, 0), w("chỉ", 60, 0), w("Điểm", 400, 0),
                w("IELTS", 0, 40), w("5.0", 400, 40),
                w("TOEIC", 0, 80), w("600", 400, 80)));
        assertEquals("| Chứng chỉ | Điểm |\n| --- | --- |\n| IELTS | 5.0 |\n| TOEIC | 600 |", text);
    }

    @Test
    void plainParagraphStaysText() {
        String text = TableLayout.layout(List.of(
                w("Sinh", 0, 0), w("viên", 50, 0), w("phải", 100, 0),
                w("đăng", 0, 30), w("ký", 50, 30)));
        assertEquals("Sinh viên phải\nđăng ký", text);
    }
}
