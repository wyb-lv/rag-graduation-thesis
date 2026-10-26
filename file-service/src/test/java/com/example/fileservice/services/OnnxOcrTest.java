package com.example.fileservice.services;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxOcrTest {

    @Test
    void textBoxesFindsEachBlobAndIgnoresWeakOnes() {
        int h = 40, w = 100;
        float[] prob = new float[h * w];
        fill(prob, w, 10, 10, 30, 20, 0.9f);  // vùng chữ
        fill(prob, w, 60, 10, 80, 20, 0.9f);  // vùng thứ hai, cách xa
        fill(prob, w, 10, 30, 30, 36, 0.4f);  // trên ngưỡng 0.3 nhưng điểm trung bình thấp
        List<float[]> boxes = OnnxOcr.textBoxes(prob, h, w);
        assertEquals(2, boxes.size());
        assertTrue(boxes.stream().allMatch(b -> b[0] < 20 || b[0] >= 50));
    }

    @Test
    void shortLineIsOneChunk() {
        int w = 200, h = 32;
        List<int[]> chunks = OnnxOcr.chunks(new int[w * h], w, h);
        assertEquals(1, chunks.size());
        assertEquals(w, chunks.get(0)[1]);
    }

    @Test
    void longLineIsCutAtBlankColumns() {
        int h = 32, w = 1200; // giới hạn 480px ở cao 32 => 1 đoạn tối đa 480px
        int[] rgb = new int[w * h];
        java.util.Arrays.fill(rgb, 0xFFFFFF);
        for (int x = 0; x < w; x++) { // chữ đen khắp nơi, trừ khoảng trống quanh x=420
            if (x < 410 || x > 430) for (int y = 5; y < 25; y++) rgb[y * w + x] = 0;
        }
        List<int[]> chunks = OnnxOcr.chunks(rgb, w, h);
        assertEquals(420, chunks.get(0)[1], 10);
    }

    @Test
    void unquoteHandlesEscapes() {
        assertEquals("a\"b\\cà", OnnxOcr.unquote("\"a\\\"b\\\\c\\u00e0\""));
    }

    private static void fill(float[] prob, int w, int x0, int y0, int x1, int y1, float v) {
        for (int y = y0; y < y1; y++) for (int x = x0; x < x1; x++) prob[y * w + x] = v;
    }
}
