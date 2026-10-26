package com.example.fileservice.services;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * OCR bằng ONNX Runtime: PaddleOCR PP-OCRv5 (phát hiện vùng chữ, không phụ thuộc ngôn ngữ) + VietOCR vgg_transformer
 * (đọc từng dòng, có đủ dấu tiếng Việt; các model recognition của PaddleOCR không có đủ ký tự tiếng Việt).
 * Mỗi vùng chữ trả về một Box, đưa tiếp cho {@link TableLayout} để dựng bảng.
 * <p>
 * Thư mục model cần: det.onnx, vgg_encoder.onnx (+ .data), vgg_decoder.onnx, vocab.json.
 */
@Slf4j
final class OnnxOcr implements AutoCloseable {

    private static final int DET_MAX_SIDE = 1600;
    private static final float DET_THRESHOLD = 0.3f;
    private static final float BOX_THRESHOLD = 0.6f;
    private static final double UNCLIP_RATIO = 1.5;
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private static final int REC_HEIGHT = 32;
    // VietOCR chỉ đọc đúng tới ~55 ký tự/lần; dòng rộng hơn (tính sau khi thu về cao 32px) phải cắt ở khoảng trống giữa các từ
    private static final int REC_CHUNK_WIDTH = 480;
    private static final int MAX_TOKENS = 128;
    private static final int TOKEN_SOS = 1;
    private static final int TOKEN_EOS = 2;
    private static final int TOKEN_OFFSET = 4; // 0 pad, 1 sos, 2 eos, 3 mask

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession det;
    private final OrtSession encoder;
    private final OrtSession decoder;
    private final String chars;

    OnnxOcr(Path modelsDir) throws OrtException, IOException {
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            det = env.createSession(modelsDir.resolve("det.onnx").toString(), options);
            encoder = env.createSession(modelsDir.resolve("vgg_encoder.onnx").toString(), options);
            decoder = env.createSession(modelsDir.resolve("vgg_decoder.onnx").toString(), options);
        }
        chars = unquote(Files.readString(modelsDir.resolve("vocab.json"), StandardCharsets.UTF_8).strip());
    }

    List<TableLayout.Box> read(BufferedImage image) throws OrtException {
        List<TableLayout.Box> result = new ArrayList<>();
        for (float[] r : detect(image)) {
            int x0 = Math.max(0, (int) r[0]), y0 = Math.max(0, (int) r[1]);
            int x1 = Math.min(image.getWidth(), (int) r[2]), y1 = Math.min(image.getHeight(), (int) r[3]);
            if (x1 - x0 < 4 || y1 - y0 < 4) continue;
            String text = readLine(image.getSubimage(x0, y0, x1 - x0, y1 - y0));
            if (!text.isBlank()) result.add(new TableLayout.Box(text, x0, y0, x1 - x0, y1 - y0));
        }
        return result;
    }

    // ── Phát hiện vùng chữ (DB) ──────────────────────────────────────────────────

    private List<float[]> detect(BufferedImage image) throws OrtException {
        int w = image.getWidth(), h = image.getHeight();
        double ratio = Math.min(1.0, (double) DET_MAX_SIDE / Math.max(w, h));
        int nw = Math.max(32, (int) Math.round(w * ratio / 32) * 32);
        int nh = Math.max(32, (int) Math.round(h * ratio / 32) * 32);

        BufferedImage scaled = new BufferedImage(nw, nh, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = scaled.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(image, 0, 0, nw, nh, null);
        g.dispose();

        int[] rgb = scaled.getRGB(0, 0, nw, nh, null, 0, nw);
        float[] input = new float[3 * nh * nw];
        for (int i = 0; i < rgb.length; i++) {
            input[i] = (((rgb[i] >> 16) & 0xFF) / 255f - MEAN[0]) / STD[0];
            input[nh * nw + i] = (((rgb[i] >> 8) & 0xFF) / 255f - MEAN[1]) / STD[1];
            input[2 * nh * nw + i] = ((rgb[i] & 0xFF) / 255f - MEAN[2]) / STD[2];
        }

        float[] prob;
        try (OnnxTensor in = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[]{1, 3, nh, nw});
             OrtSession.Result out = det.run(Map.of("x", in))) {
            FloatBuffer buf = ((OnnxTensor) out.get(0)).getFloatBuffer();
            prob = new float[buf.remaining()];
            buf.get(prob);
        }

        List<float[]> boxes = textBoxes(prob, nh, nw);
        for (float[] b : boxes) {
            b[0] = (float) (b[0] * w / nw);
            b[2] = (float) (b[2] * w / nw);
            b[1] = (float) (b[1] * h / nh);
            b[3] = (float) (b[3] * h / nh);
        }
        return boxes;
    }

    /**
     * Hậu xử lý bản đồ xác suất của DB: lấy các vùng liên thông trên ngưỡng, bỏ vùng điểm thấp,
     * nới rộng mỗi vùng (unclip) vì model dự đoán vùng chữ hơi co lại. Trả về {x0, y0, x1, y1}.
     */
    // shortcut: hộp thẳng (không xoay), đủ cho tài liệu scan/render thẳng; trang chụp nghiêng nhiều thì cần minAreaRect
    static List<float[]> textBoxes(float[] prob, int h, int w) {
        List<float[]> boxes = new ArrayList<>();
        boolean[] seen = new boolean[h * w];
        int[] stack = new int[h * w];
        for (int start = 0; start < prob.length; start++) {
            if (seen[start] || prob[start] <= DET_THRESHOLD) continue;
            int sp = 0, count = 0;
            int minX = w, minY = h, maxX = 0, maxY = 0;
            double sum = 0;
            stack[sp++] = start;
            seen[start] = true;
            while (sp > 0) {
                int p = stack[--sp];
                int x = p % w, y = p / w;
                count++;
                sum += prob[p];
                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
                        int q = ny * w + nx;
                        if (!seen[q] && prob[q] > DET_THRESHOLD) {
                            seen[q] = true;
                            stack[sp++] = q;
                        }
                    }
                }
            }
            int bw = maxX - minX + 1, bh = maxY - minY + 1;
            if (Math.min(bw, bh) < 3 || sum / count < BOX_THRESHOLD) continue;
            double d = (double) bw * bh * UNCLIP_RATIO / (2.0 * (bw + bh));
            boxes.add(new float[]{
                    (float) Math.max(0, minX - d), (float) Math.max(0, minY - d),
                    (float) Math.min(w, maxX + 1 + d), (float) Math.min(h, maxY + 1 + d)});
        }
        return boxes;
    }

    // ── Nhận dạng dòng chữ (VietOCR) ─────────────────────────────────────────────

    private String readLine(BufferedImage crop) throws OrtException {
        int w = crop.getWidth(), h = crop.getHeight();
        int[] rgb = crop.getRGB(0, 0, w, h, null, 0, w);
        List<String> parts = new ArrayList<>();
        for (int[] cut : chunks(rgb, w, h)) {
            if (cut[1] - cut[0] > 3) parts.add(recognize(rgb, w, h, cut[0], cut[1]));
        }
        return String.join(" ", parts).strip();
    }

    /** Cắt dòng rộng thành các đoạn [từ, đến) tại khoảng trống rộng nhất gần cuối cửa sổ, tránh cắt giữa chữ. */
    static List<int[]> chunks(int[] rgb, int w, int h) {
        int limit = (int) ((long) REC_CHUNK_WIDTH * h / REC_HEIGHT);
        boolean[] blank = new boolean[w];
        for (int x = 0; x < w; x++) {
            int ink = 0;
            for (int y = 0; y < h; y++) {
                int p = rgb[y * w + x];
                double gray = 0.299 * ((p >> 16) & 0xFF) + 0.587 * ((p >> 8) & 0xFF) + 0.114 * (p & 0xFF);
                if (gray < 140) ink++;
            }
            blank[x] = ink <= Math.max(1, 0.03 * h);
        }
        List<int[]> out = new ArrayList<>();
        int start = 0;
        while (w - start > limit) {
            int lo = start + (int) (limit * 0.55), hi = start + limit;
            int bestFrom = -1, bestTo = -1;
            for (int i = hi - 1; i >= lo; ) {
                if (!blank[i]) {
                    i--;
                    continue;
                }
                int j = i;
                while (j > lo && blank[j - 1]) j--;
                if (bestFrom < 0 || i - j > bestTo - bestFrom) {
                    bestFrom = j;
                    bestTo = i;
                }
                if (i - j >= 0.012 * limit) break; // đủ rộng cho một dấu cách
                i = j - 1;
            }
            int cut = bestFrom >= 0 ? (bestFrom + bestTo) / 2 : hi;
            out.add(new int[]{start, cut});
            start = cut;
        }
        out.add(new int[]{start, w});
        return out;
    }

    private String recognize(int[] rgb, int w, int h, int from, int to) throws OrtException {
        int pw = to - from;
        int nw = Math.max(32, (int) Math.ceil((int) (REC_HEIGHT * (double) pw / h) / 10.0) * 10);
        float[] input = areaResize(rgb, w, h, from, to, nw);

        try (OnnxTensor img = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), new long[]{1, 3, REC_HEIGHT, nw});
             OrtSession.Result enc = encoder.run(Map.of("img", img))) {
            OnnxTensor memory = (OnnxTensor) enc.get(0);
            long[] tokens = new long[MAX_TOKENS + 1];
            tokens[0] = TOKEN_SOS;
            int n = 1;
            StringBuilder text = new StringBuilder();
            while (n <= MAX_TOKENS) {
                int next;
                try (OnnxTensor tgt = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens, 0, n).slice(), new long[]{n, 1});
                     OrtSession.Result out = decoder.run(Map.of("tgt", tgt, "memory", memory))) {
                    FloatBuffer logits = ((OnnxTensor) out.get(0)).getFloatBuffer();
                    int vocab = logits.remaining() / n;
                    int base = (n - 1) * vocab;
                    next = 0;
                    for (int k = 1; k < vocab; k++) {
                        if (logits.get(base + k) > logits.get(base + next)) next = k;
                    }
                }
                if (next == TOKEN_EOS) break;
                tokens[n++] = next;
                if (next >= TOKEN_OFFSET && next - TOKEN_OFFSET < chars.length()) text.append(chars.charAt(next - TOKEN_OFFSET));
            }
            return text.toString();
        }
    }

    /** Thu cột [from, to) của ảnh về cao 32px, rộng nw, lấy trung bình vùng; trả về CHW chuẩn hóa về [0, 1]. */
    private static float[] areaResize(int[] rgb, int w, int h, int from, int to, int nw) {
        float[] out = new float[3 * REC_HEIGHT * nw];
        double sx = (double) (to - from) / nw, sy = (double) h / REC_HEIGHT;
        for (int y = 0; y < REC_HEIGHT; y++) {
            int y0 = (int) Math.floor(y * sy), y1 = Math.min(h, Math.max(y0 + 1, (int) Math.ceil((y + 1) * sy)));
            for (int x = 0; x < nw; x++) {
                int x0 = from + (int) Math.floor(x * sx), x1 = Math.min(to, Math.max(x0 + 1, from + (int) Math.ceil((x + 1) * sx)));
                double r = 0, g = 0, b = 0;
                int cnt = 0;
                for (int yy = y0; yy < y1; yy++) {
                    for (int xx = x0; xx < x1; xx++) {
                        int p = rgb[yy * w + xx];
                        r += (p >> 16) & 0xFF;
                        g += (p >> 8) & 0xFF;
                        b += p & 0xFF;
                        cnt++;
                    }
                }
                int i = y * nw + x;
                out[i] = (float) (r / cnt / 255);
                out[REC_HEIGHT * nw + i] = (float) (g / cnt / 255);
                out[2 * REC_HEIGHT * nw + i] = (float) (b / cnt / 255);
            }
        }
        return out;
    }

    // vocab.json là một chuỗi JSON duy nhất (có thể chứa \\ và \")
    static String unquote(String json) {
        String body = json.substring(1, json.length() - 1);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < body.length(); i++) {
            char c = body.charAt(i);
            if (c != '\\') {
                sb.append(c);
                continue;
            }
            char e = body.charAt(++i);
            if (e == 'u') {
                sb.append((char) Integer.parseInt(body.substring(i + 1, i + 5), 16));
                i += 4;
            } else {
                sb.append(switch (e) { case 'n' -> '\n'; case 't' -> '\t'; default -> e; });
            }
        }
        return sb.toString();
    }

    @Override
    public void close() throws OrtException {
        det.close();
        encoder.close();
        decoder.close();
    }
}
