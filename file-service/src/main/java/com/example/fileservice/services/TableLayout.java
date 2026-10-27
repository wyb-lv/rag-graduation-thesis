package com.example.fileservice.services;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Dựng lại văn bản từ tọa độ từng từ do OCR trả về: dòng có nhiều cụm cách xa nhau theo chiều ngang
 * (từ 2 dòng liên tiếp trở lên) được xuất thành bảng Markdown, các dòng còn lại giữ nguyên là văn bản thường.
 */
final class TableLayout {

    /** Một từ nhận dạng được cùng khung chứa nó (pixel, gốc ở góc trên trái). */
    record Box(String text, int x, int y, int w, int h) {
        int right() { return x + w; }
        double centerY() { return y + h / 2.0; }
    }

    // Khoảng cách giữa hai từ lớn hơn chừng này lần chiều cao chữ thì coi là sang ô khác (dấu cách thường ~0.3 lần)
    private static final double CELL_GAP = 1.5;
    // Dòng cách dòng trước hơn chừng này lần chiều cao chữ thì ngắt đoạn
    private static final double PARAGRAPH_GAP = 1.6;

    private TableLayout() {}

    // shortcut: ô có chữ xuống nhiều dòng thành nhiều hàng, không gộp ô trộn (merged cell); nâng cấp khi cần bảng phức tạp hơn
    static String layout(List<Box> words) {
        List<Box> boxes = words.stream().filter(b -> !b.text().isBlank()).toList();
        if (boxes.isEmpty()) return "";
        double lineHeight = medianHeight(boxes);

        List<List<List<Box>>> lines = toLines(boxes, lineHeight); // dòng -> ô -> từ
        StringBuilder out = new StringBuilder();
        int i = 0;
        double prevBottom = Double.NaN;
        while (i < lines.size()) {
            int j = i;
            while (j < lines.size() && lines.get(j).size() >= 2) j++;
            if (j - i >= 2) {
                out.append(table(lines.subList(i, j))).append('\n');
                prevBottom = bottom(lines.get(j - 1));
                i = j;
                continue;
            }
            List<List<Box>> line = lines.get(i);
            if (!Double.isNaN(prevBottom) && top(line) - prevBottom > PARAGRAPH_GAP * lineHeight) out.append('\n');
            out.append(join(line.stream().flatMap(List::stream).toList())).append('\n');
            prevBottom = bottom(line);
            i++;
        }
        return out.toString().strip();
    }

    private static List<List<List<Box>>> toLines(List<Box> boxes, double lineHeight) {
        List<Box> sorted = new ArrayList<>(boxes);
        sorted.sort(Comparator.comparingDouble(Box::centerY));

        List<List<Box>> rows = new ArrayList<>();
        List<Box> current = new ArrayList<>();
        double sum = 0;
        for (Box b : sorted) {
            if (!current.isEmpty() && b.centerY() - sum / current.size() > 0.6 * lineHeight) {
                rows.add(current);
                current = new ArrayList<>();
                sum = 0;
            }
            current.add(b);
            sum += b.centerY();
        }
        rows.add(current);

        List<List<List<Box>>> lines = new ArrayList<>();
        for (List<Box> row : rows) {
            row.sort(Comparator.comparingInt(Box::x));
            List<List<Box>> cells = new ArrayList<>();
            List<Box> cell = new ArrayList<>();
            for (Box b : row) {
                if (!cell.isEmpty() && b.x() - cell.get(cell.size() - 1).right() > CELL_GAP * lineHeight) {
                    cells.add(cell);
                    cell = new ArrayList<>();
                }
                cell.add(b);
            }
            cells.add(cell);
            lines.add(cells);
        }
        return lines;
    }

    private static String table(List<List<List<Box>>> rows) {
        int cols = rows.stream().mapToInt(List::size).max().orElse(1);
        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < rows.size(); r++) {
            List<List<Box>> row = rows.get(r);
            sb.append('|');
            for (int c = 0; c < cols; c++) {
                String text = c < row.size() ? join(row.get(c)).replace('|', '/') : "";
                sb.append(' ').append(text).append(" |");
            }
            sb.append('\n');
            if (r == 0) sb.append("|").append(" --- |".repeat(cols)).append('\n');
        }
        return sb.toString().stripTrailing();
    }

    private static String join(List<Box> words) {
        StringBuilder sb = new StringBuilder();
        for (Box b : words) {
            if (sb.length() > 0) sb.append(' ');
            sb.append(b.text().strip());
        }
        return sb.toString();
    }

    private static double top(List<List<Box>> line) {
        return line.stream().flatMap(List::stream).mapToInt(Box::y).min().orElse(0);
    }

    private static double bottom(List<List<Box>> line) {
        return line.stream().flatMap(List::stream).mapToInt(b -> b.y() + b.h()).max().orElse(0);
    }

    private static double medianHeight(List<Box> boxes) {
        int[] h = boxes.stream().mapToInt(Box::h).sorted().toArray();
        return Math.max(1, h[h.length / 2]);
    }
}
