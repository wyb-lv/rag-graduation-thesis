package com.example.fileservice.services;

import lombok.extern.slf4j.Slf4j;
import net.sourceforge.tess4j.ITessAPI;
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.poi.xwpf.usermodel.IBodyElement;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.apache.poi.xwpf.usermodel.XWPFParagraph;
import org.apache.poi.xwpf.usermodel.XWPFTable;
import org.apache.poi.xwpf.usermodel.XWPFTableRow;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Trích xuất văn bản từ file đã upload.
 * PDF: trang có lớp text thì lấy trực tiếp, trang scan (ít text) thì render ảnh rồi OCR bằng Tesseract.
 * DOCX: đọc text bằng Apache POI. Ảnh: OCR trực tiếp.
 * Bảng (DOCX, ảnh, trang PDF scan) được xuất thành bảng Markdown.
 */
@Slf4j
@Service
public class TextExtractionService {

    private static final Set<String> IMAGE_EXTENSIONS = Set.of("png", "jpg", "jpeg", "tif", "tiff", "bmp");

    private final String tessdataPath;
    private final String language;
    private final int dpi;
    private final int minTextChars;
    // null = dùng Tesseract
    private final OnnxOcr onnxOcr;

    public TextExtractionService(@Value("${ocr.tessdata-path}") String tessdataPath,
                                 @Value("${ocr.language}") String language,
                                 @Value("${ocr.dpi}") int dpi,
                                 @Value("${ocr.min-text-chars}") int minTextChars,
                                 @Value("${ocr.engine:tesseract}") String engine,
                                 @Value("${ocr.models-path:./models}") String modelsPath) {
        this.tessdataPath = Paths.get(tessdataPath).toAbsolutePath().normalize().toString();
        this.language = language;
        this.dpi = dpi;
        this.minTextChars = minTextChars;
        this.onnxOcr = "onnx".equalsIgnoreCase(engine) ? loadOnnx(Paths.get(modelsPath)) : null;

        for (String lang : language.split("\\+")) {
            if (!Files.exists(Paths.get(this.tessdataPath, lang + ".traineddata"))) {
                log.warn("Không tìm thấy {}.traineddata trong {}, OCR sẽ lỗi cho tới khi bổ sung", lang, this.tessdataPath);
            }
        }
    }

    private static OnnxOcr loadOnnx(Path modelsDir) {
        try {
            OnnxOcr engine = new OnnxOcr(modelsDir.toAbsolutePath().normalize());
            log.info("OCR dùng ONNX (PaddleOCR detection + VietOCR) từ {}", modelsDir);
            return engine;
        } catch (Throwable e) {
            log.warn("Không nạp được model ONNX ở {}, dùng Tesseract: {}", modelsDir, e.toString());
            return null;
        }
    }

    public static boolean isSupported(String fileName) {
        String ext = extension(fileName);
        return ext.equals("pdf") || ext.equals("docx") || IMAGE_EXTENSIONS.contains(ext);
    }

    public String extract(Path file) throws Exception {
        String ext = extension(file.getFileName().toString());
        String text;
        if (ext.equals("pdf")) {
            text = extractPdf(file);
        } else if (ext.equals("docx")) {
            text = extractDocx(file);
        } else if (IMAGE_EXTENSIONS.contains(ext)) {
            BufferedImage image = ImageIO.read(file.toFile());
            ITesseract tesseract = newTesseract();
            text = image != null ? ocr(tesseract, image) : tesseract.doOCR(file.toFile());
        } else {
            throw new Exception("Không hỗ trợ trích xuất văn bản cho định dạng: " + ext);
        }
        // Postgres TEXT không nhận ký tự NUL (PDFBox đôi khi sinh ra)
        return text.replace("\u0000", "").strip();
    }

    private String extractPdf(Path file) throws Exception {
        try (PDDocument pdf = Loader.loadPDF(file.toFile())) {
            PDFTextStripper stripper = new PDFTextStripper();
            PDFRenderer renderer = new PDFRenderer(pdf);
            ITesseract tesseract = null;
            StringBuilder result = new StringBuilder();
            int ocrPages = 0;

            for (int page = 0; page < pdf.getNumberOfPages(); page++) {
                stripper.setStartPage(page + 1);
                stripper.setEndPage(page + 1);
                String pageText = stripper.getText(pdf);

                if (pageText.replaceAll("\\s", "").length() < minTextChars) {
                    if (tesseract == null) {
                        tesseract = newTesseract();
                        tesseract.setVariable("user_defined_dpi", String.valueOf(dpi));
                    }
                    BufferedImage image = renderer.renderImageWithDPI(page, dpi, onnxOcr != null ? ImageType.RGB : ImageType.GRAY);
                    pageText = ocr(tesseract, image);
                    ocrPages++;
                }
                result.append(pageText.strip()).append("\n\n");
            }

            log.info("{}: {} trang, OCR {} trang scan", file.getFileName(), pdf.getNumberOfPages(), ocrPages);
            return result.toString();
        }
    }

    private String extractDocx(Path file) throws Exception {
        try (InputStream in = Files.newInputStream(file);
             XWPFDocument docx = new XWPFDocument(in)) {
            StringBuilder sb = new StringBuilder();
            for (IBodyElement element : docx.getBodyElements()) {
                if (element instanceof XWPFParagraph p) {
                    sb.append(p.getText()).append('\n');
                } else if (element instanceof XWPFTable table) {
                    appendTable(sb, table);
                }
            }
            return sb.toString();
        }
    }

    private static void appendTable(StringBuilder sb, XWPFTable table) {
        boolean first = true;
        for (XWPFTableRow row : table.getRows()) {
            sb.append('|');
            for (var cell : row.getTableCells()) {
                sb.append(' ').append(cell.getText().replaceAll("\\s+", " ").replace('|', '/').strip()).append(" |");
            }
            sb.append('\n');
            if (first) {
                sb.append("|").append(" --- |".repeat(Math.max(1, row.getTableCells().size()))).append('\n');
                first = false;
            }
        }
    }

    // OCR theo tọa độ từng từ để dựng lại bảng; doOCR thuần sẽ đọc bảng theo cột và làm lẫn các ô
    private String ocr(ITesseract tesseract, BufferedImage image) throws Exception {
        if (onnxOcr != null) return TableLayout.layout(onnxOcr.read(image));
        List<TableLayout.Box> boxes = tesseract.getWords(image, ITessAPI.TessPageIteratorLevel.RIL_WORD).stream()
                .map(w -> new TableLayout.Box(w.getText(), w.getBoundingBox().x, w.getBoundingBox().y,
                        w.getBoundingBox().width, w.getBoundingBox().height))
                .toList();
        return TableLayout.layout(boxes);
    }

    // Tesseract không thread-safe nên mỗi lần xử lý tạo instance riêng
    private ITesseract newTesseract() {
        Tesseract tesseract = new Tesseract();
        tesseract.setDatapath(tessdataPath);
        tesseract.setLanguage(language);
        return tesseract;
    }

    private static String extension(String fileName) {
        int dot = fileName.lastIndexOf('.');
        return dot < 0 ? "" : fileName.substring(dot + 1).toLowerCase(Locale.ROOT);
    }
}
