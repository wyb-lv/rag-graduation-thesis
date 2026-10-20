package com.example.fileservice.services;

import lombok.extern.slf4j.Slf4j;
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.rendering.ImageType;
import org.apache.pdfbox.rendering.PDFRenderer;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.poi.xwpf.extractor.XWPFWordExtractor;
import org.apache.poi.xwpf.usermodel.XWPFDocument;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Set;

/**
 * Trích xuất văn bản từ file đã upload.
 * PDF: trang có lớp text thì lấy trực tiếp, trang scan (ít text) thì render ảnh rồi OCR bằng Tesseract.
 * DOCX: đọc text bằng Apache POI. Ảnh: OCR trực tiếp.
 */
@Slf4j
@Service
public class TextExtractionService {

    private static final Set<String> IMAGE_EXTENSIONS = Set.of("png", "jpg", "jpeg", "tif", "tiff", "bmp");

    private final String tessdataPath;
    private final String language;
    private final int dpi;
    private final int minTextChars;

    public TextExtractionService(@Value("${ocr.tessdata-path}") String tessdataPath,
                                 @Value("${ocr.language}") String language,
                                 @Value("${ocr.dpi}") int dpi,
                                 @Value("${ocr.min-text-chars}") int minTextChars) {
        this.tessdataPath = Paths.get(tessdataPath).toAbsolutePath().normalize().toString();
        this.language = language;
        this.dpi = dpi;
        this.minTextChars = minTextChars;

        for (String lang : language.split("\\+")) {
            if (!Files.exists(Paths.get(this.tessdataPath, lang + ".traineddata"))) {
                log.warn("Không tìm thấy {}.traineddata trong {}, OCR sẽ lỗi cho tới khi bổ sung", lang, this.tessdataPath);
            }
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
            text = newTesseract().doOCR(file.toFile());
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
                    BufferedImage image = renderer.renderImageWithDPI(page, dpi, ImageType.GRAY);
                    pageText = tesseract.doOCR(image);
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
             XWPFDocument docx = new XWPFDocument(in);
             XWPFWordExtractor extractor = new XWPFWordExtractor(docx)) {
            return extractor.getText();
        }
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
