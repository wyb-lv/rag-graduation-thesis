"""OCR (app/ocr): bang Markdown, client PaddleOCR-VL, dinh tuyen trang PDF (chu hien thi -> doc thang, scan -> PaddleOCR-VL), DOCX.
Khong goi PaddleOCR-VL that: paddle_vl.parse / httpx.post duoc thay bang ban gia."""
import base64
import io

import httpx
import pymupdf
import pytest
from docx import Document

from app import config as cfg
from app.ocr import extract, markdown, paddle_vl


# ── bang Markdown ────────────────────────────────────────────────────────────────

def test_md_table_header_separator_ragged_rows_and_pipes():
    assert markdown.md_table([["A", "B"], ["x | y"], [], ["", ""]]) == "| A | B |\n| --- | --- |\n| x / y |  |"
    assert markdown.md_table([]) == ""


def test_html_table_with_colspan_and_rowspan_keeps_text_once():
    html = ("<table><tr><td colspan=\"2\">Tiêu đề</td><td>Điểm</td></tr>"
            "<tr><td rowspan=\"2\">IELTS</td><td>chứng chỉ<br>quốc tế</td><td>5.0</td></tr>"
            "<tr><td>TOEIC</td><td>600</td></tr></table>")
    assert markdown.html_table_rows(html) == [["Tiêu đề", "", "Điểm"], ["IELTS", "chứng chỉ quốc tế", "5.0"], ["", "TOEIC", "600"]]
    assert markdown.html_to_markdown("Trước\n" + html + "\nSau").count("| IELTS | chứng chỉ quốc tế | 5.0 |") == 1


def test_html_to_markdown_strips_wrappers_and_images_but_keeps_text():
    text = '<div style="text-align: center;">QUY ĐỊNH</div>\n<img src="imgs/a.jpg" alt="x" />\nĐiều 1.'
    assert markdown.html_to_markdown(text).split() == ["QUY", "ĐỊNH", "Điều", "1."]


# ── client PaddleOCR-VL ──────────────────────────────────────────────────────────

def reply(status, payload):
    return httpx.Response(status, json=payload, request=httpx.Request("POST", "http://x"))


def ok(*texts):
    return reply(200, {"errorCode": 0, "result": {"layoutParsingResults": [{"markdown": {"text": t}} for t in texts]}})


def test_parse_sends_base64_file_type_and_ignore_labels_and_converts_tables(monkeypatch):
    sent = {}

    def post(url, json, timeout):
        sent.update(url=url, body=json, timeout=timeout)
        return ok("# Tiêu đề\n<table><tr><td>a</td><td>b</td></tr></table>", "trang hai")

    monkeypatch.setattr(paddle_vl.httpx, "post", post)
    monkeypatch.setattr(cfg, "OCR_VL_URL", "http://paddle:8080")
    assert paddle_vl.parse(b"%PDF-data", pdf=True) == ["# Tiêu đề\n\n| a | b |\n| --- | --- |", "trang hai"]
    assert sent["url"] == "http://paddle:8080/layout-parsing"
    assert base64.b64decode(sent["body"]["file"]) == b"%PDF-data" and sent["body"]["fileType"] == 0
    assert sent["body"]["markdownIgnoreLabels"] == cfg.OCR_IGNORE_LABELS and sent["body"]["visualize"] is False
    monkeypatch.setattr(paddle_vl.httpx, "post", lambda *a, **k: ok("x"))
    assert paddle_vl.parse(b"\x89PNG", pdf=False) == ["x"]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_parse_server_errors_are_retryable(monkeypatch, status):
    monkeypatch.setattr(paddle_vl.httpx, "post", lambda *a, **k: reply(status, {"errorMsg": "busy"}))
    with pytest.raises(paddle_vl.OcrUnavailable):
        paddle_vl.parse(b"x", pdf=True)


def test_parse_connection_failure_does_not_leak_the_error_text(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("http://internal-host:8080 refused")

    monkeypatch.setattr(paddle_vl.httpx, "post", boom)
    with pytest.raises(paddle_vl.OcrUnavailable) as e:
        paddle_vl.parse(b"x", pdf=True)
    assert "internal-host" not in str(e.value)


def test_parse_bad_input_and_garbled_reply(monkeypatch):
    monkeypatch.setattr(paddle_vl.httpx, "post", lambda *a, **k: reply(422, {"errorMsg": "anh hong"}))
    with pytest.raises(paddle_vl.BadDocument, match="anh hong"):
        paddle_vl.parse(b"x", pdf=False)
    monkeypatch.setattr(paddle_vl.httpx, "post", lambda *a, **k: reply(200, {"result": {}}))
    with pytest.raises(paddle_vl.OcrUnavailable):
        paddle_vl.parse(b"x", pdf=True)


# ── PDF: trang nao doc thang, trang nao OCR ──────────────────────────────────────

def png():
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 140), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    return pix.tobytes("png")


def digital(page, text="Sinh vien phai dang ky hoc phan truoc khi bat dau hoc ky"):
    page.insert_text((50, 60), text, fontsize=10)


def scan_with_ghost_layer(page):  # anh nguyen trang + lop chu an (che do ve 3), nhu PDF scan da qua OCR cu
    page.insert_image(page.rect, stream=png())
    page.insert_text((50, 100), "Sinh vien phai dang ky hoc phan khong dau", fontsize=6, render_mode=3)


def scan_with_visible_patch(page):  # anh nguyen trang + mot dong chu hien thi chen them: than van ban van nam trong anh
    page.insert_image(page.rect, stream=png())
    page.insert_text((50, 100), "(Ap dung doi voi sinh vien khoa 2025)", fontsize=8)


def make_pdf(*drawers):
    doc = pymupdf.open()
    for draw in drawers:
        draw(doc.new_page())
    return doc.tobytes()


class FakeOcr:
    """Thay paddle_vl.parse: moi trang gui di thanh 'OCR-<so thu tu toan cuc>'; ghi lai so trang moi lo."""

    def __init__(self, monkeypatch):
        self.batches, self.n = [], 0
        monkeypatch.setattr(extract.paddle_vl, "parse", self)

    def __call__(self, data, pdf):
        count = pymupdf.open(stream=data, filetype="pdf").page_count if pdf else 1
        self.batches.append((count, pdf))
        out = [f"OCR-{self.n + i}" for i in range(count)]
        self.n += count
        return out


def test_born_digital_pdf_never_calls_ocr_and_keeps_ruled_table_as_markdown(monkeypatch):
    monkeypatch.setattr(extract.paddle_vl, "parse", lambda *a, **k: pytest.fail("trang co chu hien thi khong duoc OCR"))

    def draw(page):
        page.insert_text((50, 60), "THONG BAO ve viec nop chung chi ngoai ngu quoc te de xet tot nghiep", fontsize=10)
        for r, row in enumerate([["Chung chi", "Diem"], ["IELTS", "5.0"], ["TOEIC", "600"]]):
            for c, text in enumerate(row):
                rect = pymupdf.Rect(50 + c * 120, 150 + r * 24, 170 + c * 120, 174 + r * 24)
                page.draw_rect(rect, width=0.8)
                page.insert_text((rect.x0 + 4, rect.y1 - 7), text, fontsize=10)
        page.insert_text((50, 280), "Sinh vien phai nop truoc ngay mot thang ba theo quy dinh cua Truong", fontsize=10)

    result = extract.extract(make_pdf(draw), "a.pdf")
    assert (result.pages, result.ocr_pages) == (1, 0)
    assert "| Chung chi | Diem |\n| --- | --- |\n| IELTS | 5.0 |\n| TOEIC | 600 |" in result.text
    assert result.text.index("THONG BAO") < result.text.index("| Chung chi") < result.text.index("Sinh vien phai nop")
    assert all(w in result.text for w in ("THONG", "IELTS", "600", "Truong"))  # khong mat chu


def test_scan_with_ghost_layer_and_scan_with_visible_patch_both_go_to_ocr_and_ghost_text_is_dropped(monkeypatch):
    ocr = FakeOcr(monkeypatch)
    result = extract.extract(make_pdf(scan_with_ghost_layer, scan_with_visible_patch), "scan.pdf")
    assert result.text == "OCR-0\n\nOCR-1" and (result.pages, result.ocr_pages) == (2, 2)
    assert ocr.batches == [(2, True)]  # hai trang scan trong mot lo


def test_mixed_pdf_keeps_page_order_and_batches_scan_pages(monkeypatch):
    monkeypatch.setattr(cfg, "OCR_PAGES_PER_REQUEST", 2)
    ocr = FakeOcr(monkeypatch)
    result = extract.extract(make_pdf(digital, scan_with_ghost_layer, lambda p: digital(p, "Dieu 2 quy dinh ve hoc phi"),
                                      scan_with_ghost_layer, scan_with_ghost_layer), "mix.pdf")
    pages = result.text.split("\n\n")
    assert pages[0].startswith("Sinh vien") and pages[2].startswith("Dieu 2")
    assert [pages[1], pages[3], pages[4]] == ["OCR-0", "OCR-1", "OCR-2"]
    assert ocr.batches == [(2, True), (1, True)] and (result.pages, result.ocr_pages) == (5, 3)


def test_ocr_returning_the_wrong_number_of_pages_fails_instead_of_misaligning(monkeypatch):
    monkeypatch.setattr(extract.paddle_vl, "parse", lambda data, pdf: ["chi mot trang"])
    with pytest.raises(paddle_vl.OcrUnavailable):
        extract.extract(make_pdf(scan_with_ghost_layer, scan_with_ghost_layer), "scan.pdf")


def test_pdf_with_password_or_garbage_is_unreadable():
    with pytest.raises(extract.UnreadableFile):
        extract.extract(b"not a pdf", "x.pdf")
    doc = pymupdf.open()
    doc.new_page()
    locked = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="x", owner_pw="y")
    with pytest.raises(extract.UnreadableFile):
        extract.extract(locked, "locked.pdf")


def test_image_goes_to_ocr_as_an_image_and_nul_is_stripped(monkeypatch):
    seen = {}
    monkeypatch.setattr(extract.paddle_vl, "parse", lambda data, pdf: seen.update(pdf=pdf) or ["chữ\x00 trong ảnh", "trang 2"])
    result = extract.extract(png(), "a.TIF")
    assert seen == {"pdf": False} and result.text == "chữ trong ảnh\n\ntrang 2" and result.ocr_pages == 2


# ── DOCX ─────────────────────────────────────────────────────────────────────────

def docx_bytes(build):
    doc = Document()
    build(doc)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_docx_paragraphs_and_tables_keep_order_merged_cells_and_nested_text():
    def build(doc):
        doc.add_paragraph("Trước bảng")
        t = doc.add_table(rows=3, cols=3)
        t.cell(0, 0).merge(t.cell(0, 1)).text = "Tiêu đề gộp"
        t.cell(0, 2).text = "Điểm"
        t.cell(1, 0).text, t.cell(1, 1).text, t.cell(1, 2).text = "IELTS", "chứng chỉ | quốc tế", "5.0"
        t.cell(2, 0).text = "Ghi chú"
        nested = t.cell(2, 1).add_table(rows=1, cols=2)
        nested.cell(0, 0).text, nested.cell(0, 1).text = "lồng", "nhau"
        t.cell(2, 2).text = "ok"
        doc.add_paragraph("Sau bảng")

    result = extract.extract(docx_bytes(build), "a.docx")
    assert result.pages is None and result.ocr_pages == 0
    lines = result.text.split("\n")
    assert lines[0] == "Trước bảng" and lines[-1] == "Sau bảng"
    assert "| Tiêu đề gộp |  | Điểm |" in lines  # o gop ngang: them o rong, khong lap chu
    assert "| IELTS | chứng chỉ / quốc tế | 5.0 |" in lines  # dau | trong o khong pha bang
    assert any(line.startswith("| Ghi chú |") and "lồng nhau" in line and line.endswith("| ok |") for line in lines)  # bang long: giu chu


def test_unsupported_and_corrupt_files():
    with pytest.raises(extract.UnsupportedFormat):
        extract.extract(b"x", "notes.txt")
    with pytest.raises(extract.UnreadableFile):
        extract.extract(b"not a docx", "x.docx")
