"""Trich van ban + bang (Markdown) tu file upload. Chu co san thi doc thang (khong the sai chinh ta), chi trang la anh moi OCR bang PaddleOCR-VL:
  - PDF: moi trang la "scan" khi KHONG co chu hien thi (lop chu an che do ve 3 cua OCR cu nhet vao ban scan: o day lop do khong dau / sai chinh ta,
    khong duoc tin) hoac co anh phu >= OCR_SCAN_IMAGE_COVER trang (than van ban nam trong anh, chu hien thi chi la phan chen them). Trang scan
    gom lai thanh PDF con gui sang paddle_vl.parse theo lo; trang con lai doc lop chu bang PyMuPDF (bang co ke -> find_tables), moi trang doi chieu
    so ky tu voi lop chu goc, thieu thi lui ve van ban thuan cua trang.
  - DOCX: doc thang XML (python-docx), gom bang co o gop va bang long nhau.
  - Anh: OCR.

shortcut: DOCX bo qua hop van ban (text box), header / footer va chu trong anh nhung; PDF bang khong co duong ke doc nhu van ban thuong;
PDF ma hoa chu hong font (VNI / TCVN3) doc ra ky tu rac, chi nhan ra duoc khi font khong anh xa duoc sang Unicode. Nang cap khi gap tai lieu thuc te."""
import io
import logging
import threading
from dataclasses import dataclass

import pymupdf
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from .. import config as cfg
from . import paddle_vl
from .markdown import md_table
from .paddle_vl import OcrUnavailable

log = logging.getLogger("uvicorn.error")

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}
_MIN_KEPT = 0.98  # ti le ky tu toi thieu cua lop chu goc con lai sau khi dung bang; thap hon thi lui ve van ban thuan
_lock = threading.Lock()  # PyMuPDF khong thread-safe, va service PaddleOCR-VL chi xu ly mot request mot luc


class UnsupportedFormat(ValueError):
    pass


class UnreadableFile(ValueError):
    pass


@dataclass
class Result:
    text: str
    pages: int | None = None  # None: DOCX khong co khai niem trang
    ocr_pages: int = 0  # so trang phai OCR (scan); con lai doc thang


def _extension(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def is_supported(filename):
    return _extension(filename) in IMAGE_EXTENSIONS | {"pdf", "docx"}


def extract(data, filename):
    ext = _extension(filename)
    if not is_supported(filename):
        raise UnsupportedFormat(f"Khong ho tro dinh dang: {ext or filename}")
    with _lock:
        result = {"pdf": _pdf, "docx": _docx}.get(ext, _image)(data)
    result.text = result.text.replace("\x00", "").strip()  # Postgres TEXT khong nhan NUL
    return result


# ── PDF ──────────────────────────────────────────────────────────────────────────

def _chars(text):
    return len("".join(text.split()))


def _image_cover(page):
    area = page.rect.width * page.rect.height
    return min(1.0, sum((b[2] - b[0]) * (b[3] - b[1]) for b in (i["bbox"] for i in page.get_image_info())) / area)


def _is_scan(page):
    visible = sum(len(span["chars"]) for span in page.get_texttrace() if span["type"] != 3)  # type 3 = chu vo hinh (lop OCR an)
    return visible == 0 or _image_cover(page) >= cfg.OCR_SCAN_IMAGE_COVER


def _inside(block, table):
    """Tam cua khoi chu (x0, y0, x1, y1, ...) nam trong khung cua bang."""
    cx, cy = (block[0] + block[2]) / 2, (block[1] + block[3]) / 2
    return table.bbox[0] <= cx <= table.bbox[2] and table.bbox[1] <= cy <= table.bbox[3]


def _layer_text(page):
    """Van ban cua trang tu lop chu, bang co ke dung thanh bang Markdown, dung thu tu doc theo toa do."""
    raw = page.get_text()
    try:
        tables = page.find_tables().tables
    except Exception:  # find_tables loi tren trang la: van co van ban thuan
        tables = []
    if not tables:
        return raw.strip()

    items = [(b[1], b[0], b[4].strip()) for b in page.get_text("blocks", sort=True)
             if b[6] == 0 and not any(_inside(b, t) for t in tables)]
    kept = sum(_chars(i[2]) for i in items)
    for t in tables:
        rows = t.extract()
        kept += sum(_chars(c or "") for r in rows for c in r)
        items.append((t.bbox[1], t.bbox[0], md_table(rows)))
    if kept < _MIN_KEPT * _chars(raw):
        log.warning("trang %d: dung bang lam mat chu (%d/%d ky tu), dung van ban thuan", page.number + 1, kept, _chars(raw))
        return raw.strip()
    return "\n".join(text for _, _, text in sorted(items, key=lambda i: (i[0], i[1])) if text)


def _ocr_pages(doc, indexes):
    """Markdown cua cac trang scan `indexes` (theo thu tu), gui theo lo OCR_PAGES_PER_REQUEST trang."""
    out = []
    for start in range(0, len(indexes), cfg.OCR_PAGES_PER_REQUEST):
        batch = indexes[start:start + cfg.OCR_PAGES_PER_REQUEST]
        with pymupdf.open() as sub:
            for i in batch:
                sub.insert_pdf(doc, from_page=i, to_page=i)
            texts = paddle_vl.parse(sub.tobytes(deflate=True, garbage=3), pdf=True)
        if len(texts) != len(batch):  # lech trang = mat hoac xe dich noi dung: that bai con hon im lang
            raise OcrUnavailable(f"PaddleOCR-VL tra {len(texts)} trang cho lo {len(batch)} trang")
        out += texts
    return out


def _pdf(data):
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:
        raise UnreadableFile(f"Khong doc duoc PDF: {e}") from e
    with doc:
        if doc.needs_pass:
            raise UnreadableFile("PDF co mat khau")
        pages = [None if _is_scan(page) else _layer_text(page) for page in doc]
        scans = [i for i, text in enumerate(pages) if text is None]
        for i, text in zip(scans, _ocr_pages(doc, scans), strict=True):
            pages[i] = text
        log.info("OCR: PDF %d trang, %d trang scan", len(pages), len(scans))
        return Result("\n\n".join(pages), len(pages), len(scans))


# ── Anh ──────────────────────────────────────────────────────────────────────────

def _image(data):
    pages = paddle_vl.parse(data, pdf=False)  # service tu doc anh va tu tach trang TIFF; anh hong thanh BadDocument
    return Result("\n\n".join(pages), len(pages), len(pages))


# ── DOCX ─────────────────────────────────────────────────────────────────────────

def _docx_rows(table):
    """Hang -> o theo tung <w:tc> that (o gop ngang them o rong cho du cot, khong lap chu)."""
    rows = []
    for tr in table._tbl.tr_lst:
        row = []
        for tc in tr.tc_lst:
            row += [_docx_cell(_Cell(tc, table))] + [""] * (tc.grid_span - 1)
        rows.append(row)
    return rows


def _docx_cell(cell):
    nested = [c for t in cell.tables for r in _docx_rows(t) for c in r]  # bang long trong o: giu chu, khong giu cau truc
    return " ".join([p.text for p in cell.paragraphs] + nested)


def _docx(data):
    try:
        doc = Document(io.BytesIO(data))
    except Exception as e:
        raise UnreadableFile(f"Khong doc duoc DOCX: {e}") from e
    out = []
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            out.append(Paragraph(child, doc).text)
        elif child.tag == qn("w:tbl"):
            out.append(md_table(_docx_rows(Table(child, doc))))
    return Result("\n".join(out))
