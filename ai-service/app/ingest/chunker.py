"""Làm sạch văn bản OCR của quy chế/quy định và chia chunk theo cấu trúc Chương -> Điều -> khoản -> điểm.

Văn bản đầu vào thường là PDF scan được OCR nên có nhiễu: trang bìa Quyết định, mục lục,
header/footer lặp mỗi trang, số trang, và rác sinh ra từ con dấu/chữ ký/hình vẽ/sơ đồ.
Mỗi chunk được gắn tiền tố "[Tên tài liệu – Chương – Điều]" để embedding và câu trả lời biết ngữ cảnh.
"""
import re
from dataclasses import dataclass
from typing import Optional

MAX_CHARS = 1500       # chunk dài hơn mức này thì tách theo khoản
TARGET_CHARS = 1200    # mức gom khoản khi tách
MIN_CHARS = 150        # chunk ngắn hơn mức này thì gộp vào chunk kề

# OCR hay làm hỏng dấu ("Piéu", "Ei6u", "Điểu", "Chuong") nên regex tiêu đề để mờ: chấp nhận ký tự rác/ô bảng đứng trước ("! Điều 9.",
# "| : | Điều 12.", ". Điều 5.") và dấu chấm bị đọc thành "," hoặc ";" ("Điều 9, Xử lý ..."; xem _heading).
_DIEU = re.compile(r"^[\s|!*#~_:.,;'\"‘’“”`]*(?:Đ|D|P|B|E)\S{0,3}u\s*(\d{1,3})\s*([.:,;])\s*(.*)$", re.I)
_CHUONG = re.compile(r"^\s*Ch\S{1,4}ng\s+([IVXLC1lJ]{1,6})\b[.\s]*(.*)$", re.I)
_PHULUC = re.compile(r"^\s*PH[UỤ]\s*L[UỤ]C\b(.*)$", re.I)
_NOI_NHAN = re.compile(r"^\s*N[oơ]i\s+nh[aậ]n\b", re.I)
_KHOAN = re.compile(r"^\s*\d{1,2}\s*[.)]\s+\S")
_DIEM = re.compile(r"^\s*(?:[a-zđ]\)|[-•+–]\s)\s*\S")
_PAGE_NUM = re.compile(r"^\s*[|\[(]?\s*(?:Trang\s*[|:]?\s*)?\d{1,3}\s*[|\])]?\s*$", re.I)
_RUNNING_HEADER = re.compile(r"\bM[aãàáạ]\s*(?:hi[eêéệ]u|t[aàá]i\s*li[eêéệ]u)\b\s*:", re.I)
_TOC_LINE = re.compile(r"\.{5,}|…{3,}")


def _heading(s: str) -> Optional[tuple[int, str]]:
    """(số Điều, tiêu đề) nếu dòng là tiêu đề Điều, không thì None."""
    m = _DIEU.match(s)
    if not m:
        return None
    title = re.split(r"\s*\|", m.group(3))[0].strip()  # tiêu đề dựng thành hàng bảng: bỏ các ô phía sau
    if m.group(2) in ",;" and (not title[:1].isupper() or re.match(r"(?:Điều|Khoản|Điểm)\b", title)):
        return None  # "Điều 9, 10 và 11 của ..." là dẫn chiếu giữa câu, không phải tiêu đề
    return int(m.group(1)), title


def _unpipe(s: str) -> Optional[str]:
    """Tiêu đề Điều/Chương bị OCR dựng thành hàng bảng ("| Điều 3. Học phí | / |  |"): trả về ô chứa tiêu đề."""
    for cell in s.split("|"):
        c = cell.strip()
        if _heading(c) or _CHUONG.match(c):
            return c
    return None


@dataclass
class Chunk:
    text: str
    dieu: Optional[str] = None
    chuong: Optional[str] = None


def clean_lines(text: str) -> list[str]:
    """Bỏ dòng nhiễu: số trang, header lặp, dòng mục lục, rác từ con dấu/hình/sơ đồ."""
    out = []
    for raw in text.replace("\u0000", "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if "|" in s:
            s = _unpipe(s) or s  # không để bộ lọc "dòng toàn ký hiệu" xóa mất tiêu đề nằm trong hàng bảng
        if _PAGE_NUM.match(s) or _RUNNING_HEADER.search(s) or _TOC_LINE.search(s):
            continue
        letters = sum(c.isalpha() for c in s)
        digits = sum(c.isdigit() for c in s)
        # Dòng toàn ký hiệu: hình vẽ, con dấu, chữ ký. Dòng bảng nhiều số vẫn giữ.
        if len(s) >= 4 and letters / len(s) < 0.5 and digits / len(s) < 0.3:
            continue
        if len(s) <= 2:
            continue
        out.append(s)
    return out


def _join_units(lines: list[str]) -> list[str]:
    """Nối các dòng bị xuống hàng cứng của OCR thành đơn vị khoản/điểm/dòng bảng."""
    units: list[str] = []
    for s in lines:
        starts_new = _KHOAN.match(s) or _DIEM.match(s) or "|" in s
        prev_table = bool(units) and "|" in units[-1]
        if units and not starts_new and not prev_table:
            units[-1] += " " + s
        else:
            units.append(s)
    return units


def _pack(units: list[str], limit: int) -> list[str]:
    """Gom đơn vị liền nhau thành nhóm không vượt limit; không tách giữa một khoản và các điểm của nó."""
    groups: list[list[str]] = []
    for u in units:
        if _KHOAN.match(u) or not groups:
            groups.append([u])
        else:
            groups[-1].append(u)
    chunks: list[str] = []
    cur = ""
    for g in groups:
        body = "\n".join(g)
        if len(body) > limit:  # một khoản quá dài: tách theo điểm / dòng, cuối cùng theo câu
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_split_long(g, limit))
        elif cur and len(cur) + 1 + len(body) > limit:
            chunks.append(cur)
            cur = body
        else:
            cur = f"{cur}\n{body}" if cur else body
    if cur:
        chunks.append(cur)
    return chunks


def _split_long(units: list[str], limit: int) -> list[str]:
    parts: list[str] = []
    cur = ""
    for u in units:
        # Câu đơn lẻ vẫn quá dài (không có cấu trúc): cắt theo dấu chấm
        pieces = [u] if len(u) <= limit else re.split(r"(?<=[.;])\s+", u)
        for p in pieces:
            if cur and len(cur) + 1 + len(p) > limit:
                parts.append(cur)
                cur = p
            else:
                cur = f"{cur}\n{p}" if cur else p
    if cur:
        parts.append(cur)
    return parts


def _find_sections(lines: list[str]) -> list[tuple[int, int, str]]:
    """Các tiêu đề Điều thuộc phần thân: chuỗi tiêu đề có số tăng dần tốt nhất, điểm = số tiêu đề trong chuỗi trừ số tiêu đề bị
    bỏ qua NẰM GIỮA chuỗi (hòa thì lấy chuỗi bắt đầu muộn hơn, tức gọn hơn). Bìa Quyết định ("Điều 1-3"), mục lục và dẫn chiếu lạc
    dòng không vào chuỗi: nối bìa sang thân bài sẽ phải bỏ qua cả các Điều 1, 2 thật nằm giữa, còn một dòng nhiễu lạc chỉ bị bỏ
    qua một lần. (Cách cũ "lần xuất hiện cuối + lọc tham lam" bị một dòng mục lục số cao đứng sớm chặn hết tiêu đề thật phía sau.)
    """
    heads = [(i, *h) for i, s in enumerate(lines) if (h := _heading(s))]
    best: list[tuple[int, int, int]] = []  # theo từng tiêu đề: (điểm chuỗi kết thúc ở đó, vị trí bắt đầu, tiêu đề đứng trước)
    for j, (pos, n, _) in enumerate(heads):
        cand = (1, pos, -1)
        for k in range(j):
            if heads[k][1] < n:
                cand = max(cand, (best[k][0] + 1 - (j - k - 1), best[k][1], k))
        best.append(cand)
    if not best:
        return []
    j = max(range(len(heads)), key=lambda j: (best[j][0], best[j][1], heads[j][0]))
    chain = []
    while j >= 0:
        chain.append(heads[j])
        j = best[j][2]
    return chain[::-1]


def _chuong_label(m: re.Match) -> str:
    num = m.group(1).upper().translate(str.maketrans("J1L", "III"))  # OCR đọc "I" thành J/1/l
    title = m.group(2).strip()
    return f"Chương {num}" + (f" {title}" if title else "")


def chunk_document(text: str, source: str) -> list[Chunk]:
    lines = clean_lines(text)
    if not lines:
        return []
    sections = _find_sections(lines)
    if len(sections) < 3:
        return _chunk_unstructured(lines, source)

    chunks: list[Chunk] = []
    chuong: Optional[str] = None
    for s in lines[max(0, sections[0][0] - 5):sections[0][0]]:  # Chương I nằm ngay trước Điều 1
        if _CHUONG.match(s):
            chuong = _chuong_label(_CHUONG.match(s))
    starts = [i for i, _, _ in sections] + [len(lines)]
    for k, (i, n, title) in enumerate(sections):
        body_lines = []
        skipping = False
        for s in lines[i + 1:starts[k + 1]]:
            mc = _CHUONG.match(s)
            if mc:
                chuong = _chuong_label(mc)
                continue
            if _NOI_NHAN.match(s):
                skipping = True  # khối nơi nhận + chữ ký cuối văn bản
            if _PHULUC.match(s):
                skipping = False
            if not skipping:
                body_lines.append(s)
        dieu = f"Điều {n}" + (f". {title}" if title else "")
        prefix = f"[{source}" + (f" – {chuong}" if chuong else "") + f" – {dieu}]"
        units = _join_units(body_lines)
        body = "\n".join(units)
        if not body.strip() and not title:
            continue
        pieces = [body] if len(prefix) + len(body) <= MAX_CHARS else _pack(units, TARGET_CHARS)
        for p in pieces or [body]:
            chunks.append(Chunk(f"{prefix}\n{p}".strip(), dieu=dieu, chuong=chuong))
    return _merge_tiny(chunks)


def _chunk_unstructured(lines: list[str], source: str) -> list[Chunk]:
    """Thông báo, hướng dẫn, biểu mẫu: không có Điều, gom theo khoản/đoạn."""
    for i, s in enumerate(lines):  # bỏ khối nơi nhận + chữ ký cuối văn bản
        if _NOI_NHAN.match(s):
            lines = lines[:i]
            break
    units = _join_units(lines)
    prefix = f"[{source}]"
    return _merge_tiny([Chunk(f"{prefix}\n{p}") for p in _pack(units, TARGET_CHARS)])


def _merge_tiny(chunks: list[Chunk]) -> list[Chunk]:
    """Gộp chunk quá ngắn vào chunk kề sau (chunk cuối thì gộp vào chunk trước)."""
    out: list[Chunk] = []
    for c in chunks:
        if out and len(out[-1].text) < MIN_CHARS:
            prev = out.pop()
            c = Chunk(f"{prev.text}\n{c.text}", dieu=prev.dieu, chuong=prev.chuong)
        out.append(c)
    if len(out) > 1 and len(out[-1].text) < MIN_CHARS:
        last = out.pop()
        out[-1].text += f"\n{last.text}"
    return out


def infer_metadata(source: str) -> dict:
    """Suy ra loại văn bản và khóa áp dụng (khoa_tu..khoa_den, thiếu = không giới hạn) từ tên tài liệu."""
    s = source.lower()
    if re.match(r"(thông báo|tb\d*\b)", s):
        loai = "thong_bao"
    elif re.match(r"(đơn|báo cáo|giấy xác nhận|danh sách|phiếu)", s):
        loai = "bieu_mau"
    elif s.startswith("hướng dẫn"):
        loai = "huong_dan"
    elif s.startswith("quy chế"):
        loai = "quy_che"
    elif s.startswith(("quy định", "quy đinh", "quy trình")):
        loai = "quy_dinh"
    else:
        loai = "khac"
    meta = {"loai_van_ban": loai}
    m = re.search(r"kh[oó][aáà]\s*(\d{4}(?:\s*[,\-–]\s*\d{4})*)(.{0,25})", s)
    if m:
        years = [int(y) for y in re.findall(r"\d{4}", m.group(1))]
        meta["khoa"] = re.sub(r"\s+", "", m.group(1))
        after, before = m.group(2), s[:m.start()]
        if "trở về trước" in after:
            meta["khoa_den"] = max(years)
        elif "trở đi" in after or "trở về sau" in after or before.rstrip().endswith("từ"):
            meta["khoa_tu"] = min(years)
        else:
            meta["khoa_tu"], meta["khoa_den"] = min(years), max(years)
    return meta
