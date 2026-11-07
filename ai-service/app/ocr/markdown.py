"""Bang Markdown: dung tu danh sach hang (bang that cua PDF / DOCX) va doi bang HTML (PaddleOCR-VL xuat bang dang <table>) thanh bang Markdown de
chunker / embedding doc nhu van ban thuong."""
import re
from html.parser import HTMLParser


def _cell(text):
    return " ".join((text or "").split()).replace("|", "/")


def md_table(rows):
    """[[o]] -> bang Markdown (hang dau la tieu de); hang rong bi bo; dong thieu o duoc them o trong."""
    rows = [[_cell(c) for c in r] for r in rows if any(_cell(c) for c in r)]
    if not rows:
        return ""
    cols = max(len(r) for r in rows)
    lines = ["|" + "".join(f" {r[c] if c < len(r) else ''} |" for c in range(cols)) for r in rows]
    lines.insert(1, "|" + " --- |" * cols)
    return "\n".join(lines)


class _HtmlTable(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell = [], None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = {"text": [], "rs": int(a.get("rowspan") or 1), "cs": int(a.get("colspan") or 1)}
        elif tag == "br" and self._cell is not None:
            self._cell["text"].append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell["text"].append(data)


def html_table_rows(html):
    """<table> -> [[o]]: o gop (rowspan / colspan) chi giu chu o o goc, cac o con lai de trong (khong lap chu)."""
    parser = _HtmlTable()
    parser.feed(html)
    grid = {}
    for r, row in enumerate(parser.rows):
        c = 0
        for cell in row:
            while (r, c) in grid:
                c += 1
            text = "".join(cell["text"])
            for dr in range(cell["rs"]):
                for dc in range(cell["cs"]):
                    grid[(r + dr, c + dc)] = text if dr == dc == 0 else ""
            c += cell["cs"]
    if not grid:
        return []
    n_rows, n_cols = max(r for r, _ in grid) + 1, max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(n_cols)] for r in range(n_rows)]


def html_to_markdown(text):
    """Doi moi <table> trong `text` thanh bang Markdown, bo the <div> / <img> bao quanh (anh khong duoc tra ve nen the <img> vo nghia)."""
    text = re.sub(r"<table\b.*?</table>", lambda m: "\n" + md_table(html_table_rows(m.group(0))) + "\n", text, flags=re.S | re.I)
    return re.sub(r"</?div\b[^>]*>|<img\b[^>]*>", "", text, flags=re.I)
