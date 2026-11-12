"""paddle_vl.parse: layout hay gan nhan `number` cho gia tri that (vd "Tong so tin chi tich luy: 120"), nen chi bo so trang that."""
import httpx

from app import config as cfg
from app.ocr import paddle_vl

HEIGHT = 1000


def block(label, y0, y1, content="", x0=100, x1=300):
    return {"block_label": label, "block_bbox": [x0, y0, x1, y1], "block_content": content}


def parse_page(monkeypatch, blocks, markdown):
    sent = {}

    def fake_post(url, json, timeout):
        sent.update(json)
        pruned = {"width": 800, "height": HEIGHT, "parsing_res_list": blocks}
        body = {"result": {"layoutParsingResults": [{"markdown": {"text": markdown}, "prunedResult": pruned}]}}
        return httpx.Response(200, json=body)

    monkeypatch.setattr(paddle_vl.httpx, "post", fake_post)
    return paddle_vl.parse(b"%PDF", pdf=True)[0], sent


def test_number_is_not_filtered_by_label_on_the_server(monkeypatch):
    _, sent = parse_page(monkeypatch, [], "x")
    assert "number" not in sent["markdownIgnoreLabels"]
    assert "header" in sent["markdownIgnoreLabels"] and "footer" in sent["markdownIgnoreLabels"]


def test_value_on_the_same_row_as_its_label_is_kept_even_near_the_top_edge(monkeypatch):
    blocks = [block("text", 92, 118, "- Tổng số tín chỉ tích lũy:"), block("number", 95, 115, "120", x0=550, x1=590)]
    text, _ = parse_page(monkeypatch, blocks, "- Tổng số tín chỉ tích lũy:\n\n120\n\n- Số tín chỉ khác:")
    assert "120" in text.split("\n\n")


def test_lone_number_at_the_bottom_edge_is_a_page_number_and_is_dropped(monkeypatch):
    blocks = [block("text", 100, 140, "Nội dung"), block("number", 960, 980, "4", x0=480, x1=500)]
    text, _ = parse_page(monkeypatch, blocks, "Nội dung\n\n4")
    assert text == "Nội dung"


def test_lone_number_at_the_top_edge_is_dropped_and_only_the_first_match(monkeypatch):
    blocks = [block("number", 20, 40, "7"), block("text", 300, 340, "7")]
    text, _ = parse_page(monkeypatch, blocks, "7\n\n7")
    assert text == "7"  # khoi o dau trang bi go, doan van cung noi dung o giua trang giu nguyen


def test_lone_number_in_the_middle_of_the_page_is_content(monkeypatch):
    blocks = [block("text", 100, 140, "Nội dung"), block("number", 480, 500, "15")]
    text, _ = parse_page(monkeypatch, blocks, "Nội dung\n\n15")
    assert text == "Nội dung\n\n15"


def test_number_beside_a_dropped_footer_block_is_still_a_page_number(monkeypatch):
    assert "footer" in cfg.OCR_IGNORE_LABELS
    blocks = [block("text", 100, 140, "Nội dung"), block("footer", 955, 985, "ĐH Hoa Sen"), block("number", 960, 980, "3", x0=700, x1=720)]
    text, _ = parse_page(monkeypatch, blocks, "Nội dung\n\n3")
    assert text == "Nội dung"


def test_old_server_without_pruned_result_leaves_the_text_untouched(monkeypatch):
    monkeypatch.setattr(paddle_vl.httpx, "post", lambda *a, **k: httpx.Response(
        200, json={"result": {"layoutParsingResults": [{"markdown": {"text": "A\n\n4"}}]}}))
    assert paddle_vl.parse(b"%PDF", pdf=True) == ["A\n\n4"]
