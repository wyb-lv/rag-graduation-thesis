"""Nhận diện tiêu đề Điều trên văn bản OCR hỏng. Mỗi dạng lỗi dưới đây lấy từ văn bản thật trong kho (77 văn bản; trước khi sửa 26/45
văn bản có cấu trúc bị mất hoặc gắn nhầm tiêu đề, xem PROJECT_CONTEXT mục 9)."""
import pytest

from app.ingest.chunker import _find_sections, _heading, chunk_document, clean_lines


def body(topic):
    return (f"1. Nội dung quy định về {topic} được áp dụng cho toàn thể sinh viên và học viên của trường trong suốt khóa học.\n"
            f"2. Các đơn vị liên quan có trách nhiệm hướng dẫn và kiểm tra việc thực hiện quy định về {topic} theo đúng thời hạn.\n")


def labels(text):
    return [c.dieu for c in chunk_document(text, "Quy định mẫu")]


def test_heading_rendered_as_a_table_row_is_not_dropped_as_noise():
    # "| Điều 3. Học phí | / |  |  |": bộ lọc "dòng toàn ký hiệu" từng xóa mất dòng này nên nội dung Điều 3 bị gắn vào Điều 2
    text = (f"Điều 1. Phạm vi\n{body('phạm vi')}Điều 2. Giải thích\n{body('giải thích')}"
            f"| Điều 3. Học phí | / |  |  |\n{body('học phí')}Điều 4. Nghĩa vụ\n{body('nghĩa vụ')}")
    assert labels(text) == ["Điều 1. Phạm vi", "Điều 2. Giải thích", "Điều 3. Học phí", "Điều 4. Nghĩa vụ"]
    third = chunk_document(text, "Quy định mẫu")[2].text
    assert "học phí" in third and "giải thích" not in third


def test_table_row_heading_with_junk_cells_around_it_is_found():
    assert _heading("| : | Điều 12. Trách nhiệm của Phòng Tài chính - Kế toán |") == (12, "Trách nhiệm của Phòng Tài chính - Kế toán")
    assert "Điều 8. Đánh giá về ý thức" in clean_lines("| ! | .. Điều 8. Đánh giá về ý thức công dân |  |")[0]


@pytest.mark.parametrize("line, expected", [
    ("Điều 9, Xử lý học phí khi SV/ HV nghỉ học", (9, "Xử lý học phí khi SV/ HV nghỉ học")),  # OCR đọc "." thành ","
    ("Điều 9; Quy định cách ghi số hiệu phôi văn bằng", (9, "Quy định cách ghi số hiệu phôi văn bằng")),
    ("Điều 9. Xử lý", (9, "Xử lý")),
    ("! Điều 9. Trách nhiệm của Khoa, Viện", (9, "Trách nhiệm của Khoa, Viện")),  # ký tự rác đứng trước
    (". Điều 5. Đánh giá về ý thức tham gia học tập", (5, "Đánh giá về ý thức tham gia học tập")),
    ("' Điều 6. Chuẩn đầu ra tiếng Anh", (6, "Chuẩn đầu ra tiếng Anh")),
    ("Ei6u 4. Hinh vi vi ph4m liOm chinh hgc thu$t", (4, "Hinh vi vi ph4m liOm chinh hgc thu$t")),  # "Điều" bị đọc thành "Ei6u"
    ("Điều 15. Tố chức thực hiện | Ĩg\\", (15, "Tố chức thực hiện")),  # ô bảng thừa phía sau không vào tiêu đề
])
def test_garbled_headings_are_recognized_and_titles_are_clean(line, expected):
    assert _heading(line) == expected


@pytest.mark.parametrize("line", [
    "Điều 9, 10 và 11 của Quy chế này quy định về",  # dẫn chiếu đầu dòng do OCR xuống hàng giữa câu
    "Điều 9, khoản 2 quy định rằng sinh viên",
    "Điều 9, Khoản 2 của Quy định này",
    "Điều 9, Điều 10 và Điều 11 quy định",
    "Điều 9 của Quy định này",
    "theo Điều 4. Nghĩa vụ",
    "Điều kiện xét tốt nghiệp",
])
def test_cross_references_and_plain_sentences_are_not_headings(line):
    assert _heading(line) is None


def test_cover_decision_heading_does_not_hijack_the_body_when_a_body_heading_is_lost():
    # Cấu trúc của "Quy định về học phí": bìa Quyết định có Điều 1-3, thân bài mất tiêu đề Điều 3. Trước đây "Điều 3" của bìa
    # thắng ("lần xuất hiện cuối") và loại luôn Điều 1, 2 thật; toàn bộ phạm vi/giải thích từ ngữ bị gắn dưới "Các Ông/ Bà ...".
    text = ("QUYẾT ĐỊNH\nĐiều 1, Ban hành kèm theo Quyết định này Quy định chính sách học phí.\n"
            "Điều 2. Quyết định này có hiệu lực kể từ ngày ký và thay thế quyết định cũ.\n"
            "Điều 3. Các Ông/ Bà Trưởng phòng, Trưởng khoa, Trung tâm, Viện chịu trách nhiệm thi hành.\nHIỆU TRƯỞNG\n"
            f"Điều 1. Phạm vi và đối tượng áp dụng\n{body('phạm vi')}Điều 2. Giải thích từ ngữ\n{body('thuật ngữ')}{body('thời gian đào tạo')}"
            f"Điều 4. Nghĩa vụ nộp học phí\n{body('nghĩa vụ')}Điều 5. Thời hạn nộp học phí\n{body('thời hạn')}")
    got = labels(text)
    assert got[:2] == ["Điều 1. Phạm vi và đối tượng áp dụng", "Điều 2. Giải thích từ ngữ"]
    assert not any("chịu trách" in g or "Ban hành" in g for g in got)
    assert "Điều 3. Các Ông/ Bà" not in "".join(got)


def test_a_high_numbered_toc_line_without_dots_does_not_block_the_real_headings():
    # Văn bản "Quy chế tuyển sinh": mục lục không có dấu chấm nên không bị loại; dòng "Điều 8" của nó chặn Điều 1,2,3,5,6,7 thật
    toc = "Điều 4. Yêu cầu chung trong tuyển sinh :- xx\nĐiều 8. Cơ sở dữ liệu tuyển sinh :- xx\n"
    real = "".join(f"Điều {n}. Mục {n}\n{body(f'mục {n}')}" for n in (1, 2, 3, 5, 6, 7, 9))
    assert [d.split(".")[0] for d in labels(toc + real)] == ["Điều 1", "Điều 2", "Điều 3", "Điều 5", "Điều 6", "Điều 7", "Điều 9"]


def test_a_stray_lower_number_in_the_middle_of_the_body_is_ignored():
    real = "".join(f"Điều {n}. Mục {n}\n{body(f'mục {n}')}" for n in (1, 2, 3, 4))
    stray = "Điều 2. Chương trình đào tạo được nêu ở trên\n"  # dòng tình cờ bắt đầu bằng "Điều 2." giữa Điều 4 và Điều 5
    tail = "".join(f"Điều {n}. Mục {n}\n{body(f'mục {n}')}" for n in (5, 6))
    assert [d.split(".")[0] for d in labels(real + stray + tail)] == [f"Điều {n}" for n in range(1, 7)]


def test_the_longest_increasing_chain_prefers_the_later_compact_one_on_a_tie():
    lines = ["Điều 1. Ban hành kèm theo", "Điều 2. Hiệu lực", "Điều 3. Trách nhiệm", "x", "Điều 1. Phạm vi", "Điều 2. Mục đích", "Điều 3. Quyền"]
    assert [i for i, _, _ in _find_sections(lines)] == [4, 5, 6]


def test_no_headings_gives_no_sections():
    assert _find_sections(["Thông báo", "Sinh viên nộp hồ sơ trước ngày 30/9"]) == []
