"""Nhiễu của OCR mới (PaddleOCR-VL xuất markdown): tiêu đề có "## ", VLM lặp vô hạn một từ, mục lục thiếu dấu chấm dẫn."""
from app.ingest.chunker import chunk_document, clean_lines

DIEU = """Điều 1. Phạm vi điều chỉnh
1. Quy chế này áp dụng cho sinh viên hệ chính quy toàn trường và các đơn vị có liên quan đến việc tổ chức đào tạo.
Điều 2. Mục đích
Bảo đảm quyền lợi của người học và sự công bằng trong đánh giá kết quả học tập, rèn luyện của sinh viên trong toàn trường.
Điều 3. Tổ chức thực hiện
Các đơn vị thuộc Trường có trách nhiệm thực hiện đúng Quy chế này và báo cáo kết quả vào cuối mỗi học kỳ chính.
"""


def test_markdown_prefixed_chuong_is_recognised():
    chunks = chunk_document("## CHƯƠNG I. QUY ĐỊNH CHUNG\n" + DIEU, "Quy chế mẫu")
    assert all(c.chuong == "Chương I QUY ĐỊNH CHUNG" for c in chunks)
    assert "CHƯƠNG" not in "\n".join(c.text.split("\n", 1)[1] for c in chunks)  # tiêu đề Chương không lọt vào thân chunk


def test_chuong_on_a_toc_line_with_a_page_number_is_not_used_as_label():
    chunks = chunk_document("Chương III QUYỀN HẠN VÀ TRÁCH NHIỆM 12\n" + DIEU, "Quy chế mẫu")
    assert all(c.chuong is None for c in chunks)


def test_endless_repetition_is_collapsed():
    assert clean_lines("| ĐẠI HỌC HOA SEN " + "PHẠI " * 800 + "|") == ["| ĐẠI HỌC HOA SEN PHẠI |"]
    assert clean_lines("Giờ hài " * 50) == ["Giờ hài"]  # cụm hai từ lặp


def test_ordinary_repetition_is_left_alone():
    line = "Sinh viên được điểm 0 0 0 nếu vi phạm quy chế thi ba lần liên tiếp."
    assert clean_lines(line) == [line]


def test_form_appendix_heading_with_blank_dot_tail_is_still_skipped():
    out = "\n".join(c.text for c in chunk_document(
        DIEU + "| Phụ lục IV: Mẫu danh sách sinh viên đề nghị công nhận tốt nghiệp KHOA................\nDanh sách sinh viên cần điền vào đây.\n", "Quy chế mẫu"))
    assert "Danh sách sinh viên cần điền" not in out


def test_page_counter_n_over_m_is_dropped():
    assert clean_lines("18/27") == [] and clean_lines("Trang 5") == []


def test_toc_entry_of_an_appendix_is_still_dropped():
    assert clean_lines("Phụ lục IV. Các mẫu thẻ sinh viên ..... 28") == []
