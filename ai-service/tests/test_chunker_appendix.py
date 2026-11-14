"""Phụ lục biểu mẫu / hướng dẫn dùng phần mềm không vào chunk; phụ lục có nội dung quy định (lưu đồ, bảng quy đổi) vẫn giữ.
Văn bản không có phụ lục như vậy thì chunker không bỏ gì (luật: app/texts/appendix_skip.jsonl)."""
import pytest

from app.ingest.chunker import chunk_document

BODY = """Chương I. QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
1. Quy chế này áp dụng cho sinh viên hệ chính quy toàn trường và các đơn vị có liên quan đến việc tổ chức đào tạo.
Điều 2. Mục đích
Bảo đảm quyền lợi của người học và sự công bằng trong đánh giá kết quả học tập, rèn luyện của sinh viên trong toàn trường.
Điều 3. Tổ chức thực hiện
Các đơn vị thuộc Trường có trách nhiệm thực hiện đúng Quy chế này và báo cáo kết quả vào cuối mỗi học kỳ chính.
"""

FLOWCHART = """PHỤ LỤC I. LƯU ĐỒ QUY TRÌNH XÉT TỐT NGHIỆP
Bước 1: Phòng Đào tạo thông báo mốc thời gian xét tốt nghiệp cho toàn bộ các Khoa trong trường.
"""

FORM = """PHỤ LỤC II. BIỂU MẪU ĐƠN ĐỀ NGHỊ CÔNG NHẬN
Kính gửi: Phòng Đào tạo đại học, họ và tên sinh viên điền vào đây trước khi nộp.
"""


def joined(text):
    return "\n".join(c.text for c in chunk_document(text, "Quy chế mẫu"))


def test_form_appendix_is_dropped_and_flowchart_kept():
    out = joined(BODY + FLOWCHART + FORM)
    assert "Phòng Đào tạo thông báo mốc thời gian" in out
    assert "Kính gửi" not in out and "BIỂU MẪU" not in out


@pytest.mark.parametrize("heading", [
    "## PHỤ LỤC 2: HƯƠNG DẪN SỬ DỤNG HỆ THỐNG ĐĂNG KÝ HỌC PHẦN",  # "## " của OCR markdown, "HƯỚNG" bị đọc thành "HƯƠNG"
    "Phụ lục IV. Các mẫu thẻ sinh viên",
    "PHỤ LỤC 1: MẦU SỐ GÓC",  # "MẪU" bị đọc thành "MẦU"
    "Phụ lục III. Thông tin tuyển sinh",
])
def test_other_skipped_appendix_titles(heading):
    assert "NỘI DUNG PHỤ LỤC" not in joined(BODY + heading + "\nNỘI DUNG PHỤ LỤC phải bị bỏ khỏi chunk.\n")


@pytest.mark.parametrize("title", ["CÁC BIỂU MẪU LIÊN QUAN", "# MẦU PHIẾU TỔNG HỢP (Kèm theo Quyết định số 3256)"])
def test_title_on_the_next_line(title):
    assert "NỘI DUNG PHỤ LỤC" not in joined(BODY + f"PHỤ LỤC II\n{title}\nNỘI DUNG PHỤ LỤC phải bị bỏ khỏi chunk.\n")


def test_skipping_stops_at_the_next_appendix():
    out = joined(BODY + FORM + "PHỤ LỤC III. BẢNG QUY ĐỔI ĐIỂM\nThang điểm 100 quy đổi sang thang điểm 10 theo bảng sau.\n")
    assert "Thang điểm 100 quy đổi" in out and "Kính gửi" not in out


def test_sample_word_in_parentheses_or_at_the_end_is_not_a_form():
    out = joined(BODY + "PHỤ LỤC III. BẢNG TIÊU CHÍ (Áp dụng cho kế hoạch học tập mẫu)\nTiêu chí 1: sinh viên lập kế hoạch học tập.\n"
                 "PHỤ LỤC IV. LỘ TRÌNH HỌC TẬP MẪU\nHọc kỳ 1: các học phần đại cương.\n")
    assert "Tiêu chí 1" in out and "Học kỳ 1" in out


def test_sentence_starting_with_appendix_is_not_a_heading():
    out = joined(BODY + "Phụ lục I quy định mẫu đơn đề nghị và hướng dẫn nộp hồ sơ.\nSinh viên nộp hồ sơ tại Phòng Đào tạo.\n")
    assert "Sinh viên nộp hồ sơ tại Phòng Đào tạo" in out


def test_no_appendix_means_nothing_is_removed():
    assert joined(BODY).count("Điều") >= 3 and "Tổ chức thực hiện" in joined(BODY)
