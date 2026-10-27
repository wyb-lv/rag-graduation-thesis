from app.ingest.chunker import chunk_document, infer_metadata

SAMPLE = """BỘ GIÁO DỤC VÀ ĐÀO TẠO
QUYẾT ĐỊNH
Điều 1. Ban hành kèm theo Quyết định này Quy chế mẫu.
Nơi nhận:
- Lưu VT.
MỤC LỤC
Điều 1. Phạm vi ........ 3
Chương I. QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
1. Quy chế này áp dụng cho sinh viên
hệ chính quy toàn trường.
Mã hiệu: QC-XX/0125/HSU
3
2. Sinh viên phải tuân thủ nội quy của trường trong suốt quá trình học tập tại trường.
Điều 2. Mục đích
Bảo đảm quyền lợi của người học và sự công bằng trong đánh giá kết quả học tập, rèn luyện, đồng thời xác lập trách nhiệm của các đơn vị trong nhà trường.
Chương II. QUYỀN LỢI
Điều 3. Quyền của sinh viên
Sinh viên được học tập, nghiên cứu và sử dụng thư viện theo quy định hiện hành của trường, được tham gia các hoạt động ngoại khóa và được hưởng các chế độ chính sách khác.
— i A Y — |
"""


def test_cleans_cover_toc_and_noise_and_keeps_structure():
    chunks = chunk_document(SAMPLE, "Quy chế mẫu")
    assert [c.dieu for c in chunks] == ["Điều 1. Phạm vi điều chỉnh", "Điều 2. Mục đích", "Điều 3. Quyền của sinh viên"]
    assert chunks[0].chuong.startswith("Chương I") and chunks[2].chuong.startswith("Chương II")
    joined = "\n".join(c.text for c in chunks)
    for noise in ("Mã hiệu", "Nơi nhận", "MỤC LỤC", "Ban hành kèm theo", "— i A Y"):
        assert noise not in joined
    assert "hệ chính quy toàn trường" in chunks[0].text  # dòng bị OCR xuống hàng được nối lại
    assert chunks[0].text.startswith("[Quy chế mẫu – Chương I")


def test_unstructured_notice_becomes_single_chunk():
    chunks = chunk_document("Thông báo nộp chứng chỉ\nSinh viên nộp chứng chỉ trước ngày 30/9 tại Phòng Đào tạo để được xét miễn học phần." * 2, "TB 70")
    assert len(chunks) == 1 and chunks[0].text.startswith("[TB 70]")


def test_infer_metadata():
    assert infer_metadata("Quy định học và CĐR tiếng Anh Khóa 2024 (cập nhật)") == {
        "loai_van_ban": "quy_dinh", "khoa": "2024", "khoa_tu": 2024, "khoa_den": 2024}
    assert infer_metadata("Thông báo 70/TB")["loai_van_ban"] == "thong_bao"


def test_infer_khoa_range():
    assert infer_metadata("Quy định X Khóa 2024")["khoa_tu"] == infer_metadata("Quy định X Khóa 2024")["khoa_den"] == 2024
    only_from = infer_metadata("Quy định X từ Khóa 2026 trở về sau")
    assert only_from["khoa_tu"] == 2026 and "khoa_den" not in only_from
    only_to = infer_metadata("Quy chế (Khóa 2020 trở về trước)")
    assert only_to["khoa_den"] == 2020 and "khoa_tu" not in only_to
    assert (lambda m: (m["khoa_tu"], m["khoa_den"]))(infer_metadata("Quy định Tiếng Anh khóa 2018, 2019")) == (2018, 2019)
    assert "khoa_tu" not in infer_metadata("Quy định điểm danh tham dự lớp học")
