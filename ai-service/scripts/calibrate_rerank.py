"""Đo lại RERANK_MIN_SCORE trên corpus đã nạp trong Qdrant.

Chạy (cần Qdrant có dữ liệu, tải BGE-M3 + reranker):
    python -m scripts.calibrate_rerank [--khoa 2024]

Với mỗi câu hỏi lấy điểm rerank của đoạn tốt nhất. Câu trong phạm vi phải vượt ngưỡng, câu ngoài phạm vi phải
dưới ngưỡng. In bảng từng câu, quét ngưỡng và gợi ý giá trị. Đổi corpus/model thì chạy lại.
"""
import argparse
import sys

from app.rag import query_model as qm

# (câu hỏi, từ khóa chữ thường phải xuất hiện trong tên tài liệu của đoạn đúng)
IN_SCOPE = [
    ("Sinh viên vắng mặt bao nhiêu buổi thì bị cấm thi?", "điểm danh"),
    ("Điểm danh bằng thiết bị di động như thế nào?", "điểm danh"),
    ("Điều kiện để được xét học bổng khuyến khích học tập là gì?", "học bổng"),
    ("Mức học bổng Ưu tú là bao nhiêu phần trăm học phí?", "học bổng"),
    ("Sinh viên được cấp bảng điểm trong trường hợp nào?", "bảng điểm"),
    ("Quy định về việc đóng học phí và xử lý nợ học phí", "học phí"),
    ("Thẻ sinh viên bị mất thì làm thế nào để cấp lại?", "thẻ sinh viên"),
    ("Sinh viên được mượn tối đa bao nhiêu sách ở thư viện?", "thư viện"),
    ("Hành vi đạo văn bị xử lý như thế nào?", "liêm chính"),
    ("Điều kiện xét và công nhận tốt nghiệp đối với sinh viên hệ chính quy?", "tốt nghiệp"),
    ("Sinh viên đạt kết quả tốt nghiệp cao được khen thưởng gì?", "khen thưởng"),
    ("Giấy xác nhận sinh viên được cấp trong bao lâu?", "giấy xác nhận"),
    ("Điều kiện miễn học phần tiếng Anh EII bằng chứng chỉ quốc tế", "tiếng anh"),
    ("Chuẩn đầu ra tiếng Anh đối với sinh viên các ngành không chuyên ngữ", "tiếng anh"),
    ("Sinh viên phải học Giáo dục quốc phòng và an ninh khi nào?", "quốc phòng"),
    ("Quyền và nghĩa vụ của người học là gì?", "người học"),
    ("Điểm rèn luyện được xếp loại như thế nào?", "rèn luyện"),
    ("Sinh viên đăng ký học phần tối đa bao nhiêu tín chỉ mỗi học kỳ?", "học phần"),
    ("Thời gian tối đa để hoàn thành chương trình đại học?", "đào tạo"),
    ("Cố vấn học tập có nhiệm vụ gì?", "cố vấn"),
    ("Chương trình Field trip do ai tổ chức?", "field trip"),
    ("Điều kiện công nhận thực tập nghề nghiệp dưới hình thức tích lũy", "tích lũy"),
    ("Quy định về in ấn và cấp phát văn bằng, chứng chỉ", "văn bằng"),
    ("Quy định sử dụng hệ thống công nghệ thông tin của trường", "công nghệ thông tin"),
    ("Điều kiện để mở ngành đào tạo mới", "mở ngành"),
    ("Giảng viên có trách nhiệm gì trong công tác giảng dạy?", "giảng dạy"),
    ("Phương thức xét tuyển đại học năm 2026", "tuyển sinh"),
]
# Gồm cả câu gần chủ đề nhưng tài liệu không có đáp án
OUT_OF_SCOPE = [
    "Thủ đô của nước Pháp là gì?",
    "Công thức nấu phở bò",
    "Giá vàng hôm nay bao nhiêu?",
    "Ai vô địch World Cup 2022?",
    "Cách cài đặt Python trên Windows",
    "Thời tiết Hà Nội ngày mai thế nào?",
    "Giải phương trình bậc hai x^2 - 5x + 6 = 0",
    "Học phí trường Đại học Bách Khoa là bao nhiêu?",
    "Lương của giảng viên trường Hoa Sen là bao nhiêu?",
    "Ký túc xá của trường có bao nhiêu phòng?",
]


def top_score(question, khoa):
    hits = qm.rerank(question, qm.retrieve(question, "hybrid", khoa=khoa))
    return hits


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--khoa", type=int, default=None)
    khoa = ap.parse_args().khoa

    inside, outside = [], []
    print("== Trong phạm vi: điểm đoạn tốt nhất | đoạn đúng trong top-5? | tài liệu top-1")
    for q, kw in IN_SCOPE:
        hits = top_score(q, khoa)
        ok = any(kw in (h["source"] or "").lower() for h in hits)
        inside.append((hits[0]["rerank_score"], ok))
        print(f"{hits[0]['rerank_score']:7.3f} {'OK ' if ok else 'SAI'} {q[:55]:55} -> {(hits[0]['source'] or '')[:50]}")
    print("\n== Ngoài phạm vi")
    for q in OUT_OF_SCOPE:
        hits = top_score(q, khoa)
        outside.append(hits[0]["rerank_score"])
        print(f"{hits[0]['rerank_score']:7.3f}     {q[:55]:55} -> {(hits[0]['source'] or '')[:50]}")

    good = [s for s, ok in inside if ok]
    print(f"\nTrong phạm vi (đoạn đúng): {len(good)}/{len(inside)} câu, điểm thấp nhất {min(good):.3f}")
    print(f"Ngoài phạm vi: cao nhất {max(outside):.3f}")
    print("\nngưỡng | câu trong phạm vi được trả lời | câu ngoài phạm vi bị chặn")
    for t in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
        kept = sum(s >= t for s, _ in inside)
        blocked = sum(s < t for s in outside)
        print(f"{t:6.2f} | {kept:2d}/{len(inside)} | {blocked:2d}/{len(outside)}")
    lo, hi = max(outside), min(good)
    print(f"\nGợi ý: ngưỡng trong ({lo:.3f}, {hi:.3f})" + (f", điểm giữa {(lo + hi) / 2:.3f}" if lo < hi else " — hai nhóm chồng nhau, cần chọn đánh đổi"))


if __name__ == "__main__":
    main()
