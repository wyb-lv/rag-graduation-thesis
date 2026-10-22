"""Script thực nghiệm so sánh đối chứng (Ablation Study):

1. So sánh chuẩn hóa văn bản TTS (Đọc thô vs Đọc chuẩn hóa).
2. So sánh độ trễ phát âm thanh đầu tiên TTFA (Chờ tuần tự vs Streaming tách câu).
3. So sánh sửa lỗi thuật ngữ học vụ (Whisper Raw vs Phonetic Fix).
"""
import sys
import time
from pathlib import Path

# Đảm bảo import được module từ app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import speech


def benchmark_academic_normalizer():
    """Thực nghiệm 1: So sánh văn bản đầu vào của bộ đọc TTS."""
    print("\n" + "=" * 70)
    print("THỰC NGHIỆM 1: CHUẨN HÓA VĂN BẢN HỌC VỤ CHO GIỌNG ĐỌC TTS")
    print("=" * 70)

    test_cases = [
        "Theo [QuyDinhHocBong.pdf, Điều 5, Khoản 2], sinh viên cần đạt GPA 3.60 - 4.00.",
        "Điểm rèn luyện đạt từ 15/30 điểm trở lên mới đủ điều kiện xét tuyển ĐH.",
        "Học sinh THPT có chứng chỉ IELTS 6.5 được miễn thi tiếng Anh theo QĐ của trường.",
    ]

    for i, raw in enumerate(test_cases, 1):
        baseline = speech.normalize_academic_text(raw, enable_academic_norm=False)
        optimized = speech.normalize_academic_text(raw, enable_academic_norm=True)

        print(f"\n[Mẫu {i}]:")
        print(f"  - Bản gốc (Baseline) : {baseline}")
        print(f"  - Chuẩn hóa (Optimized): {optimized}")


def benchmark_latency_ttfa():
    """Thực nghiệm 2: Đo độ trễ thời gian chờ âm thanh đầu tiên (TTFA)."""
    print("\n" + "=" * 70)
    print("THỰC NGHIỆM 2: ĐỘ TRỄ ÂM THANH ĐẦU TIÊN (TTFA: TIME-TO-FIRST-AUDIO)")
    print("=" * 70)

    # Giả lập luồng token stream từ LLM (mỗi token trễ ~40ms giống tốc độ Groq)
    simulated_tokens = [
        "Quy ", "chế ", "học ", "vụ ", "năm ", "2024 ", "quy ", "định ", "rõ. ",  # Câu 1 (~9 token)
        "Sinh ", "viên ", "phải ", "hoàn ", "thành ", "tối ", "thiểu ", "15 ", "tín ", "chỉ. ",  # Câu 2
        "Nếu ", "bị ", "điểm ", "F ", "thì ", "phải ", "đăng ", "ký ", "học ", "lại.",  # Câu 3
    ]

    def token_stream():
        for token in simulated_tokens:
            time.sleep(0.04)  # 40ms mỗi token
            yield token

    # 1. Chạy chế độ Thông thường (Baseline - Đợi hết bài)
    start_base = time.perf_counter()
    chunks_base = list(speech.prepare_tts_chunks(token_stream(), enable_sentence_chunking=False))
    ttfa_base = chunks_base[0]["elapsed_ms"] if chunks_base else 0

    # 2. Chạy chế độ Tách câu song song (Optimized)
    start_opt = time.perf_counter()
    chunks_opt = list(speech.prepare_tts_chunks(token_stream(), enable_sentence_chunking=True))
    ttfa_opt = chunks_opt[0]["elapsed_ms"] if chunks_opt else 0

    print(f"\n- Thời gian chờ câu đầu tiên ở cách Thông thường (Baseline) : {ttfa_base:.1f} ms")
    print(f"- Thời gian chờ câu đầu tiên ở cách Tối ưu (Optimized)        : {ttfa_opt:.1f} ms")
    if ttfa_base > 0:
        improvement = ((ttfa_base - ttfa_opt) / ttfa_base) * 100
        print(f"==> Tốc độ phát âm thanh câu đầu tiên nhanh hơn: {improvement:.1f}%!")


def benchmark_phonetic_correction():
    """Thực nghiệm 3: So sánh độ chính xác thuật ngữ trước và sau sửa lỗi."""
    print("\n" + "=" * 70)
    print("THỰC NGHIỆM 3: ĐỘ CHÍNH XÁC THUẬT NGỮ HỌC VỤ (PHONETIC CORRECTION)")
    print("=" * 70)

    raw_transcriptions = [
        "xét tuyển ngành công nghệ thông tin bằng khối a không không",
        "em muốn hỏi điều kiện để đạt học bổng di pi ây bao nhiêu",
        "nếu em thi môn giải tích bị điểm ép thì có được thi lại không",
        "chuẩn đầu ra tiếng anh có chấp nhận chứng chỉ ai eo không",
    ]

    print(f"{'Whisper thô (Baseline)':<40} | {'Sau sửa lỗi (Optimized)'}")
    print("-" * 75)
    for raw in raw_transcriptions:
        corrected = speech.apply_phonetic_fix(raw)
        print(f"{raw:<40} | {corrected}")


if __name__ == "__main__":
    benchmark_academic_normalizer()
    benchmark_latency_ttfa()
    benchmark_phonetic_correction()
    print("\n" + "=" * 70)
    print("HOÀN THÀNH TẤT CẢ CÁC BÀI THỰC NGHIỆM!")