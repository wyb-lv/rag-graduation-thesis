import io
import math
import struct
import wave
import pytest

from app import config as cfg, texts
from app.speech import stt, tts


def make_wav_bytes(freq: float = 440.0, duration_s: float = 0.5, amplitude: int = 12000) -> bytes:
    """Tạo dữ liệu WAV PCM 16-bit mono phục vụ kiểm thử."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            val = int(amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def test_apply_phonetic_fix_replaces_academic_homophones():
    raw = "Cho em hỏi quy chế tính chỉ và điều kiện xét học bỗng tại hắt ét u"
    fixed = stt.apply_phonetic_fix(raw)
    assert "tín chỉ" in fixed
    assert "học bổng" in fixed
    assert "HSU" in fixed


def test_normalize_academic_text_expands_abbreviations_and_symbols():
    raw = "Theo QĐ-ĐHHS năm 2024-2025, SV ngành CNTT cần ĐTBCTL >= 2.5/4.0 và đạt >= 50% số TC."
    normalized = tts.normalize_academic_text(raw)
    assert "Quyết định Đại học Hoa Sen" in normalized
    assert "2024 đến 2025" in normalized
    assert "sinh viên" in normalized
    assert "Công nghệ thông tin" in normalized
    assert "điểm trung bình chung tích lũy" in normalized
    assert "lớn hơn hoặc bằng 2.5 chấm" not in normalized
    assert "2 chấm 5 trên 4 chấm 0" in normalized
    assert "50 phần trăm" in normalized
    assert "tín chỉ" in normalized


def test_preprocess_audio_vad_detects_silence_and_voiced_audio():
    silent_wav = make_wav_bytes(amplitude=0)
    _, is_speech_silent = stt.preprocess_audio_vad(silent_wav)
    assert is_speech_silent is False

    voiced_wav = make_wav_bytes(amplitude=14000)
    processed_bytes, is_speech_voiced = stt.preprocess_audio_vad(voiced_wav)
    assert is_speech_voiced is True
    assert len(processed_bytes) > 44


def test_validate_audio_bytes_rejects_empty_oversize_and_invalid_magic():
    with pytest.raises(ValueError):
        stt.validate_audio_bytes(b"", filename="empty.wav")

    oversized = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * (cfg.MAX_AUDIO_BYTES + 10)
    with pytest.raises(stt.AudioTooLargeError):
        stt.validate_audio_bytes(oversized, filename="huge.wav")

    with pytest.raises(stt.UnsupportedAudioFormatError):
        stt.validate_audio_bytes(b"%PDF-1.7 fake document content", filename="fake.wav")


def test_split_stream_segments_keeps_first_segment_short_for_low_ttfb():
    long_answer = (
        "Dựa trên Quy chế đào tạo hiện hành của Đại học Hoa Sen, sinh viên cần tích lũy đủ số tín chỉ bắt buộc "
        "và tự chọn theo khung chương trình đào tạo của từng ngành học cụ thể. "
        "Ngoài ra, điểm trung bình chung tích lũy toàn khóa phải đạt từ 2.0 trở lên và hoàn thành chuẩn đầu ra ngoại ngữ. "
        "Sinh viên cũng cần hoàn thành kỳ thực tập tốt nghiệp và khóa luận tốt nghiệp đúng tiến độ."
    )
    segments = tts._split_stream_segments(long_answer, first_max_chars=120, next_max_chars=280)
    assert len(segments) >= 2
    assert len(segments[0]) <= 120
    assert all(tts.is_speakable_text(seg) for seg in segments)


def test_speech_lexicon_files_load_and_prompt_fits_whisper_limit():
    phonetic_rules = texts.read_rules("stt_phonetic_rules.jsonl", "replace")
    tts_rules = texts.read_rules("tts_rules.jsonl", "replace")
    assert len(phonetic_rules) > 17
    assert len(tts_rules) > 24

    prompt = stt.DOMAIN_PROMPT
    for term in ("HSU", "Hoa Sen Elite", "Field trip", "VEPT", "TOEFL", "TOCFL", "EII", "EIC", "CĐR", "BM_04B"):
        assert term in prompt

    # Whisper sử dụng bảng BPE ~50k token (tương đồng gpt2 trên tiếng Việt có dấu) với giới hạn 224 token cho prompt
    try:
        import tiktoken
    except ImportError:
        tiktoken = None
    if tiktoken is not None:
        assert len(tiktoken.get_encoding("gpt2").encode(prompt)) <= 224


def test_apply_phonetic_fix_covers_78_doc_terms_and_avoids_false_positives():
    raw_certs = (
        "Sinh viên nộp tốp phồ, tốc phồ hoặc vi i pi ti hay áp tít, tô pích, gi eo pi ti "
        "sau khi học i ai ai và i ai xi để đạt xê đê rờ."
    )
    fixed_certs = stt.apply_phonetic_fix(raw_certs)
    for expected in ("TOEFL", "TOCFL", "VEPT", "APTIS", "TOPIK", "JLPT", "EII", "EIC", "CĐR"):
        assert expected in fixed_certs

    raw_academic = (
        "Hỏi cố vẫn học tập của chương trình hoa sen e lít về phiêu chíp, "
        "liên chính học thuật trên tơ nít in, thực tặp nhận thức, xê tê đê tê, đê ca hát pê, "
        "gờ đê tê xê và biểu mẩu bê mờ không bốn bê hoặc bm 01."
    )
    fixed_academic = stt.apply_phonetic_fix(raw_academic)
    for expected in (
        "cố vấn học tập",
        "Hoa Sen Elite",
        "Field trip",
        "liêm chính học thuật",
        "Turnitin",
        "thực tập nhận thức",
        "CTĐT",
        "ĐKHP",
        "GDTC",
        "biểu mẫu",
        "BM_04B",
        "BM_01",
    ):
        assert expected in fixed_academic

    # Không thay nhầm từ tiếng Việt thông thường
    clean_normal = "Ăn tô phở xong rồi điểm ai cao nhất để lấy áo vét cạn và mở hộc bàn"
    assert stt.apply_phonetic_fix(clean_normal) == clean_normal


def test_normalize_academic_text_handles_78_doc_codes_and_compound_ordering():
    doc_codes = (
        "Căn cứ 1388/QĐ-ĐHHS, 10/TB-ĐTĐH, 157/NQ-ĐHHS, 1863/QĐ-BGH, "
        "02/2022/TT-BGDĐT, 17/VBHN-BGDĐT, 99/2019/NĐ-CP, 274/2006/QĐ-TTg, "
        "QC-ĐTĐH, QC-CVHT và QC-NH."
    )
    norm_docs = tts.normalize_academic_text(doc_codes)
    for expected in (
        "1388 Quyết định Đại học Hoa Sen",
        "10 Thông báo Phòng Đào tạo đại học",
        "157 Nghị quyết Đại học Hoa Sen",
        "1863 Quyết định Ban Giám hiệu",
        "02 năm 2022 Thông tư Bộ Giáo dục và Đào tạo",
        "17 Văn bản hợp nhất Bộ Giáo dục và Đào tạo",
        "99 năm 2019 Nghị định Chính phủ",
        "274 năm 2006 Quyết định Thủ tướng Chính phủ",
        "Quy chế Đào tạo trình độ đại học",
        "Quy chế Cố vấn học tập",
        "Quy chế Người học",
    ):
        assert expected in norm_docs

    compounds_and_units = (
        "SV lập KHHT-CN theo KHHT-M của CTĐT, trình CVHT duyệt trước khi ĐKHP hoặc ĐKMH; "
        "hoàn thành TTNT/TTNN dưới HTTL bằng BM_01 và BM_04B/TTTL gửi TT.TNSV&QHDN, "
        "P.ĐTĐH, P.KT-ĐBCL, P.TC-KT và P.CNTT; học đủ GDQP-AN, GDTC, NN2 để đạt CĐR và XTN."
    )
    norm_units = tts.normalize_academic_text(compounds_and_units)
    for expected in (
        "kế hoạch học tập cá nhân",
        "kế hoạch học tập mẫu",
        "chương trình đào tạo",
        "cố vấn học tập",
        "đăng ký học phần",
        "đăng ký môn học",
        "thực tập nhận thức và thực tập nghề nghiệp",
        "hình thức tích lũy",
        "biểu mẫu 1",
        "biểu mẫu 4B thực tập tích lũy",
        "Trung tâm Trải nghiệm sinh viên và Quan hệ doanh nghiệp",
        "Phòng Đào tạo đại học",
        "Phòng Khảo thí và Đảm bảo chất lượng",
        "Phòng Tài chính Kế toán",
        "Phòng Công nghệ thông tin",
        "Giáo dục quốc phòng và an ninh",
        "Giáo dục thể chất",
        "ngoại ngữ hai",
        "chuẩn đầu ra",
        "xét tốt nghiệp",
    ):
        assert expected in norm_units
    assert "tín chỉ-KT" not in norm_units
    assert "P.Công nghệ" not in norm_units


def test_normalize_academic_text_grades_english_certs_and_safe_boundaries():
    raw = (
        "SV đạt TBTL >= 2.5 hoặc TBMH < 4.0 hay TBHP 3,2/4,0 cần GĐCT xem KQHT và DSSV "
        "từ ĐVQLCM lên HĐKHĐT qua HTTTQL. Kết quả xếp loại XS Tốt Khá TB Yếu; "
        "chứng chỉ IELTS, TOEIC, TOEFL, TOCFL, VEPT và học phần EII, EIC giữ nguyên."
    )
    normalized = tts.normalize_academic_text(raw)
    for expected in (
        "trung bình tích lũy lớn hơn hoặc bằng 2.5",
        "trung bình môn học nhỏ hơn 4.0",
        "trung bình học phần 3 chấm 2 trên 4 chấm 0",
        "giám đốc chương trình",
        "kết quả học tập",
        "danh sách sinh viên",
        "đơn vị quản lý chuyên môn",
        "Hội đồng Khoa học và Đào tạo",
        "hệ thống thông tin quản lý",
        "XS Tốt Khá TB Yếu",
        "IELTS, TOEIC, TOEFL, TOCFL, VEPT",
        "EII, EIC",
    ):
        assert expected in normalized
