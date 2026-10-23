from qdrant_client import models

from app.rag import query_model as qm


def test_filter_by_khoa_accepts_open_ended_documents():
    flt = qm.build_filter(khoa=2024)
    assert len(flt.must) == 4
    tu, den = flt.must[2], flt.must[3]
    assert tu.should[1].key == "khoa_tu" and tu.should[1].range.lte == 2024
    assert den.should[1].key == "khoa_den" and den.should[1].range.gte == 2024
    assert all(isinstance(c.should[0], models.IsEmptyCondition) for c in (tu, den))  # thiếu field = áp dụng mọi khóa


def test_filter_without_khoa_unchanged():
    assert len(qm.build_filter().must) == 2
