"""Chay voi Qdrant that o QDRANT_URL. Nguon va dich la collection tam; collection that khong bi dong vao."""
import hashlib
import json
import re
import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from app import bm25
from app import config as cfg
from app import qdrant_setup
from scripts import migrate_add_bm25

pytestmark = pytest.mark.integration

TEXTS = {
    1: "[Quy chế học bổng – Điều 15. Học bổng khuyến khích]\n1. Sinh viên có điểm trung bình từ 3,2 được xét học bổng.",
    2: "[Quy chế học bổng – Điều 16. Học bổng tài năng]\n1. Sinh viên đạt giải olympic quốc tế.",
    3: "[Thông báo số 84]\nNộp chứng chỉ tiếng Anh trước tuần 10.",
    4: "?!... —",  # không có token: điểm vẫn được chép, chỉ không có vector bm25
}
UUID_ID = "7f9c0a54-3b52-4c1f-9d3e-2a6c1b0e8d11"


def unit(i):
    v = np.zeros(cfg.EMBED_DIM, dtype="float32")
    v[i] = 1.0
    return v.tolist()


@pytest.fixture(scope="module")
def client():
    c = QdrantClient(url=cfg.QDRANT_URL, timeout=60)  # mac dinh 5s: de timeout khi may dang tai nang
    try:
        c.get_collections()
    except Exception as e:
        pytest.skip(f"Khong ket noi duoc Qdrant {cfg.QDRANT_URL}: {e}")
    return c


@pytest.fixture
def source(client):
    """Collection nguồn dạng cũ (dense + sparse của BGE-M3, chưa có bm25), có id số và id UUID."""
    name = f"test_mig_src_{uuid.uuid4().hex[:8]}"
    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=cfg.EMBED_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    try:  # loi o buoc nap du lieu van phai xoa collection tam
        client.create_payload_index(name, "kho", models.PayloadSchemaType.KEYWORD)
        ids = [1, 2, 3, UUID_ID]
        client.upsert(name, wait=True, points=[
            models.PointStruct(
                id=pid,
                vector={"dense": unit(i), "sparse": models.SparseVector(indices=[10 + i], values=[0.5 + i / 10])},
                payload={"source": "s", "text": TEXTS[i], "document_id": i, "kho": "A" if i % 2 else "B"},
            )
            for i, pid in enumerate(ids, start=1)
        ])
        yield name
    finally:
        assert name.startswith("test_")
        client.delete_collection(name)


@pytest.fixture
def target_name(client):
    name = f"test_mig_dst_{uuid.uuid4().hex[:8]}"
    yield name
    assert name.startswith("test_")
    if client.collection_exists(name):
        client.delete_collection(name)


def fingerprint(client, name):
    """Toàn bộ nội dung collection (id, payload, vector) + số điểm + cấu hình vector."""
    info = client.get_collection(name)
    points, _ = client.scroll(name, limit=1000, with_payload=True, with_vectors=True)
    body = sorted(([str(p.id), p.payload, json.loads(json.dumps(p.vector, default=lambda o: o.model_dump()))] for p in points),
                  key=lambda row: row[0])
    config = info.config.params.model_dump(mode="json")
    return info.points_count, hashlib.sha256(json.dumps([body, config], sort_keys=True).encode()).hexdigest()


def run(args, capsys):
    code = migrate_add_bm25.main(args)
    return code, capsys.readouterr()


def test_migration_copies_every_point_adds_bm25_and_prints_the_env_changes(client, source, target_name, capsys):
    before = fingerprint(client, source)
    code, out = run(["--source", source, "--target", target_name], capsys)

    assert code == 0, out.err
    assert fingerprint(client, source) == before  # nguồn không bị đụng tới
    assert client.count(target_name, exact=True).count == client.count(source, exact=True).count == 4

    avg = float(re.search(r"BM25_AVG_LEN = ([0-9.]+)", out.out).group(1))
    expected_avg = round(sum(len(bm25.bm25_tokens(t)) for t in TEXTS.values()) / len(TEXTS), 2)
    assert avg == expected_avg
    assert f'QDRANT_COLLECTION = "{target_name}"' in out.out  # khác mặc định của config.py nên phải in
    assert "HYBRID_LEGS" not in out.out and "BM25_TOKENIZER" not in out.out  # đã là mặc định: không cần dặn đặt

    src_points = {str(p.id): p for p in client.scroll(source, limit=100, with_payload=True, with_vectors=True)[0]}
    dst_points = {str(p.id): p for p in client.scroll(target_name, limit=100, with_payload=True, with_vectors=True)[0]}
    assert set(dst_points) == set(src_points) and UUID_ID in dst_points  # cùng id
    for pid, src in src_points.items():
        dst = dst_points[pid]
        assert dst.payload == src.payload
        assert dst.vector["dense"] == src.vector["dense"]  # dense chép nguyên, không embed lại
        assert "sparse" not in dst.vector  # sparse của BGE-M3 đã được BM25 thay thế
        expected = bm25.bm25_doc_vector(src.payload["text"], avg_len=avg)
        if expected.indices:
            assert dst.vector["bm25"].indices == expected.indices
            assert dst.vector["bm25"].values == pytest.approx(expected.values, rel=1e-5)
        else:
            assert "bm25" not in dst.vector
    assert qdrant_setup.vector_names(client, target_name) == {"dense", "bm25"}


def test_running_twice_gives_identical_counts_and_content(client, source, target_name, capsys):
    before = fingerprint(client, source)
    assert run(["--source", source, "--target", target_name], capsys)[0] == 0
    first = fingerprint(client, target_name)
    assert run(["--source", source, "--target", target_name, "--batch-size", "2"], capsys)[0] == 0
    assert fingerprint(client, target_name) == first
    assert fingerprint(client, source) == before


def test_rerun_after_a_partial_copy_completes_it(client, source, target_name, capsys):
    assert run(["--source", source, "--target", target_name], capsys)[0] == 0
    full = fingerprint(client, target_name)
    client.delete(target_name, points_selector=models.PointIdsList(points=[1, UUID_ID]), wait=True)  # như thể chạy dở
    assert client.count(target_name, exact=True).count == 2
    assert run(["--source", source, "--target", target_name], capsys)[0] == 0
    assert fingerprint(client, target_name) == full


def test_mismatched_counts_are_reported(client, source, target_name, capsys):
    assert run(["--source", source, "--target", target_name], capsys)[0] == 0
    client.upsert(target_name, wait=True, points=[models.PointStruct(
        id=999, vector={"dense": unit(5)}, payload={"source": "s", "text": "diem thua"})])
    code, out = run(["--source", source, "--target", target_name], capsys)
    assert code == 1 and "KHÔNG KHỚP" in out.err and "BM25_AVG_LEN" not in out.out


def test_target_must_differ_from_source(client, source, capsys):
    before = fingerprint(client, source)
    code, out = run(["--source", source, "--target", source], capsys)
    assert code == 2 and "khác --source" in out.err
    assert fingerprint(client, source) == before


def test_existing_target_without_bm25_is_refused(client, source, target_name, capsys):
    client.create_collection(
        target_name,
        vectors_config={"dense": models.VectorParams(size=cfg.EMBED_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )
    code, out = run(["--source", source, "--target", target_name], capsys)
    assert code == 2 and "bm25" in out.err
    assert client.count(target_name, exact=True).count == 0


def test_source_without_a_dense_vector_is_refused(client, target_name, capsys):
    name = f"test_mig_src_{uuid.uuid4().hex[:8]}"
    client.create_collection(name, vectors_config={}, sparse_vectors_config={"sparse": models.SparseVectorParams()})
    try:
        code, out = run(["--source", name, "--target", target_name], capsys)
        assert code == 2 and "dense" in out.err
        assert not client.collection_exists(target_name)
    finally:
        client.delete_collection(name)


def test_missing_or_empty_source_is_refused(client, target_name, capsys):
    assert run(["--source", "test_khong_ton_tai", "--target", target_name], capsys)[0] == 2
    empty = f"test_mig_src_{uuid.uuid4().hex[:8]}"
    client.create_collection(empty, vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)})
    try:
        assert run(["--source", empty, "--target", target_name], capsys)[0] == 2
        assert not client.collection_exists(target_name)  # chưa tạo đích khi nguồn rỗng
    finally:
        client.delete_collection(empty)


def test_tokenizer_option_changes_the_measured_average(client, source, target_name, capsys):
    _, out = run(["--source", source, "--target", target_name, "--tokenizer", "syllable"], capsys)
    avg = float(re.search(r"BM25_AVG_LEN = ([0-9.]+)", out.out).group(1))
    assert avg == round(sum(len(bm25.bm25_tokens(t, "syllable")) for t in TEXTS.values()) / len(TEXTS), 2)
    assert 'BM25_TOKENIZER = "syllable"' in out.out
