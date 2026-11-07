import scipy.linalg  # noqa: F401  (must load BLAS before torch to avoid a Windows DLL/heap crash)
import sklearn.metrics.pairwise  # noqa: F401
from types import SimpleNamespace

import pytest

from app import api
from app import config as cfg
from app import conversations, session
from app.ingest import embed_consumer
from app.ocr import consumer as ocr_consumer
from app.rag import query_model as qm
from app.router import router_llm


def pytest_collection_modifyitems(config, items):
    """Test `live` goi Gemini that va an vao han muc moi ngay cua app: bo qua tru khi chon ro `pytest -m live`."""
    if "live" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="goi Gemini that (ton han muc); chay bang `pytest -m live`")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def offline_services(request, monkeypatch, tmp_path):
    """Router, viet lai cau hoi, session, luu tru hoi thoai, cache Groq va RabbitMQ consumer luon bat trong app (khong co cong tac), nen moi
    test mac dinh chay voi ban gia khong cham mang: router loi (-> cau hoi di thang vao RAG), viet lai khong doi cau hoi, Mongo bo qua,
    consumer khong khoi dong, session moi trong RAM. Test nao can hanh vi that thi tu thay lai (xem test_router_session_api.py)."""
    async def router_offline(question, history, session_facts, previous_was_clarify=False):
        return router_llm.Routed(None, "error", "test")  # router loi: cau hoi di thang vao RAG

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(api, "router_llm", SimpleNamespace(resolve=router_offline, apply_policy=router_llm.apply_policy))  # chi trong api; module that van nguyen
    monkeypatch.setattr(qm, "query_rewrite", SimpleNamespace(rewrite=lambda question: None))  # chi trong query_model; module that van nguyen
    monkeypatch.setattr(session, "_store", None)
    monkeypatch.setattr(cfg, "GROQ_CACHE_DIR", str(tmp_path / "groq_cache"))
    monkeypatch.setattr(embed_consumer, "start_consumer", lambda: None)
    monkeypatch.setattr(ocr_consumer, "start_consumer", lambda: None)
    if not request.node.get_closest_marker("integration"):  # test MongoDB that dung chinh conversations.py
        monkeypatch.setattr(conversations, "add_turn", nothing)
        monkeypatch.setattr(conversations, "ensure_indexes", nothing)
