import scipy.linalg  # noqa: F401  (must load BLAS before torch to avoid a Windows DLL/heap crash)
import sklearn.metrics.pairwise  # noqa: F401
import pytest

from app import config as cfg


def pytest_collection_modifyitems(config, items):
    """Test `live` goi Gemini that va an vao han muc moi ngay cua app: bo qua tru khi chon ro `pytest -m live`."""
    if "live" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="goi Gemini that (ton han muc); chay bang `pytest -m live`")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def feature_flags_off(monkeypatch):
    """config.py goi load_dotenv(override=True) nen `.env` cua may dev co the bat cac co nay; test nao can thi tu bat lai."""
    for flag in ("ROUTER_ENABLED", "SESSION_ENABLED", "QUERY_REWRITE_ENABLED"):
        monkeypatch.setattr(cfg, flag, False)
