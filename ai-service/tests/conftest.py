import scipy.linalg  # noqa: F401  (must load BLAS before torch to avoid a Windows DLL/heap crash)
import sklearn.metrics.pairwise  # noqa: F401
import socket

import pytest

_LOCAL_HOSTS = {None, "", "localhost", "127.0.0.1", "::1"}


def pytest_collection_modifyitems(config, items):
    """Test `live` goi Gemini that va an vao han muc moi ngay cua app: bo qua tru khi chon ro `pytest -m live`."""
    if "live" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="goi Gemini that (ton han muc); chay bang `pytest -m live`")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def no_external_network(request, monkeypatch):
    """Router (Gemini), viet lai cau va tra loi (Groq) luon bat, nen test nao khong tu gia lap chung se goi mang that va tru han muc
    moi ngay. Chan moi lan phan giai ten mien ngoai may (chi cho localhost: Qdrant, TestClient); router / viet lai loi thi cau hoi di
    thang vao RAG nhu khi chung hong. Test `integration` va `live` duoc di mang vi chung can model that / Qdrant / Gemini."""
    if request.node.get_closest_marker("live") or request.node.get_closest_marker("integration"):
        return
    real_getaddrinfo = socket.getaddrinfo

    def guarded(host, *args, **kwargs):
        if host not in _LOCAL_HOSTS:
            raise socket.gaierror(f"test khong duoc goi mang that: {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded)
