"""OCR consumer bao ket qua ve file-service: /internal/** chi nhan request mang INTERNAL_TOKEN dung."""
import httpx

from app import config as cfg
from app.ocr import consumer


def test_report_sends_the_internal_token_header(monkeypatch):
    monkeypatch.setattr(cfg, "INTERNAL_TOKEN", "s3cret")
    seen = {}

    def fake_put(url, **kwargs):
        seen["url"], seen["headers"], seen["json"] = url, kwargs["headers"], kwargs["json"]
        return httpx.Response(204)

    monkeypatch.setattr(consumer.httpx, "put", fake_put)
    assert consumer._report(42, "PROCESSING") == 204
    assert seen["url"].endswith("/internal/documents/42/ocr")
    assert seen["headers"] == {"X-Internal-Token": "s3cret"}
    assert seen["json"]["status"] == "PROCESSING"
