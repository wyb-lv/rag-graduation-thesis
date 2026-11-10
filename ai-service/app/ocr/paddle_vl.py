"""Client cua PaddleOCR-VL-1.6: pipeline `paddleocr[doc-parser]` (PP-DocLayoutV3 tach layout + VLM 0.9B doc tung phan tu: chu, bang, cong thuc)
chay trong container chinh thuc `paddleocr-vl-api` cua docker-compose.yml (VLM phuc vu boi `paddleocr-vlm-server`, vLLM). ai-service chi gui trang
scan sang `POST /layout-parsing` va nhan Markdown tung trang; khong cai PaddlePaddle trong image ai-service.

Service cua PaddleOCR-VL xu ly mot request mot luc, nen o day cung chi gui tuan tu (extract.py giu khoa)."""
import base64

import httpx

from .. import config as cfg
from .markdown import html_to_markdown


class OcrUnavailable(RuntimeError):
    """Service OCR chua len / qua tai / het gio / loi noi bo: thu lai sau duoc (consumer tra job ve hang doi)."""


class BadDocument(ValueError):
    """Service tu choi file (anh hong, PDF hong): thu lai cung vo ich."""


def parse(data, pdf):
    """Gui `data` (PDF hoac anh, TIFF nhieu trang tinh la nhieu trang) sang PaddleOCR-VL, tra ve [Markdown moi trang]."""
    body = {
        "file": base64.b64encode(data).decode("ascii"),
        "fileType": 0 if pdf else 1,
        "markdownIgnoreLabels": cfg.OCR_IGNORE_LABELS,
        "returnMarkdownImages": False,
        "visualize": False,
    }
    try:
        resp = httpx.post(f"{cfg.OCR_VL_URL}/layout-parsing", json=body, timeout=httpx.Timeout(cfg.OCR_VL_TIMEOUT_S, connect=10.0))
    except httpx.HTTPError as e:  # khong ghi noi dung loi: co the chua URL noi bo
        raise OcrUnavailable(f"khong goi duoc PaddleOCR-VL ({type(e).__name__})") from e
    if resp.status_code >= 500 or resp.status_code == 429:
        raise OcrUnavailable(f"PaddleOCR-VL tra {resp.status_code}")
    if resp.status_code >= 400:
        raise BadDocument(_message(resp))
    try:
        results = resp.json()["result"]["layoutParsingResults"]
        return [html_to_markdown(r["markdown"]["text"]).strip() for r in results]
    except (ValueError, KeyError, TypeError) as e:
        raise OcrUnavailable("PaddleOCR-VL tra ket qua sai dinh dang") from e


def _message(resp):
    try:
        return str(resp.json().get("errorMsg") or resp.status_code)
    except ValueError:
        return str(resp.status_code)
