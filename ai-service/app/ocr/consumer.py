"""RabbitMQ consumer OCR: nhan job {documentId} tu hang doi `file-service.ocr.jobs` (file-service khai bao hang doi va gui job khi upload / chay lai OCR),
tai file qua link tai cua file-service (GET /api/documents/download/{id}), OCR bang extract.extract, roi bao ket qua ve file-service
(PUT /internal/documents/{id}/ocr: PROCESSING -> COMPLETED / FAILED). file-service luu van ban va tu gui job embed sang embed_consumer.

Chay trong thread nen. Moi job xu ly o thread rieng de thread cua pika van tra loi heartbeat trong luc OCR keo dai nhieu phut.
Loi tam thoi (file-service / PaddleOCR-VL chua len, mang) -> doi roi tra job ve hang doi; qua so lan giao cua hang doi (x-delivery-limit)
thi RabbitMQ chuyen vao dead-letter queue va file-service danh dau FAILED. Loi vinh vien (file hong, sai dinh dang) -> bao FAILED ngay.
shortcut: RabbitMQ huy consumer neu mot job giu ack qua `consumer_timeout` (mac dinh 30 phut); file scan qua dai thi tang gia tri nay o RabbitMQ."""
import json
import logging
import re
import threading
import time
from functools import partial

import httpx

from .. import config as cfg
from ..ingest.embed_consumer import FILE_SERVICE_URL, RABBITMQ_URL
from . import extract, paddle_vl

log = logging.getLogger("uvicorn.error")

OCR_QUEUE = "file-service.ocr.jobs"
RETRY_DELAY_S = 5  # cho truoc khi noi lai RabbitMQ
JOB_BACKOFF_S = 60  # cho truoc khi tra job loi tam thoi ve hang doi (tranh dot het so lan giao luc dich vu con dang khoi dong)
DOWNLOAD_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


def _report(document_id, status, text=None, error=None):
    resp = httpx.put(f"{FILE_SERVICE_URL}/internal/documents/{document_id}/ocr", headers={"X-Internal-Token": cfg.INTERNAL_TOKEN},
                     json={"status": status, "text": text, "error": error}, timeout=60)
    return resp.status_code


def _file_name(resp):
    match = re.search(r'filename="?([^";]+)', resp.headers.get("content-disposition", ""))
    return match.group(1) if match else ""


def process(document_id):
    """Xu ly mot tai lieu. Tra ve binh thuong = xong hoac bo qua (ack); nem loi = tam thoi, thu lai."""
    started = _report(document_id, "PROCESSING")
    if started == 404:
        log.info("OCR: tai lieu %s da bi xoa, bo qua job", document_id)
        return
    if started == 409:
        log.info("OCR: tai lieu %s da xu ly xong, bo qua job trung", document_id)
        return
    if started >= 300:
        raise RuntimeError(f"file-service tra {started} khi bat dau OCR")

    file = httpx.get(f"{FILE_SERVICE_URL}/api/documents/download/{document_id}", timeout=DOWNLOAD_TIMEOUT)
    file.raise_for_status()
    begin = time.perf_counter()
    try:
        result = extract.extract(file.content, _file_name(file))
    except (extract.UnsupportedFormat, extract.UnreadableFile, paddle_vl.BadDocument) as e:
        log.warning("OCR: tai lieu %s loi vinh vien: %s", document_id, e)
        _report(document_id, "FAILED", error=str(e))
        return
    reported = _report(document_id, "COMPLETED", text=result.text)
    if reported >= 300 and reported != 404:
        raise RuntimeError(f"file-service tra {reported} khi luu ket qua OCR")
    log.info("OCR: tai lieu %s xong sau %.0fs (%s trang, %d trang scan)", document_id, time.perf_counter() - begin, result.pages, result.ocr_pages)


def _work(connection, channel, delivery_tag, body):
    ack = partial(channel.basic_ack, delivery_tag)
    try:
        document_id = json.loads(body)["documentId"]
    except (ValueError, KeyError, TypeError):  # message hong: giao lai cung vo ich
        log.error("OCR: bo message khong hop le")
        verdict = ack
    else:
        try:
            process(document_id)
            verdict = ack
        except Exception as e:  # chi ghi loai loi: thong bao loi cua httpx co the chua URL noi bo
            log.error("OCR: tai lieu %s loi tam thoi (%s), tra job ve hang doi sau %ds", document_id, type(e).__name__, JOB_BACKOFF_S)
            time.sleep(JOB_BACKOFF_S)
            verdict = partial(channel.basic_nack, delivery_tag, requeue=True)
    try:
        connection.add_callback_threadsafe(verdict)  # ack / nack phai chay o thread cua connection
    except Exception as e:  # mat ket noi: RabbitMQ tu giao lai job, file-service bo qua neu da xong
        log.error("OCR: khong gui duoc ack (%s)", type(e).__name__)


def _dispatch(connection):
    """Callback cua pika: moi message xu ly o thread rieng de thread cua connection van tra loi heartbeat luc OCR keo dai."""
    def on_message(channel, method, properties, body):
        threading.Thread(target=_work, args=(connection, channel, method.delivery_tag, body), name="ocr-job", daemon=True).start()
    return on_message


def _run_consumer():
    try:
        import pika
    except ImportError:
        log.warning("pika chua duoc cai, OCR consumer khong khoi dong.")
        return
    while True:
        try:
            params = pika.URLParameters(RABBITMQ_URL)
            params.heartbeat = 60
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=OCR_QUEUE, passive=True)  # file-service khai bao (quorum + dead-letter); chua co thi thu lai
            channel.basic_qos(prefetch_count=1)  # mot file OCR mot luc
            channel.basic_consume(queue=OCR_QUEUE, on_message_callback=_dispatch(connection))
            log.info("OCR consumer san sang, dang nghe queue '%s'.", OCR_QUEUE)
            channel.start_consuming()
        except Exception as e:
            log.error("OCR consumer loi (%s). Thu lai sau %ds...", type(e).__name__, RETRY_DELAY_S)
            time.sleep(RETRY_DELAY_S)


def start_consumer():
    """Khoi dong consumer trong daemon thread."""
    threading.Thread(target=_run_consumer, name="ocr-consumer", daemon=True).start()
    log.info("OCR consumer thread da khoi dong.")
