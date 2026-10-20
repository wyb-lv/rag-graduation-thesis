"""RabbitMQ consumer chạy trong background thread, nhận job embed từ file-service.

Khi file-service hoàn thành OCR, nó publish một message vào exchange `file-service.embed`
với routing key `embed.jobs`. Consumer này lắng nghe queue `ai-service.embed.jobs`,
tải văn bản từ file-service qua HTTP, rồi gọi embed_service.embed_document().

Biến môi trường:
    RABBITMQ_URL     amqp://user:pass@host:5672/   (bắt buộc nếu dùng RabbitMQ)
    FILE_SERVICE_URL http://file-service:8080        (để tải văn bản)

Nếu RABBITMQ_URL không được cấu hình thì consumer không khởi động (graceful degradation).
"""
import json
import logging
import os
import threading
import time
from typing import Optional

import httpx

from . import embed_service

log = logging.getLogger("uvicorn.error")

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "")
FILE_SERVICE_URL = os.environ.get("FILE_SERVICE_URL", "http://localhost:8080")

EMBED_EXCHANGE = "file-service.embed"
EMBED_QUEUE = "ai-service.embed.jobs"
EMBED_ROUTING_KEY = "embed.jobs"
EMBED_DEAD_QUEUE = "ai-service.embed.jobs.dead"

MAX_RETRIES = 3
RETRY_DELAY = 5  # giây


def _fetch_document_text(document_id: int) -> Optional[str]:
    """Gọi file-service GET /api/documents/{id}/text để lấy văn bản đã OCR."""
    url = f"{FILE_SERVICE_URL}/api/documents/{document_id}/text"
    try:
        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("text")
    except Exception as e:
        log.error("Lỗi khi lấy văn bản tài liệu %d từ file-service: %s", document_id, e)
        return None


def _process_job(body: dict):
    """Xử lý một job embed (UPSERT hoặc DELETE)."""
    document_id = body.get("documentId")
    action = body.get("action", "UPSERT")

    if action == "DELETE":
        log.info("Nhận yêu cầu xóa vector Qdrant cho tài liệu %s", document_id)
        embed_service.delete_document_vectors(document_id)
        return

    source = body.get("source", f"Tài liệu #{document_id}")
    kho = body.get("kho")
    hieu_luc_tu = body.get("hieuLucTu")
    hieu_luc_den = body.get("hieuLucDen")
    text = body.get("text")

    # Nếu message không kèm text thì lấy từ file-service
    if not text:
        text = _fetch_document_text(document_id)

    if not text:
        log.warning("Không có văn bản cho tài liệu %d, bỏ qua job embed.", document_id)
        return

    embed_service.embed_document(
        document_id=document_id,
        text=text,
        source=source,
        kho=kho,
        hieu_luc_tu=hieu_luc_tu,
        hieu_luc_den=hieu_luc_den,
    )


def _run_consumer():
    """Vòng lặp consumer với reconnect tự động."""
    try:
        import pika
    except ImportError:
        log.warning("pika chưa được cài, RabbitMQ consumer không khởi động. Cài bằng: pip install pika")
        return

    while True:
        try:
            log.info("RabbitMQ embed consumer: đang kết nối đến %s...", RABBITMQ_URL)
            params = pika.URLParameters(RABBITMQ_URL)
            params.heartbeat = 60
            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            # Khai báo exchange và queue (idempotent)
            channel.exchange_declare(exchange=EMBED_EXCHANGE, exchange_type="direct", durable=True)
            channel.exchange_declare(exchange=f"{EMBED_EXCHANGE}.dead", exchange_type="direct", durable=True)
            channel.queue_declare(
                queue=EMBED_DEAD_QUEUE,
                durable=True,
            )
            channel.queue_bind(exchange=f"{EMBED_EXCHANGE}.dead", queue=EMBED_DEAD_QUEUE, routing_key=EMBED_ROUTING_KEY)
            channel.queue_declare(
                queue=EMBED_QUEUE,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": f"{EMBED_EXCHANGE}.dead",
                    "x-dead-letter-routing-key": EMBED_ROUTING_KEY,
                    "x-delivery-limit": MAX_RETRIES,
                    "x-queue-type": "quorum",
                },
            )
            channel.queue_bind(exchange=EMBED_EXCHANGE, queue=EMBED_QUEUE, routing_key=EMBED_ROUTING_KEY)
            channel.basic_qos(prefetch_count=1)

            log.info("RabbitMQ embed consumer sẵn sàng, đang lắng nghe queue '%s'.", EMBED_QUEUE)

            def on_message(ch, method, properties, body_bytes):
                try:
                    body = json.loads(body_bytes)
                    log.info("Nhận embed job: %s", body)
                    _process_job(body)
                    ch.basic_ack(delivery_tag=method.delivery_tag)
                except Exception as e:
                    log.error("Lỗi xử lý embed job: %s", e, exc_info=True)
                    # nack + requeue=False → vào dead-letter queue sau MAX_RETRIES
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            channel.basic_consume(queue=EMBED_QUEUE, on_message_callback=on_message)
            channel.start_consuming()

        except Exception as e:
            log.error("RabbitMQ embed consumer lỗi: %s. Thử lại sau %ds...", e, RETRY_DELAY)
            time.sleep(RETRY_DELAY)


def start_consumer():
    """Khởi động consumer trong daemon thread nếu RABBITMQ_URL được cấu hình."""
    if not RABBITMQ_URL:
        log.info("RABBITMQ_URL chưa cấu hình, bỏ qua RabbitMQ embed consumer.")
        return
    t = threading.Thread(target=_run_consumer, name="embed-consumer", daemon=True)
    t.start()
    log.info("RabbitMQ embed consumer thread đã khởi động.")
