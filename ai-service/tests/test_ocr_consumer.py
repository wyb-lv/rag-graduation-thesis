"""OCR consumer (app/ocr/consumer.py): xu ly mot job (tai file qua link file-service, OCR, bao ket qua) va ack / nack. Khong co RabbitMQ / file-service that."""
import httpx
import pytest

from app.ocr import consumer, extract, paddle_vl


class FileService:
    """Thay httpx.put / httpx.get cua consumer: ghi lai cac lan bao trang thai, tra file gia."""

    def __init__(self, monkeypatch, start=204, finish=204, file_status=200, name='attachment; filename="Quy che.PDF"'):
        self.reports, self.downloads = [], []
        self.start, self.finish, self.file_status, self.name = start, finish, file_status, name
        monkeypatch.setattr(consumer.httpx, "put", self.put)
        monkeypatch.setattr(consumer.httpx, "get", self.get)

    def put(self, url, json, timeout):
        self.reports.append((url, json))
        return httpx.Response(self.start if json["status"] == "PROCESSING" else self.finish)

    def get(self, url, timeout):
        self.downloads.append(url)
        return httpx.Response(self.file_status, content=b"%PDF", headers={"content-disposition": self.name},
                              request=httpx.Request("GET", url))


def statuses(fs):
    return [body["status"] for _, body in fs.reports]


def test_success_downloads_by_link_ocrs_with_the_original_file_name_and_reports_text(monkeypatch):
    fs = FileService(monkeypatch)
    seen = {}
    monkeypatch.setattr(consumer.extract, "extract", lambda data, name: seen.update(data=data, name=name) or extract.Result("Điều 1", 3, 2))
    consumer.process(42)
    assert fs.downloads == [f"{consumer.FILE_SERVICE_URL}/api/documents/download/42"]
    assert seen == {"data": b"%PDF", "name": "Quy che.PDF"}
    assert statuses(fs) == ["PROCESSING", "COMPLETED"]
    assert fs.reports[0][0].endswith("/internal/documents/42/ocr")
    assert fs.reports[1][1] == {"status": "COMPLETED", "text": "Điều 1", "error": None}


@pytest.mark.parametrize("start", [404, 409])
def test_deleted_or_already_finished_document_is_skipped_without_downloading(monkeypatch, start):
    fs = FileService(monkeypatch, start=start)
    monkeypatch.setattr(consumer.extract, "extract", lambda *a: pytest.fail("khong duoc OCR"))
    consumer.process(1)
    assert fs.downloads == [] and statuses(fs) == ["PROCESSING"]


@pytest.mark.parametrize("error", [extract.UnreadableFile("PDF co mat khau"), extract.UnsupportedFormat("x"), paddle_vl.BadDocument("anh hong")])
def test_permanent_failure_is_reported_as_failed_and_acked_not_retried(monkeypatch, error):
    fs = FileService(monkeypatch)

    def fail(data, name):
        raise error

    monkeypatch.setattr(consumer.extract, "extract", fail)
    consumer.process(1)  # khong nem: job duoc ack
    assert statuses(fs) == ["PROCESSING", "FAILED"] and fs.reports[1][1]["error"] == str(error)


def test_transient_failures_raise_so_the_job_goes_back_to_the_queue(monkeypatch):
    FileService(monkeypatch)

    def down(data, name):
        raise paddle_vl.OcrUnavailable("PaddleOCR-VL tra 503")

    monkeypatch.setattr(consumer.extract, "extract", down)
    with pytest.raises(paddle_vl.OcrUnavailable):
        consumer.process(1)
    FileService(monkeypatch, file_status=503)
    with pytest.raises(httpx.HTTPStatusError):
        consumer.process(1)
    FileService(monkeypatch, start=500)
    with pytest.raises(RuntimeError):
        consumer.process(1)


def test_file_name_without_header_is_empty_so_extract_rejects_it():
    assert consumer._file_name(httpx.Response(200)) == ""


class Channel:
    def __init__(self):
        self.calls = []

    def basic_ack(self, tag):
        self.calls.append(("ack", tag))

    def basic_nack(self, tag, requeue):
        self.calls.append(("nack", tag, requeue))


class Connection:
    def add_callback_threadsafe(self, fn):
        fn()


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(consumer.time, "sleep", lambda s: None)


def run(monkeypatch, body, process):
    monkeypatch.setattr(consumer, "process", process)
    channel = Channel()
    consumer._work(Connection(), channel, 9, body)
    return channel.calls


def test_work_acks_success_and_unreadable_messages_and_requeues_transient_errors(monkeypatch, no_sleep):
    done = []
    assert run(monkeypatch, b'{"documentId": 5}', done.append) == [("ack", 9)] and done == [5]
    assert run(monkeypatch, b"not json", lambda i: pytest.fail("khong duoc xu ly")) == [("ack", 9)]
    assert run(monkeypatch, b'{"id": 5}', lambda i: pytest.fail("khong duoc xu ly")) == [("ack", 9)]

    def down(i):
        raise httpx.ConnectError("http://internal:8080 refused")

    assert run(monkeypatch, b'{"documentId": 5}', down) == [("nack", 9, True)]


def test_a_value_error_inside_processing_is_a_retry_not_a_dropped_job(monkeypatch, no_sleep):
    def bug(i):
        raise ValueError("loi trong luc xu ly")

    assert run(monkeypatch, b'{"documentId": 5}', bug) == [("nack", 9, True)]


def test_work_survives_a_lost_connection_when_acking(monkeypatch, no_sleep):
    monkeypatch.setattr(consumer, "process", lambda i: None)

    class Gone:
        def add_callback_threadsafe(self, fn):
            raise RuntimeError("connection closed")

    consumer._work(Gone(), Channel(), 1, b'{"documentId": 1}')  # khong nem
