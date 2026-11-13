from unittest.mock import MagicMock, patch

from celery import Celery
from celery.app.task import Task as CeleryTask
from celery.backends.cache import CacheBackend

from triton.config import settings
from triton.models import Task
from triton.services.downloader import download_audio
from triton.services.ocr import extract_text, extract_text_from_page
from triton.workers.celery_app import celery_app
from triton.workers.cpu_ml_tasks import ocr_parallel


def test_transcribe_file_returns_text():
    mock_segment = MagicMock()
    mock_segment.text = "Hello world"
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95
    mock_info.duration = 10.5

    with patch("triton.services.transcriber._get_model") as mock_model:
        mock_model.return_value.transcribe.return_value = ([mock_segment], mock_info)
        from triton.services.transcriber import transcribe_file

        result = transcribe_file("/tmp/test.wav")
        assert result["text"] == "Hello world"
        assert result["metadata"]["language"] == "en"


OCR_PAGES = [
    {"input_path": "/tmp/test.pdf", "rec_texts": ["Hello", "World"], "rec_scores": [0.99, 0.98]},
    {"input_path": "/tmp/test.pdf", "rec_texts": ["Page two"], "rec_scores": [0.97]},
]


def test_extract_text_joins_recognised_lines_of_every_page():
    with patch("triton.services.ocr._get_ocr") as mock_ocr:
        mock_ocr.return_value.ocr.return_value = OCR_PAGES
        result = extract_text("/tmp/test.png")

    assert result["text"] == "Hello\nWorld\nPage two"
    assert result["metadata"]["pages"] == 2


def test_extract_text_from_page_returns_recognised_lines():
    with patch("triton.services.ocr._get_ocr") as mock_ocr:
        mock_ocr.return_value.ocr.return_value = OCR_PAGES[:1]
        assert extract_text_from_page("/tmp/page.png") == "Hello\nWorld"


def test_download_audio_ignores_playlist_in_url(tmp_path):
    with patch.object(settings, "upload_dir", str(tmp_path)), patch("yt_dlp.YoutubeDL") as mock_ydl:
        ydl = mock_ydl.return_value.__enter__.return_value
        ydl.extract_info.return_value = {"id": "abc"}
        path = download_audio("https://www.youtube.com/watch?v=abc&list=WL", "youtube")

    assert mock_ydl.call_args.args[0]["noplaylist"] is True
    assert path == str(tmp_path / "abc.wav")


def test_ocr_parallel_queues_one_ocr_page_per_scanned_page(db_session, tmp_path):
    task = Task(type="pdf", file_path="/tmp/scan.pdf")
    db_session.add(task)
    db_session.commit()
    task_id = task.id
    pages = [str(tmp_path / "page_0000.png"), str(tmp_path / "page_0001.png")]
    queued = []

    def record(self, args=None, kwargs=None, **options):
        queued.append((self.name, args))

    # Real chord dispatch down to Task.apply_async: eager mode would skip the group checks that rejected link_error.
    with (
        patch("triton.database.SessionLocal", return_value=db_session),
        patch("triton.services.ocr.try_pymupdf_text", return_value=None),
        patch("triton.services.ocr.split_pdf_to_images", return_value=pages),
        patch("tempfile.mkdtemp", return_value=str(tmp_path)),
        patch.object(Celery, "backend", CacheBackend(app=celery_app, url="memory://")),
        patch.object(CeleryTask, "apply_async", record),
    ):
        ocr_parallel(str(task_id), "/tmp/scan.pdf")

    saved = db_session.get(Task, task_id)
    assert (saved.status, saved.step, saved.error_message) == ("processing", "ocr", None)
    assert [args for name, args in queued if name == "triton.workers.cpu_ml_tasks.ocr_page"] == [
        (pages[0], 0),
        (pages[1], 1),
    ]
