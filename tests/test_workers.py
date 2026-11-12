from unittest.mock import MagicMock, patch

from triton.config import settings
from triton.services.downloader import download_audio


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
        from triton.services.ocr import extract_text

        result = extract_text("/tmp/test.png")

    assert result["text"] == "Hello\nWorld\nPage two"
    assert result["metadata"]["pages"] == 2


def test_extract_text_from_page_returns_recognised_lines():
    with patch("triton.services.ocr._get_ocr") as mock_ocr:
        mock_ocr.return_value.ocr.return_value = OCR_PAGES[:1]
        from triton.services.ocr import extract_text_from_page

        assert extract_text_from_page("/tmp/page.png") == "Hello\nWorld"


def test_download_audio_ignores_playlist_in_url(tmp_path):
    with patch.object(settings, "upload_dir", str(tmp_path)), patch("yt_dlp.YoutubeDL") as mock_ydl:
        ydl = mock_ydl.return_value.__enter__.return_value
        ydl.extract_info.return_value = {"id": "abc"}
        path = download_audio("https://www.youtube.com/watch?v=abc&list=WL", "youtube")

    assert mock_ydl.call_args.args[0]["noplaylist"] is True
    assert path == str(tmp_path / "abc.wav")
