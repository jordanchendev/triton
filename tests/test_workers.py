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


def test_extract_text_returns_text():
    mock_result = [[([0, 0], ("Hello", 0.99)), ([0, 0], ("World", 0.98))]]

    with patch("triton.services.ocr._get_ocr") as mock_ocr:
        mock_ocr.return_value.ocr.return_value = mock_result
        from triton.services.ocr import extract_text

        result = extract_text("/tmp/test.png")
        assert "Hello" in result["text"]
        assert "World" in result["text"]


def test_download_audio_ignores_playlist_in_url(tmp_path):
    with patch.object(settings, "upload_dir", str(tmp_path)), patch("yt_dlp.YoutubeDL") as mock_ydl:
        ydl = mock_ydl.return_value.__enter__.return_value
        ydl.extract_info.return_value = {"id": "abc"}
        path = download_audio("https://www.youtube.com/watch?v=abc&list=WL", "youtube")

    assert mock_ydl.call_args.args[0]["noplaylist"] is True
    assert path == str(tmp_path / "abc.wav")
