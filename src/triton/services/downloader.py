import os

from triton.config import settings


def download_audio(url: str, task_type: str) -> str:
    """Download audio from URL using yt-dlp. Returns path to downloaded file."""
    import yt_dlp

    os.makedirs(settings.upload_dir, exist_ok=True)
    output_template = os.path.join(settings.upload_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]
        return os.path.join(settings.upload_dir, f"{video_id}.wav")
