from triton.config import settings

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _model


def transcribe_file(file_path: str) -> dict:
    """Transcribe audio/video file and return text + metadata."""
    model = _get_model()
    segments, info = model.transcribe(file_path, beam_size=5)

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text)

    return {
        "text": "".join(text_parts),
        "metadata": {
            "language": info.language,
            "language_probability": round(info.language_probability, 2),
            "duration": round(info.duration, 1),
        },
    }
