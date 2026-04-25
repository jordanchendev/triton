from triton.workers.celery_app import celery_app


@celery_app.task(name="triton.workers.gpu_tasks.transcribe")
def transcribe(task_id: str, file_path: str):
    """Transcribe audio/video file using faster-whisper on GPU."""
    from triton.services.task_runner import run_task
    from triton.services.transcriber import transcribe_file

    run_task(task_id, "transcribing", lambda: transcribe_file(file_path), device="gpu")


@celery_app.task(name="triton.workers.gpu_tasks.ocr")
def ocr(task_id: str, file_path: str):
    """Extract text from PDF/image using PaddleOCR on GPU."""
    from triton.services.ocr import extract_text
    from triton.services.task_runner import run_task

    run_task(task_id, "ocr", lambda: extract_text(file_path), device="gpu")
