import os

from celery import chord

from triton.workers.celery_app import celery_app


@celery_app.task(name="triton.workers.cpu_ml_tasks.ocr_parallel")
def ocr_parallel(task_id: str, file_path: str):
    """Split PDF into pages and dispatch parallel OCR tasks.
    Tries PyMuPDF text extraction first — skips OCR if PDF has embedded text.
    """
    from triton.services.ocr import try_pymupdf_text, split_pdf_to_images
    from triton.database import SessionLocal
    from triton.models import Task
    from datetime import datetime, timezone
    import uuid
    import tempfile

    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == uuid.UUID(task_id)).first()
        if not task:
            return

        task.status = "processing"
        task.step = "extracting"
        task.started_at = datetime.now(timezone.utc)
        if not task.device:
            task.device = "cpu"
        db.commit()

        # Fast path: PyMuPDF text extraction for native PDFs
        fast_result = try_pymupdf_text(file_path)
        if fast_result:
            task.status = "completed"
            task.step = None
            task.result_text = fast_result["text"]
            task.metadata_ = fast_result["metadata"]
            task.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # Slow path: split PDF and parallel OCR
        task.step = "splitting"
        db.commit()

        output_dir = tempfile.mkdtemp(prefix="ocr_pages_")
        page_images = split_pdf_to_images(file_path, output_dir)

        task.step = "ocr"
        db.commit()

        page_tasks = [ocr_page.s(img_path, page_num) for page_num, img_path in enumerate(page_images)]
        callback = ocr_aggregate.s(task_id, len(page_images))
        error_callback = ocr_chord_error.s(task_id)
        chord(page_tasks)(callback, link_error=error_callback)

    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


@celery_app.task(name="triton.workers.cpu_ml_tasks.ocr_page")
def ocr_page(image_path: str, page_num: int) -> dict:
    """OCR a single page image. Returns {page_num, text}. Resilient — returns empty text on error."""
    try:
        from triton.services.ocr import extract_text_from_page
        text = extract_text_from_page(image_path)
    except Exception as e:
        text = ""

    # Clean up page image
    try:
        os.remove(image_path)
    except OSError:
        pass

    return {"page_num": page_num, "text": text}


@celery_app.task(name="triton.workers.cpu_ml_tasks.ocr_aggregate")
def ocr_aggregate(page_results: list[dict], task_id: str, total_pages: int):
    """Aggregate parallel OCR results and update task."""
    from triton.database import SessionLocal
    from triton.models import Task
    from datetime import datetime, timezone
    import uuid

    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == uuid.UUID(task_id)).first()
        if not task:
            return

        page_results.sort(key=lambda x: x["page_num"])
        full_text = "\n\n".join(r["text"] for r in page_results if r["text"])

        task.status = "completed"
        task.step = None
        task.result_text = full_text
        task.metadata_ = {"pages": total_pages, "method": "paddleocr"}
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


@celery_app.task(name="triton.workers.cpu_ml_tasks.ocr_chord_error")
def ocr_chord_error(request, exc, traceback, task_id: str):
    """Handle chord failure — mark task as failed instead of leaving it stuck."""
    from triton.database import SessionLocal
    from triton.models import Task
    from datetime import datetime, timezone
    import uuid

    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == uuid.UUID(task_id)).first()
        if task and task.status == "processing":
            task.status = "failed"
            task.error_message = f"OCR chord failed: {exc}"
            task.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


@celery_app.task(name="triton.workers.cpu_ml_tasks.transcribe_cpu")
def transcribe_cpu(task_id: str, file_path: str):
    """Transcribe audio/video using faster-whisper on CPU."""
    from triton.services.task_runner import run_task
    from triton.services.transcriber import transcribe_file
    run_task(task_id, "transcribing", lambda: transcribe_file(file_path), device="cpu")
