import os

from celery import chord

from triton.workers.celery_app import celery_app


@celery_app.task(name="triton.workers.cpu_ml_tasks.ocr_parallel")
def ocr_parallel(task_id: str, file_path: str):
    """Split PDF into pages and dispatch parallel OCR tasks."""
    from triton.services.ocr import split_pdf_to_images
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
        task.step = "splitting"
        task.started_at = datetime.now(timezone.utc)
        db.commit()

        # Split PDF into page images
        output_dir = tempfile.mkdtemp(prefix="ocr_pages_")
        page_images = split_pdf_to_images(file_path, output_dir)

        task.step = "ocr"
        db.commit()

        # Dispatch parallel OCR for each page, then aggregate
        page_tasks = [ocr_page.s(img_path, page_num) for page_num, img_path in enumerate(page_images)]
        callback = ocr_aggregate.s(task_id, len(page_images))
        chord(page_tasks)(callback)

    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


@celery_app.task(name="triton.workers.cpu_ml_tasks.ocr_page")
def ocr_page(image_path: str, page_num: int) -> dict:
    """OCR a single page image. Returns {page_num, text}."""
    from triton.services.ocr import extract_text_from_page

    text = extract_text_from_page(image_path)

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

        # Sort by page number and combine
        page_results.sort(key=lambda x: x["page_num"])
        full_text = "\n\n".join(r["text"] for r in page_results if r["text"])

        task.status = "completed"
        task.step = None
        task.result_text = full_text
        task.metadata_ = {"pages": total_pages}
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()


@celery_app.task(name="triton.workers.cpu_ml_tasks.transcribe_cpu")
def transcribe_cpu(task_id: str, file_path: str):
    """Transcribe audio/video using faster-whisper on CPU."""
    from triton.services.transcriber import transcribe_file
    from triton.database import SessionLocal
    from triton.models import Task
    from datetime import datetime, timezone
    import uuid

    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == uuid.UUID(task_id)).first()
        if not task:
            return

        task.status = "processing"
        task.step = "transcribing"
        task.started_at = datetime.now(timezone.utc)
        db.commit()

        result = transcribe_file(file_path)

        task.status = "completed"
        task.step = None
        task.result_text = result["text"]
        task.metadata_ = result.get("metadata")
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()
