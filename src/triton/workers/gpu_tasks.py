from triton.workers.celery_app import celery_app


@celery_app.task(name="triton.workers.gpu_tasks.transcribe")
def transcribe(task_id: str, file_path: str):
    """Transcribe audio/video file using faster-whisper."""
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


@celery_app.task(name="triton.workers.gpu_tasks.ocr")
def ocr(task_id: str, file_path: str):
    """Extract text from PDF/image using PaddleOCR."""
    from triton.services.ocr import extract_text
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
        task.step = "ocr"
        task.started_at = datetime.now(timezone.utc)
        db.commit()

        result = extract_text(file_path)

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
