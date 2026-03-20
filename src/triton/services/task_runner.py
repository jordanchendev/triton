import uuid
from datetime import datetime, timezone
from typing import Callable

from triton.database import SessionLocal
from triton.models import Task


def run_task(task_id: str, step_name: str, process_fn: Callable[[], dict], device: str | None = None):
    """Generic task runner that handles DB lifecycle, status transitions, and error handling.

    Args:
        task_id: UUID string of the task
        step_name: Step label (e.g. "transcribing", "ocr")
        process_fn: Callable that returns {"text": ..., "metadata": ...}
        device: Device used ("gpu", "cpu", or None)
    """
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == uuid.UUID(task_id)).first()
        if not task:
            return

        task.status = "processing"
        task.step = step_name
        task.started_at = datetime.now(timezone.utc)
        if device:
            task.device = device
        db.commit()

        result = process_fn()

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
