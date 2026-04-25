from datetime import UTC

from triton.workers.celery_app import celery_app


@celery_app.task(name="triton.workers.cpu_tasks.download_and_transcribe")
def download_and_transcribe(task_id: str, url: str, task_type: str, device: str = "gpu"):
    """Download media from URL then dispatch transcription to GPU or CPU."""
    import uuid
    from datetime import datetime

    from triton.database import SessionLocal
    from triton.models import Task
    from triton.services.downloader import download_audio

    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == uuid.UUID(task_id)).first()
        if not task:
            return

        task.status = "downloading"
        task.step = "downloading"
        task.started_at = datetime.now(UTC)
        db.commit()

        file_path = download_audio(url, task_type)
        task.file_path = file_path
        db.commit()

        # Chain to transcription worker based on device
        if device == "cpu":
            from triton.workers.cpu_ml_tasks import transcribe_cpu

            transcribe_cpu.delay(task_id, file_path)
        else:
            from triton.workers.gpu_tasks import transcribe

            transcribe.delay(task_id, file_path)
    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now(UTC)
        db.commit()
    finally:
        db.close()
