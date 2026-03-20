import os
import shutil
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from triton.config import settings
from triton.database import get_db
from triton.models import Task
from triton.schemas import TaskCreate, TaskResponse, TaskListResponse, TaskType, DeviceType

router = APIRouter()


def _resolve_device(device: DeviceType) -> str:
    """Resolve 'auto' device to 'gpu' or 'cpu' based on queue status."""
    if device != DeviceType.auto:
        return device.value
    try:
        from triton.workers.celery_app import celery_app
        inspect = celery_app.control.inspect(timeout=1)
        active = inspect.active() or {}
        # Check if any gpu worker has active tasks
        for worker_name, tasks in active.items():
            if "gpu" in worker_name and len(tasks) > 0:
                return "cpu"
        return "gpu"
    except Exception:
        return "gpu"


def _dispatch_transcribe(task_id: str, source: str, device: str):
    """Dispatch transcription to GPU or CPU worker."""
    if device == "cpu":
        from triton.workers.cpu_ml_tasks import transcribe_cpu
        transcribe_cpu.delay(task_id, source)
    else:
        from triton.workers.gpu_tasks import transcribe
        transcribe.delay(task_id, source)


def _dispatch_ocr(task_id: str, file_path: str, device: str):
    """Dispatch OCR to GPU (single) or CPU (parallel pages)."""
    if device == "cpu":
        from triton.workers.cpu_ml_tasks import ocr_parallel
        ocr_parallel.delay(task_id, file_path)
    else:
        from triton.workers.gpu_tasks import ocr
        ocr.delay(task_id, file_path)


@router.post("", response_model=TaskResponse, status_code=201)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    task = Task(type=payload.type.value, source_url=payload.source_url)
    db.add(task)
    db.commit()
    db.refresh(task)

    device = _resolve_device(payload.device)
    task_id = str(task.id)

    if payload.type in (TaskType.youtube, TaskType.podcast):
        from triton.workers.cpu_tasks import download_and_transcribe
        download_and_transcribe.delay(task_id, payload.source_url, payload.type.value)
    elif payload.type in (TaskType.video, TaskType.audio):
        _dispatch_transcribe(task_id, payload.source_url, device)
    elif payload.type in (TaskType.pdf, TaskType.image):
        _dispatch_ocr(task_id, payload.source_url, device)

    return task


@router.post("/upload", response_model=TaskResponse, status_code=201)
def create_task_with_upload(
    type: TaskType = Query(...),
    device: DeviceType = Query(DeviceType.auto),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if type not in (TaskType.video, TaskType.audio, TaskType.pdf, TaskType.image):
        raise HTTPException(status_code=400, detail="Upload only supported for video, audio, pdf, image")

    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{uuid.uuid4()}_{file.filename}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    task = Task(type=type.value, file_path=file_path)
    db.add(task)
    db.commit()
    db.refresh(task)

    resolved_device = _resolve_device(device)
    task_id = str(task.id)

    if type in (TaskType.video, TaskType.audio):
        _dispatch_transcribe(task_id, file_path, resolved_device)
    elif type in (TaskType.pdf, TaskType.image):
        _dispatch_ocr(task_id, file_path, resolved_device)

    return task


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: uuid.UUID, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("", response_model=TaskListResponse)
def list_tasks(
    type: TaskType | None = None,
    status: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(Task)
    if type:
        query = query.filter(Task.type == type.value)
    if status:
        query = query.filter(Task.status == status)
    total = query.count()
    tasks = query.order_by(Task.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return TaskListResponse(tasks=tasks, total=total, page=page, page_size=page_size)


@router.delete("/{task_id}", status_code=204)
def delete_task(task_id: uuid.UUID, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(task)
    db.commit()
