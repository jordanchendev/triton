import os
import shutil
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from triton.config import settings
from triton.database import get_db
from triton.models import Task
from triton.schemas import TaskCreate, TaskResponse, TaskListResponse, TaskType

router = APIRouter()


@router.post("", response_model=TaskResponse, status_code=201)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    task = Task(type=payload.type.value, source_url=payload.source_url)
    db.add(task)
    db.commit()
    db.refresh(task)

    # Dispatch to appropriate worker
    task_id = str(task.id)
    if payload.type in (TaskType.youtube, TaskType.podcast):
        from triton.workers.cpu_tasks import download_and_transcribe
        download_and_transcribe.delay(task_id, payload.source_url, payload.type.value)
    elif payload.type in (TaskType.video, TaskType.audio):
        from triton.workers.gpu_tasks import transcribe
        transcribe.delay(task_id, payload.source_url)
    elif payload.type in (TaskType.pdf, TaskType.image):
        from triton.workers.gpu_tasks import ocr
        ocr.delay(task_id, payload.source_url)

    return task


@router.post("/upload", response_model=TaskResponse, status_code=201)
def create_task_with_upload(
    type: TaskType = Query(...),
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

    task_id = str(task.id)
    if type in (TaskType.video, TaskType.audio):
        from triton.workers.gpu_tasks import transcribe
        transcribe.delay(task_id, file_path)
    elif type in (TaskType.pdf, TaskType.image):
        from triton.workers.gpu_tasks import ocr
        ocr.delay(task_id, file_path)

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
