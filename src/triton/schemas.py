import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    youtube = "youtube"
    podcast = "podcast"
    video = "video"
    audio = "audio"
    pdf = "pdf"
    image = "image"


class TaskStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class DeviceType(str, Enum):
    gpu = "gpu"
    cpu = "cpu"
    auto = "auto"


class TaskCreate(BaseModel):
    type: TaskType
    source_url: str | None = None
    device: DeviceType = DeviceType.auto


class TaskResponse(BaseModel):
    id: uuid.UUID
    type: TaskType
    source_url: str | None
    status: TaskStatus
    step: str | None
    result_text: str | None
    metadata: dict | None = Field(None, validation_alias="metadata_")
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    total: int
    page: int
    page_size: int


class DocumentCreate(BaseModel):
    type: str
    source_url: str | None = None
    title: str | None = None
    content: str
    metadata: dict | None = None


class DocumentResponse(BaseModel):
    id: uuid.UUID
    type: str
    source_url: str | None
    title: str | None
    content: str
    metadata: dict | None = Field(None, validation_alias="metadata_")
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    page: int
    page_size: int


class ScheduleCreate(BaseModel):
    name: str
    cron_expression: str
    type: TaskType
    config: dict


class ScheduleUpdate(BaseModel):
    name: str | None = None
    cron_expression: str | None = None
    config: dict | None = None
    enabled: bool | None = None


class ScheduleResponse(BaseModel):
    id: uuid.UUID
    name: str
    cron_expression: str
    type: str
    config: dict
    enabled: bool
    last_run: datetime | None
    next_run: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_memory_used_mb: int | None
    gpu_memory_total_mb: int | None
    queue_gpu_length: int
    queue_cpu_length: int
