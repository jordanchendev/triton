# Triton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU-accelerated media-to-text REST API service that accepts YouTube URLs, audio/video files, PDFs, and images, converts them to text using faster-whisper and PaddleOCR, and stores results in PostgreSQL.

**Architecture:** FastAPI serves the REST API. Celery with Redis handles async task processing via two separate queues (GPU for transcription/OCR, CPU for downloads). PostgreSQL stores tasks, documents, and schedules. Docker Compose orchestrates all services.

**Tech Stack:** Python 3.11, FastAPI, Celery, Redis, PostgreSQL, SQLAlchemy, Alembic, faster-whisper, PaddleOCR, yt-dlp, Docker Compose

**Spec:** `docs/design.md`

---

## File Structure

```
triton/
├── src/
│   └── triton/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app entry point
│       ├── config.py               # Settings via pydantic-settings
│       ├── database.py             # SQLAlchemy engine + session
│       ├── models.py               # SQLAlchemy ORM models
│       ├── schemas.py              # Pydantic request/response schemas
│       ├── api/
│       │   ├── __init__.py
│       │   ├── tasks.py            # POST/GET/DELETE /tasks
│       │   ├── documents.py        # POST/GET /documents
│       │   ├── schedules.py        # CRUD /schedules
│       │   └── health.py           # GET /health
│       ├── workers/
│       │   ├── __init__.py
│       │   ├── celery_app.py       # Celery config with GPU/CPU queues
│       │   ├── gpu_tasks.py        # Whisper + PaddleOCR celery tasks
│       │   └── cpu_tasks.py        # yt-dlp download celery tasks
│       └── services/
│           ├── __init__.py
│           ├── transcriber.py      # faster-whisper wrapper
│           ├── ocr.py              # PaddleOCR wrapper
│           └── downloader.py       # yt-dlp wrapper
├── tests/
│   ├── conftest.py                 # Fixtures: test DB, test client, mock workers
│   ├── test_api_tasks.py
│   ├── test_api_documents.py
│   ├── test_api_schedules.py
│   ├── test_api_health.py
│   └── test_workers.py
├── alembic/
│   ├── env.py
│   └── versions/
├── alembic.ini
├── docker-compose.yml              # Production with GPU
├── docker-compose.dev.yml          # Local dev without GPU
├── Dockerfile
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/triton/__init__.py`
- Create: `src/triton/config.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "triton"
version = "0.1.0"
description = "GPU-accelerated media-to-text worker for OpenClaw"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "psycopg2-binary>=2.9",
    "pydantic-settings>=2.0",
    "celery[redis]>=5.4",
    "redis>=5.0",
    "yt-dlp>=2024.0",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
gpu = [
    "faster-whisper>=1.0",
    "paddlepaddle-gpu>=2.6",
    "paddleocr>=2.7",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "pytest-cov>=5.0",
]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
.eggs/
.env
*.egg
.venv/
venv/
*.log
.pytest_cache/
.coverage
htmlcov/
```

- [ ] **Step 3: Create .env.example**

```bash
# Database
DATABASE_URL=postgresql://triton:triton@localhost:5432/triton

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000

# Models
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16

# Storage
UPLOAD_DIR=/data/tmp
```

- [ ] **Step 4: Create src/triton/__init__.py**

```python
```

- [ ] **Step 5: Create src/triton/config.py**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://triton:triton@localhost:5432/triton"
    redis_url: str = "redis://localhost:6379/0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    upload_dir: str = "/data/tmp"

    model_config = {"env_file": ".env"}


settings = Settings()
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore .env.example src/triton/__init__.py src/triton/config.py
git commit -m "feat: project scaffolding with config"
```

---

## Task 2: Database Models + Migrations

**Files:**
- Create: `src/triton/database.py`
- Create: `src/triton/models.py`
- Create: `alembic.ini`
- Create: `alembic/env.py`

- [ ] **Step 1: Create src/triton/database.py**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from triton.config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 2: Create src/triton/models.py**

```python
import uuid
from datetime import datetime, timezone

from sqlalchemy import Index, String, Text, Boolean
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from triton.database import Base


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    file_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    step: Mapped[str | None] = mapped_column(String(50))
    result_text: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    started_at: Mapped[datetime | None] = mapped_column()
    completed_at: Mapped[datetime | None] = mapped_column()

    __table_args__ = (
        Index("idx_tasks_status", "status"),
        Index("idx_tasks_type", "type"),
        Index("idx_tasks_created_at", "created_at"),
    )


class Schedule(Base):
    __tablename__ = "schedules"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    cron_expression: Mapped[str] = mapped_column(String(100), nullable=False)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run: Mapped[datetime | None] = mapped_column()
    next_run: Mapped[datetime | None] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_documents_type", "type"),
        Index("idx_documents_created_at", "created_at"),
    )
```

- [ ] **Step 3: Create alembic.ini and alembic/env.py**

`alembic.ini`:
```ini
[alembic]
script_location = alembic
sqlalchemy.url = postgresql://triton:triton@localhost:5432/triton

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

`alembic/env.py`:
```python
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from triton.config import settings
from triton.database import Base
from triton.models import Task, Schedule, Document  # noqa: F401

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 4: Generate initial migration**

Run: `cd /Users/jordanchen/Workspace/Projects/aquarium/triton && alembic revision --autogenerate -m "initial tables"`

- [ ] **Step 5: Commit**

```bash
git add src/triton/database.py src/triton/models.py alembic.ini alembic/
git commit -m "feat: database models and migrations for tasks, schedules, documents"
```

---

## Task 3: Pydantic Schemas

**Files:**
- Create: `src/triton/schemas.py`
- Create: `tests/conftest.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write test for schemas**

`tests/conftest.py`:
```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from triton.database import Base, get_db
from triton.main import app

from fastapi.testclient import TestClient


@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

`tests/test_schemas.py`:
```python
import pytest
from pydantic import ValidationError

from triton.schemas import TaskCreate, DocumentCreate, ScheduleCreate


def test_task_create_with_url():
    task = TaskCreate(type="youtube", source_url="https://youtube.com/watch?v=xxx")
    assert task.type == "youtube"
    assert task.source_url == "https://youtube.com/watch?v=xxx"


def test_task_create_invalid_type():
    with pytest.raises(ValidationError):
        TaskCreate(type="invalid_type", source_url="https://example.com")


def test_document_create():
    doc = DocumentCreate(type="web", content="Hello world", title="Test")
    assert doc.content == "Hello world"


def test_schedule_create():
    sched = ScheduleCreate(
        name="Daily YouTube",
        cron_expression="0 8 * * *",
        type="youtube",
        config={"channel_ids": ["UC123"]},
    )
    assert sched.cron_expression == "0 8 * * *"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jordanchen/Workspace/Projects/aquarium/triton && python -m pytest tests/test_schemas.py -v`
Expected: FAIL — `triton.schemas` does not exist

- [ ] **Step 3: Create src/triton/schemas.py**

```python
import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


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


class TaskCreate(BaseModel):
    type: TaskType
    source_url: str | None = None


class TaskResponse(BaseModel):
    id: uuid.UUID
    type: TaskType
    source_url: str | None
    status: TaskStatus
    step: str | None
    result_text: str | None
    metadata: dict | None
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
    metadata: dict | None
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/triton/schemas.py tests/conftest.py tests/test_schemas.py
git commit -m "feat: pydantic schemas for tasks, documents, schedules"
```

---

## Task 4: FastAPI App + Task Endpoints

**Files:**
- Create: `src/triton/main.py`
- Create: `src/triton/api/__init__.py`
- Create: `src/triton/api/tasks.py`
- Create: `tests/test_api_tasks.py`

- [ ] **Step 1: Write failing tests for task endpoints**

`tests/test_api_tasks.py`:
```python
def test_create_task(client):
    response = client.post("/tasks", json={"type": "youtube", "source_url": "https://youtube.com/watch?v=test"})
    assert response.status_code == 201
    data = response.json()
    assert data["type"] == "youtube"
    assert data["status"] == "queued"
    assert "id" in data


def test_get_task(client):
    create = client.post("/tasks", json={"type": "audio"})
    task_id = create.json()["id"]
    response = client.get(f"/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["id"] == task_id


def test_get_task_not_found(client):
    response = client.get("/tasks/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_list_tasks(client):
    client.post("/tasks", json={"type": "youtube", "source_url": "https://example.com/1"})
    client.post("/tasks", json={"type": "pdf"})
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["tasks"]) == 2


def test_list_tasks_filter_by_type(client):
    client.post("/tasks", json={"type": "youtube", "source_url": "https://example.com/1"})
    client.post("/tasks", json={"type": "pdf"})
    response = client.get("/tasks?type=youtube")
    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_delete_task(client):
    create = client.post("/tasks", json={"type": "audio"})
    task_id = create.json()["id"]
    response = client.delete(f"/tasks/{task_id}")
    assert response.status_code == 204
    get_response = client.get(f"/tasks/{task_id}")
    assert get_response.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_tasks.py -v`
Expected: FAIL

- [ ] **Step 3: Create src/triton/main.py**

```python
from fastapi import FastAPI

from triton.api import tasks, documents, schedules, health

app = FastAPI(title="Triton", description="Media-to-text GPU worker for OpenClaw")

app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(schedules.router, prefix="/schedules", tags=["schedules"])
app.include_router(health.router, tags=["health"])
```

- [ ] **Step 4: Create src/triton/api/__init__.py**

```python
```

- [ ] **Step 5: Create src/triton/api/tasks.py**

```python
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

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
```

- [ ] **Step 6: Create stub routers for documents, schedules, health**

These are needed so `main.py` imports don't fail. Full implementation in Tasks 5-7.

`src/triton/api/documents.py`:
```python
from fastapi import APIRouter
router = APIRouter()
```

`src/triton/api/schedules.py`:
```python
from fastapi import APIRouter
router = APIRouter()
```

`src/triton/api/health.py`:
```python
from fastapi import APIRouter
router = APIRouter()
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_tasks.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/triton/main.py src/triton/api/ tests/test_api_tasks.py
git commit -m "feat: task API endpoints with CRUD operations"
```

---

## Task 5: Document Endpoints

**Files:**
- Modify: `src/triton/api/documents.py`
- Create: `tests/test_api_documents.py`

- [ ] **Step 1: Write failing tests**

`tests/test_api_documents.py`:
```python
def test_create_document(client):
    response = client.post("/documents", json={
        "type": "web",
        "title": "Market Analysis",
        "content": "S&P 500 rose 2% today...",
        "source_url": "https://example.com/article",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["content"] == "S&P 500 rose 2% today..."
    assert "id" in data


def test_get_document(client):
    create = client.post("/documents", json={"type": "tweet", "content": "BTC to 100k"})
    doc_id = create.json()["id"]
    response = client.get(f"/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["content"] == "BTC to 100k"


def test_get_document_not_found(client):
    response = client.get("/documents/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404


def test_list_documents(client):
    client.post("/documents", json={"type": "web", "content": "Article 1"})
    client.post("/documents", json={"type": "tweet", "content": "Tweet 1"})
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json()["total"] == 2


def test_list_documents_filter_by_type(client):
    client.post("/documents", json={"type": "web", "content": "Article 1"})
    client.post("/documents", json={"type": "tweet", "content": "Tweet 1"})
    response = client.get("/documents?type=web")
    assert response.status_code == 200
    assert response.json()["total"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_documents.py -v`
Expected: FAIL

- [ ] **Step 3: Implement src/triton/api/documents.py**

```python
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from triton.database import get_db
from triton.models import Document
from triton.schemas import DocumentCreate, DocumentResponse, DocumentListResponse

router = APIRouter()


@router.post("", response_model=DocumentResponse, status_code=201)
def create_document(payload: DocumentCreate, db: Session = Depends(get_db)):
    doc = Document(
        type=payload.type,
        source_url=payload.source_url,
        title=payload.title,
        content=payload.content,
        metadata_=payload.metadata,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("", response_model=DocumentListResponse)
def list_documents(
    type: str | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(Document)
    if type:
        query = query.filter(Document.type == type)
    total = query.count()
    documents = query.order_by(Document.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return DocumentListResponse(documents=documents, total=total, page=page, page_size=page_size)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_documents.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/triton/api/documents.py tests/test_api_documents.py
git commit -m "feat: document API endpoints for lobster to write text directly"
```

---

## Task 6: Schedule Endpoints

**Files:**
- Modify: `src/triton/api/schedules.py`
- Create: `tests/test_api_schedules.py`

- [ ] **Step 1: Write failing tests**

`tests/test_api_schedules.py`:
```python
def test_create_schedule(client):
    response = client.post("/schedules", json={
        "name": "Daily YouTube Check",
        "cron_expression": "0 8 * * *",
        "type": "youtube",
        "config": {"channel_ids": ["UC123"]},
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Daily YouTube Check"
    assert data["enabled"] is True


def test_get_schedules(client):
    client.post("/schedules", json={
        "name": "Job 1", "cron_expression": "0 8 * * *",
        "type": "youtube", "config": {},
    })
    response = client.get("/schedules")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_update_schedule(client):
    create = client.post("/schedules", json={
        "name": "Job 1", "cron_expression": "0 8 * * *",
        "type": "youtube", "config": {},
    })
    sched_id = create.json()["id"]
    response = client.put(f"/schedules/{sched_id}", json={"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False


def test_delete_schedule(client):
    create = client.post("/schedules", json={
        "name": "Job 1", "cron_expression": "0 8 * * *",
        "type": "youtube", "config": {},
    })
    sched_id = create.json()["id"]
    response = client.delete(f"/schedules/{sched_id}")
    assert response.status_code == 204
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_schedules.py -v`
Expected: FAIL

- [ ] **Step 3: Implement src/triton/api/schedules.py**

```python
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from triton.database import get_db
from triton.models import Schedule
from triton.schemas import ScheduleCreate, ScheduleUpdate, ScheduleResponse

router = APIRouter()


@router.post("", response_model=ScheduleResponse, status_code=201)
def create_schedule(payload: ScheduleCreate, db: Session = Depends(get_db)):
    schedule = Schedule(
        name=payload.name,
        cron_expression=payload.cron_expression,
        type=payload.type.value,
        config=payload.config,
    )
    db.add(schedule)
    db.commit()
    db.refresh(schedule)
    return schedule


@router.get("", response_model=list[ScheduleResponse])
def list_schedules(db: Session = Depends(get_db)):
    return db.query(Schedule).order_by(Schedule.created_at.desc()).all()


@router.put("/{schedule_id}", response_model=ScheduleResponse)
def update_schedule(schedule_id: uuid.UUID, payload: ScheduleUpdate, db: Session = Depends(get_db)):
    schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(schedule, field, value)
    db.commit()
    db.refresh(schedule)
    return schedule


@router.delete("/{schedule_id}", status_code=204)
def delete_schedule(schedule_id: uuid.UUID, db: Session = Depends(get_db)):
    schedule = db.query(Schedule).filter(Schedule.id == schedule_id).first()
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    db.delete(schedule)
    db.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_schedules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/triton/api/schedules.py tests/test_api_schedules.py
git commit -m "feat: schedule CRUD endpoints"
```

---

## Task 7: Health Endpoint

**Files:**
- Modify: `src/triton/api/health.py`
- Create: `tests/test_api_health.py`

- [ ] **Step 1: Write failing test**

`tests/test_api_health.py`:
```python
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gpu_available" in data
    assert "queue_gpu_length" in data
    assert "queue_cpu_length" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_health.py -v`
Expected: FAIL

- [ ] **Step 3: Implement src/triton/api/health.py**

```python
from fastapi import APIRouter

from triton.schemas import HealthResponse

router = APIRouter()


def _get_gpu_info() -> tuple[bool, int | None, int | None]:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(", ")
            return True, int(used), int(total)
    except (FileNotFoundError, Exception):
        pass
    return False, None, None


def _get_queue_length(queue_name: str) -> int:
    try:
        from triton.workers.celery_app import celery_app
        with celery_app.connection_or_acquire() as conn:
            return conn.default_channel.queue_declare(queue_name, passive=True).message_count
    except Exception:
        return 0


@router.get("/health", response_model=HealthResponse)
def health_check():
    gpu_available, gpu_used, gpu_total = _get_gpu_info()
    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        gpu_memory_used_mb=gpu_used,
        gpu_memory_total_mb=gpu_total,
        queue_gpu_length=_get_queue_length("gpu"),
        queue_cpu_length=_get_queue_length("cpu"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_api_health.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/triton/api/health.py tests/test_api_health.py
git commit -m "feat: health endpoint with GPU and queue status"
```

---

## Task 8: Celery Setup + Worker Configuration

**Files:**
- Create: `src/triton/workers/__init__.py`
- Create: `src/triton/workers/celery_app.py`
- Create: `src/triton/workers/gpu_tasks.py`
- Create: `src/triton/workers/cpu_tasks.py`

- [ ] **Step 1: Create src/triton/workers/__init__.py**

```python
```

- [ ] **Step 2: Create src/triton/workers/celery_app.py**

```python
from celery import Celery

from triton.config import settings

celery_app = Celery("triton", broker=settings.redis_url, backend=settings.redis_url)

celery_app.conf.update(
    task_routes={
        "triton.workers.gpu_tasks.*": {"queue": "gpu"},
        "triton.workers.cpu_tasks.*": {"queue": "cpu"},
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,  # GPU tasks are heavy, process one at a time
)
```

- [ ] **Step 3: Create src/triton/workers/gpu_tasks.py**

```python
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
```

- [ ] **Step 4: Create src/triton/workers/cpu_tasks.py**

```python
from triton.workers.celery_app import celery_app


@celery_app.task(name="triton.workers.cpu_tasks.download_and_transcribe")
def download_and_transcribe(task_id: str, url: str, task_type: str):
    """Download media from URL then dispatch GPU transcription."""
    from triton.services.downloader import download_audio
    from triton.database import SessionLocal
    from triton.models import Task
    from triton.workers.gpu_tasks import transcribe
    from datetime import datetime, timezone
    import uuid

    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == uuid.UUID(task_id)).first()
        if not task:
            return

        task.status = "downloading"
        task.step = "downloading"
        task.started_at = datetime.now(timezone.utc)
        db.commit()

        file_path = download_audio(url, task_type)
        task.file_path = file_path
        db.commit()

        # Chain to GPU worker
        transcribe.delay(task_id, file_path)
    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        task.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        db.close()
```

- [ ] **Step 5: Commit**

```bash
git add src/triton/workers/
git commit -m "feat: celery workers with GPU/CPU queue separation"
```

---

## Task 9: Service Wrappers (Transcriber, OCR, Downloader)

**Files:**
- Create: `src/triton/services/__init__.py`
- Create: `src/triton/services/transcriber.py`
- Create: `src/triton/services/ocr.py`
- Create: `src/triton/services/downloader.py`
- Create: `tests/test_workers.py`

- [ ] **Step 1: Create src/triton/services/__init__.py**

```python
```

- [ ] **Step 2: Create src/triton/services/transcriber.py**

```python
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
```

- [ ] **Step 3: Create src/triton/services/ocr.py**

```python
_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True)
    return _ocr


def extract_text(file_path: str) -> dict:
    """Extract text from PDF or image using PaddleOCR."""
    ocr = _get_ocr()
    results = ocr.ocr(file_path, cls=True)

    text_parts = []
    for page in results:
        if page is None:
            continue
        for line in page:
            text_parts.append(line[1][0])

    return {
        "text": "\n".join(text_parts),
        "metadata": {
            "pages": len(results),
        },
    }
```

- [ ] **Step 4: Create src/triton/services/downloader.py**

```python
import os

from triton.config import settings


def download_audio(url: str, task_type: str) -> str:
    """Download audio from URL using yt-dlp. Returns path to downloaded file."""
    import yt_dlp

    os.makedirs(settings.upload_dir, exist_ok=True)
    output_template = os.path.join(settings.upload_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info["id"]
        return os.path.join(settings.upload_dir, f"{video_id}.wav")
```

- [ ] **Step 5: Write tests with mocked services**

`tests/test_workers.py`:
```python
from unittest.mock import patch, MagicMock


def test_transcribe_file_returns_text():
    mock_segment = MagicMock()
    mock_segment.text = "Hello world"
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95
    mock_info.duration = 10.5

    with patch("triton.services.transcriber._get_model") as mock_model:
        mock_model.return_value.transcribe.return_value = ([mock_segment], mock_info)
        from triton.services.transcriber import transcribe_file
        result = transcribe_file("/tmp/test.wav")
        assert result["text"] == "Hello world"
        assert result["metadata"]["language"] == "en"


def test_extract_text_returns_text():
    mock_result = [[([0, 0], ("Hello", 0.99)), ([0, 0], ("World", 0.98))]]

    with patch("triton.services.ocr._get_ocr") as mock_ocr:
        mock_ocr.return_value.ocr.return_value = mock_result
        from triton.services.ocr import extract_text
        result = extract_text("/tmp/test.png")
        assert "Hello" in result["text"]
        assert "World" in result["text"]
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_workers.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/triton/services/ tests/test_workers.py
git commit -m "feat: service wrappers for faster-whisper, PaddleOCR, yt-dlp"
```

---

## Task 10: Wire Task Creation to Celery Dispatch

**Files:**
- Modify: `src/triton/api/tasks.py`

- [ ] **Step 1: Update create_task to dispatch celery tasks**

In `src/triton/api/tasks.py`, update the `create_task` function:

```python
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
```

- [ ] **Step 2: Update conftest.py to mock celery**

Add to `tests/conftest.py`:
```python
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_celery():
    with patch("triton.workers.cpu_tasks.download_and_transcribe.delay"), \
         patch("triton.workers.gpu_tasks.transcribe.delay"), \
         patch("triton.workers.gpu_tasks.ocr.delay"):
        yield
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/triton/api/tasks.py tests/conftest.py
git commit -m "feat: wire task creation to celery dispatch"
```

---

## Task 11: File Upload Support

**Files:**
- Modify: `src/triton/api/tasks.py`

- [ ] **Step 1: Add file upload endpoint**

Add to `src/triton/api/tasks.py`:

```python
import os
import shutil

from fastapi import UploadFile, File

from triton.config import settings


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
```

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add src/triton/api/tasks.py
git commit -m "feat: file upload endpoint for video, audio, pdf, image"
```

---

## Task 12: Docker Compose

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `docker-compose.dev.yml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini .

CMD ["uvicorn", "triton.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create docker-compose.dev.yml (local dev, no GPU)**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://triton:triton@postgres:5432/triton
      - REDIS_URL=redis://redis:6379/0
      - UPLOAD_DIR=/data/tmp
    volumes:
      - upload_data:/data/tmp
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started

  cpu-worker:
    build: .
    command: celery -A triton.workers.celery_app worker -Q cpu -c 2 --loglevel=info
    environment:
      - DATABASE_URL=postgresql://triton:triton@postgres:5432/triton
      - REDIS_URL=redis://redis:6379/0
      - UPLOAD_DIR=/data/tmp
    volumes:
      - upload_data:/data/tmp
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: triton
      POSTGRES_PASSWORD: triton
      POSTGRES_DB: triton
    ports:
      - "5432:5432"
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U triton"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pg_data:
  upload_data:
```

- [ ] **Step 3: Create docker-compose.yml (production with GPU)**

```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://triton:triton@postgres:5432/triton
      - REDIS_URL=redis://redis:6379/0
      - UPLOAD_DIR=/data/tmp
    volumes:
      - /data/tmp:/data/tmp
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped

  gpu-worker:
    build:
      context: .
      dockerfile: Dockerfile
    command: celery -A triton.workers.celery_app worker -Q gpu -c 1 --loglevel=info
    environment:
      - DATABASE_URL=postgresql://triton:triton@postgres:5432/triton
      - REDIS_URL=redis://redis:6379/0
      - WHISPER_DEVICE=cuda
      - UPLOAD_DIR=/data/tmp
    volumes:
      - /data/tmp:/data/tmp
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  cpu-worker:
    build: .
    command: celery -A triton.workers.celery_app worker -Q cpu -c 2 --loglevel=info
    environment:
      - DATABASE_URL=postgresql://triton:triton@postgres:5432/triton
      - REDIS_URL=redis://redis:6379/0
      - UPLOAD_DIR=/data/tmp
    volumes:
      - /data/tmp:/data/tmp
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    volumes:
      - /data/redis:/data
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: triton
      POSTGRES_PASSWORD: triton
      POSTGRES_DB: triton
    volumes:
      - /data/postgres:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U triton"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped
```

- [ ] **Step 4: Test local dev compose**

Run: `cd /Users/jordanchen/Workspace/Projects/aquarium/triton && docker compose -f docker-compose.dev.yml up --build -d`
Then: `curl http://localhost:8000/health`
Expected: JSON response with `"status": "ok"`

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml docker-compose.dev.yml
git commit -m "feat: Docker Compose for dev and production environments"
```

---

## Task 13: Run Migrations in Docker + Initial Push

- [ ] **Step 1: Add migration command to startup**

Create `scripts/entrypoint.sh`:
```bash
#!/bin/bash
set -e
alembic upgrade head
exec "$@"
```

Update Dockerfile to add:
```dockerfile
COPY scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh
ENTRYPOINT ["scripts/entrypoint.sh"]
```

- [ ] **Step 2: Test full stack locally**

```bash
docker compose -f docker-compose.dev.yml up --build -d
# Wait for startup
curl -s http://localhost:8000/health | python -m json.tool
curl -s -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"type": "web", "content": "Test document", "title": "Test"}' | python -m json.tool
curl -s http://localhost:8000/documents | python -m json.tool
```

Expected: All return valid JSON responses

- [ ] **Step 3: Commit and push**

```bash
git add scripts/ Dockerfile
git commit -m "feat: entrypoint with auto-migration"
git push -u origin main
```

---

## Summary

| Task | Description | Depends On |
|------|-------------|------------|
| 1 | Project scaffolding | - |
| 2 | Database models + migrations | 1 |
| 3 | Pydantic schemas | 1 |
| 4 | Task API endpoints | 2, 3 |
| 5 | Document API endpoints | 2, 3 |
| 6 | Schedule API endpoints | 2, 3 |
| 7 | Health endpoint | 3 |
| 8 | Celery setup + workers | 1 |
| 9 | Service wrappers | 1 |
| 10 | Wire tasks to Celery | 4, 8 |
| 11 | File upload support | 10 |
| 12 | Docker Compose | all above |
| 13 | Migrations + push | 12 |
