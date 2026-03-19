# Triton

GPU-accelerated media-to-text REST API. Accepts YouTube URLs, audio/video files, PDFs, and images, converts them to text using faster-whisper and PaddleOCR, and stores results in PostgreSQL.

## Architecture

```
┌───────────┐    ┌────────────┐    ┌───────────────┐
│  Client   │───▶│  FastAPI   │───▶│  PostgreSQL   │
│           │    │ (REST API) │    │ (tasks, docs) │
└───────────┘    └─────┬──────┘    └───────────────┘
                     │
                ┌────▼─────┐
                │  Redis   │
                │ (broker) │
                └────┬─────┘
           ┌─────────┴─────────┐
     ┌─────▼──────┐       ┌────▼─────┐
     │ GPU Worker │       │CPU Worker│
     │ -whisper   │       │ -yt-dlp  │
     │ -PaddleOCR │       │ -download│
     └────────────┘       └──────────┘
```

- **API**: FastAPI REST endpoints for tasks, documents, schedules, health
- **GPU Worker**: Celery worker running faster-whisper (transcription) and PaddleOCR (text extraction)
- **CPU Worker**: Celery worker running yt-dlp (YouTube/podcast downloads), chains to GPU worker
- **Two queues**: `gpu` and `cpu` to prevent download tasks from blocking GPU processing

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/tasks` | Create task (auto-dispatches to worker) |
| POST | `/tasks/upload` | Upload file for processing |
| GET | `/tasks/{id}` | Get task status and result |
| GET | `/tasks` | List tasks (filter by type, status) |
| DELETE | `/tasks/{id}` | Delete task |
| POST | `/documents` | Store text document |
| GET | `/documents/{id}` | Get document |
| GET | `/documents` | List documents (filter by type) |
| POST | `/schedules` | Create recurring schedule |
| GET | `/schedules` | List schedules |
| PUT | `/schedules/{id}` | Update schedule |
| DELETE | `/schedules/{id}` | Delete schedule |
| GET | `/health` | GPU status, VRAM, queue lengths |

### Task Types

| Type | Source | Worker |
|------|--------|--------|
| `youtube` | URL | CPU → GPU (download then transcribe) |
| `podcast` | URL | CPU → GPU (download then transcribe) |
| `video` | File/URL | GPU (transcribe) |
| `audio` | File/URL | GPU (transcribe) |
| `pdf` | File/URL | GPU (OCR) |
| `image` | File/URL | GPU (OCR) |

## Tech Stack

- Python 3.11, FastAPI, Celery, Redis, PostgreSQL, SQLAlchemy, Alembic
- **faster-whisper** (large-v3, CTranslate2) — ~5GB VRAM
- **PaddleOCR** (chinese_cht) — ~2GB VRAM
- **yt-dlp** for YouTube/podcast downloads
- Docker Compose (5 services)

## Quick Start

### Local Development (no GPU)

```bash
docker compose -f docker-compose.dev.yml up --build -d
curl http://localhost:8000/health
```

### Production (with NVIDIA GPU)

```bash
docker compose up --build -d
```

### Run Tests

```bash
uv venv && uv pip install -e ".[dev]"
uv run python -m pytest tests/ -v
```

## Environment Variables

See [`.env.example`](.env.example) for all available settings.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://triton:triton@localhost:5432/triton` | PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis broker |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `UPLOAD_DIR` | `/data/tmp` | Temp file storage |

