# OpenClaw Stormtrooper — Media-to-Text GPU Worker

**Date**: 2026-03-19
**Status**: Approved
**Machine**: stormtrooper (i7-8700, 32GB RAM, RTX 4070 Ti SUPER 16GB)

---

## 1. System Purpose

Stormtrooper serves as OpenClaw's dedicated GPU-accelerated media-to-text worker.
It receives tasks via REST API from the lobster or auto-fetches from
scheduled sources. All results are stored in PostgreSQL for downstream consumption.

**Scope**: Media to text conversion only. Analysis is handled by the lobster.

**Investment domains**: US stocks, Taiwan stocks, cryptocurrency.

---

## 2. Architecture — Task-Oriented Monolith (Option C)

```
Lobster --REST API--> FastAPI (API + Scheduler)
                           |
                           v
                       Redis Queue
                      /          \
                GPU Queue      CPU Queue
                   |              |
              GPU Worker     CPU Worker
                   |              |
                   \      v      /
                     PostgreSQL
```

### Why two queues?

GPU is a scarce resource. Whisper and PaddleOCR need GPU; downloading YouTube
videos does not. Splitting into GPU/CPU queues prevents CPU-bound tasks from
waiting behind GPU-bound tasks, and vice versa.

---

## 3. Task Types & Worker Assignment

| Task        | Input      | CPU Worker          | GPU Worker                | Output                |
|-------------|------------|---------------------|---------------------------|-----------------------|
| YouTube     | URL        | yt-dlp download     | faster-whisper transcribe | Transcript + metadata |
| Podcast     | RSS/URL    | Download audio      | faster-whisper transcribe | Transcript + metadata |
| Video file  | Upload     | -                   | faster-whisper transcribe | Transcript            |
| Audio file  | Upload     | -                   | faster-whisper transcribe | Transcript            |
| PDF         | Upload     | -                   | PaddleOCR                 | Full text             |
| Image       | Upload     | -                   | PaddleOCR                 | Recognized text       |

Note: Web/News and Twitter/X are text-native sources. The lobster handles
scraping those directly and writes results to stormtrooper's DB via the
documents API (see Section 6).

---

## 4. Technology Stack

| Component       | Technology                 | Reason                                           |
|-----------------|----------------------------|--------------------------------------------------|
| API framework   | FastAPI                    | Async, auto OpenAPI docs                         |
| Task queue      | Celery + Redis             | Mature, supports GPU/CPU queue separation        |
| Speech-to-text  | faster-whisper (large-v3)  | 4x faster than openai/whisper, half VRAM usage   |
| OCR             | PaddleOCR                  | Strong CJK support, GPU accelerated              |
| YouTube DL      | yt-dlp                     | Standard tool, actively maintained               |
| Database        | PostgreSQL                 | Structured storage, full-text search             |
| Scheduler       | APScheduler / Celery Beat  | Cron-style scheduled tasks                       |
| Deployment      | Docker Compose             | Single command to start all services             |

---

## 5. Disk Layout

```
nvme1n1 (234G, NVMe SSD)   -> OS + Docker + model files (faster-whisper, PaddleOCR)
sdb     (233G, SATA SSD)    -> PostgreSQL + Redis + temp processing files
nvme0n1 (477G, NVMe)        -> Windows (DO NOT TOUCH)
sda/sdc/sdd (HDD)           -> Not used for daily operations
```

### Mount plan for sdb

Mount sdb to `/data` with subdirectories:
- `/data/postgres/` - PostgreSQL data
- `/data/redis/` - Redis persistence
- `/data/tmp/` - Temporary files during processing (downloaded audio, uploaded files)

---

## 6. API Design

### Endpoints

```
POST   /tasks              -> Create task (URL or file upload)
GET    /tasks/{id}         -> Query task status and result
GET    /tasks              -> List tasks (paginated, filterable)
DELETE /tasks/{id}         -> Cancel/delete task

POST   /schedules          -> Create scheduled job
GET    /schedules           -> List schedules
PUT    /schedules/{id}     -> Update schedule
DELETE /schedules/{id}     -> Delete schedule

GET    /health              -> Health check (GPU status, queue length)

POST   /documents           -> Lobster writes processed text directly to DB
GET    /documents/{id}      -> Query a document
GET    /documents           -> List documents (paginated, filterable)
```

### Task lifecycle

```
queued -> downloading -> processing -> completed
                                    -> failed (with error message)
```

### Example: YouTube transcription

```
1. Create task
POST /tasks
{
  "type": "youtube",
  "url": "https://youtube.com/watch?v=xxx"
}
Response: {"task_id": "abc123", "status": "queued"}

2. Check progress
GET /tasks/abc123
Response: {"task_id": "abc123", "status": "processing", "step": "transcribing"}

3. Get result
GET /tasks/abc123
Response: {
  "task_id": "abc123",
  "status": "completed",
  "result": {
    "text": "...",
    "source": "youtube",
    "duration": 1832,
    "language": "zh"
  }
}
```

---

## 7. Database Schema

```sql
CREATE TABLE tasks (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type          VARCHAR(20) NOT NULL,
    source_url    TEXT,
    file_path     TEXT,
    status        VARCHAR(20) NOT NULL DEFAULT 'queued',
    step          VARCHAR(50),
    result_text   TEXT,
    metadata      JSONB,
    error_message TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at    TIMESTAMPTZ,
    completed_at  TIMESTAMPTZ
);

CREATE TABLE schedules (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    cron_expression VARCHAR(100) NOT NULL,
    type            VARCHAR(20) NOT NULL,
    config          JSONB NOT NULL,
    enabled         BOOLEAN DEFAULT TRUE,
    last_run        TIMESTAMPTZ,
    next_run        TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE documents (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type          VARCHAR(20) NOT NULL,
    source_url    TEXT,
    title         TEXT,
    content       TEXT NOT NULL,
    metadata      JSONB,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_type ON tasks(type);
CREATE INDEX idx_tasks_created_at ON tasks(created_at DESC);
CREATE INDEX idx_documents_type ON documents(type);
CREATE INDEX idx_documents_created_at ON documents(created_at DESC);
```

---

## 8. Prerequisites (Before Implementation)

| Step | Action                                    | Notes                                      |
|------|-------------------------------------------|--------------------------------------------|
| 1    | Upgrade Ubuntu 18.04 to 22.04 or 24.04   | 18.04 is EOL, many packages won't install  |
| 2    | Update NVIDIA driver                      | Latest stable for RTX 4070 Ti SUPER        |
| 3    | Install Docker + NVIDIA Container Toolkit | Enable GPU access inside containers        |
| 4    | Format and mount sdb to /data             | ext4, add to /etc/fstab for auto-mount     |
| 5    | Clean up unused software                  | Remove anaconda3, old CUDA samples, etc.   |

---

## 9. Docker Compose Services

```yaml
services:
  api:         # FastAPI + APScheduler
  redis:       # Task queue broker
  gpu-worker:  # Celery worker with GPU access (faster-whisper, PaddleOCR)
  cpu-worker:  # Celery worker for downloads/scraping
  postgres:    # Database
```

Each service runs as a Docker container. GPU worker uses
`deploy.resources.reservations.devices` to access the RTX 4070 Ti SUPER.

---

## 10. VRAM Budget

| Model                    | VRAM    |
|--------------------------|---------|
| faster-whisper large-v3  | ~5 GB   |
| PaddleOCR                | ~2 GB   |
| **Total**                | **~7 GB** |
| **Available**            | **16 GB** |
| **Headroom**             | **9 GB**  |

Sufficient headroom to run both models concurrently.

---

## 11. Security Considerations

- API should be accessible only from lobster (firewall / bind to internal IP)
- No authentication needed if network-isolated; add API key if exposed
- Uploaded files are processed then deleted from /data/tmp/
- No secrets stored in code; use environment variables

---

## 12. Future Considerations (Not In Scope)

- Local LLM for analysis (can leverage remaining 9GB VRAM headroom)
- Web dashboard for browsing results
- Multi-GPU support
- Webhook callbacks when tasks complete
