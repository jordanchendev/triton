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
    worker_prefetch_multiplier=1,
    imports=["triton.workers.gpu_tasks", "triton.workers.cpu_tasks"],
)
