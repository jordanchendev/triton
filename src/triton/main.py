from fastapi import FastAPI

from triton.api import tasks, documents, schedules, health

app = FastAPI(title="Triton", description="Media-to-text GPU worker for OpenClaw")

app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(schedules.router, prefix="/schedules", tags=["schedules"])
app.include_router(health.router, tags=["health"])
