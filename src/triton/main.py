from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from triton.api import tasks, documents, schedules, health

app = FastAPI(title="Triton", description="Media-to-text GPU worker for OpenClaw")

# --- CORS middleware (allow Kairos frontend dev/preview origins) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Kairos Vite dev server
        "http://localhost:4173",  # Kairos Vite preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
app.include_router(documents.router, prefix="/documents", tags=["documents"])
app.include_router(schedules.router, prefix="/schedules", tags=["schedules"])
app.include_router(health.router, tags=["health"])
