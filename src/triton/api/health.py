from fastapi import APIRouter

from triton.schemas import HealthResponse
from triton.services.queue import get_queue_length

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
    # Fallback: check via ctranslate2 (installed with faster-whisper)
    try:
        import ctranslate2
        if ctranslate2.get_supported_compute_types("cuda"):
            return True, None, None
    except Exception:
        pass
    return False, None, None


@router.get("/health", response_model=HealthResponse)
def health_check():
    gpu_available, gpu_used, gpu_total = _get_gpu_info()
    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        gpu_memory_used_mb=gpu_used,
        gpu_memory_total_mb=gpu_total,
        queue_gpu_length=get_queue_length("gpu"),
        queue_cpu_length=get_queue_length("cpu"),
    )
