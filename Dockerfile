FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini .
ARG INSTALL_ML=false
RUN pip install --no-cache-dir . && \
    if [ "$INSTALL_ML" = "true" ]; then \
        pip install --no-cache-dir faster-whisper paddlepaddle paddleocr; \
    fi
COPY scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh

ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["uvicorn", "triton.main:app", "--host", "0.0.0.0", "--port", "8000"]
