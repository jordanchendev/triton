FROM python:3.11-slim

COPY --from=denoland/deno:bin-2.9.7 /deno /usr/local/bin/deno
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /usr/local/bin/uv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock .
COPY src/ src/
COPY alembic/ alembic/
COPY alembic.ini .
ARG INSTALL_ML=false
RUN extra=$([ "$INSTALL_ML" = "true" ] && echo "--extra cpu-ml"); \
    uv export --frozen --no-dev $extra --no-hashes --no-emit-project -o /tmp/req.txt && \
    pip install --no-cache-dir -r /tmp/req.txt && \
    pip install --no-cache-dir --no-deps .
COPY scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh

ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["uvicorn", "triton.main:app", "--host", "0.0.0.0", "--port", "8000"]
