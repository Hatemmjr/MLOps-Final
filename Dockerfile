# ─── Bonus A — Dockerfile ────────────────────────────────────────────────────
# Pinned base image | non-root user | minimal layers
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11.9-slim AS base

# Prevent .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ──────────────────────────────────────────────────────
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# ── Install Python dependencies (as root, before switching user) ──────────────
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY configs/ configs/
COPY src/ src/
COPY monitoring/ monitoring/

# ── Switch to non-root user ───────────────────────────────────────────────────
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
