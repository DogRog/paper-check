# syntax=docker/dockerfile:1.7

# --- Base image with uv and Python 3.13 ---
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

ENV UV_SYSTEM_PYTHON=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Resolve and download dependencies first for better caching ---
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# --- Copy project and install in venv ---
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Expose FastAPI default port
EXPOSE 8000

# Healthcheck hitting the FastAPI /health endpoint using Python stdlib
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import sys,urllib.request;\
resp=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3);\
sys.exit(0 if getattr(resp,'status',200)==200 else 1)" 

# Run with uvicorn; server:app is a thin wrapper that imports backend.app
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
