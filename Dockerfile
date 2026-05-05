# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY acon-main ./acon-main
COPY src ./src
COPY configs ./configs
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Editable installs: react.py resolves PROJECT_ROOT via __file__.parents[3],
# which must land at /app so /app/acon-main/experiments/.../prompts is reachable.
RUN pip install --no-cache-dir -e ./acon-main \
 && pip install --no-cache-dir -e .

RUN mkdir -p /input /output /tmp/runs

ENTRYPOINT ["/app/entrypoint.sh"]
