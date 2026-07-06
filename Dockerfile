FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

LABEL org.opencontainers.image.source="https://github.com/austinjeng/clipCC"
LABEL org.opencontainers.image.description="Video classification API using SigLIP2 models"

COPY requirements-prod.txt .

ARG TORCH_VARIANT=cpu
# Install torch + torchvision from the variant index (cpu/cu126). Both pins MUST
# match requirements-prod.txt AND come from the same index: variant wheels report
# e.g. 2.12.0+cpu / 0.27.0+cpu, so the next step treats them as already present
# and does not overwrite them with default-PyPI (CUDA) wheels.
RUN pip install --no-cache-dir torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install --no-cache-dir -r requirements-prod.txt

COPY app/ app/

ENV CLIP_CACHE_DIR=/app/models
RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
