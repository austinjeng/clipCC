FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

ARG TORCH_VARIANT=cpu
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ app/

ARG MODEL_NAME=ViT-L-14
ARG PRETRAINED=laion2b_s32b_b82k

ENV CLIP_CACHE_DIR=/app/models

RUN python -c "import json, open_clip; \
open_clip.create_model_and_transforms('${MODEL_NAME}', pretrained='${PRETRAINED}', cache_dir='/app/models'); \
json.dump({'model_name': '${MODEL_NAME}', 'pretrained': '${PRETRAINED}', 'cache_dir': '/app/models'}, open('/app/.baked_model', 'w'))"

EXPOSE 8000

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
