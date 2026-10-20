FROM python:3.13-slim

# Ban torch: cu126 (GPU NVIDIA, mac dinh) hoac cpu (image nho hon nhieu, khong can GPU):
#   docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Model BGE-M3 + reranker tai ve day (mount volume de khong tai lai moi lan tao container)
    HF_HOME=/data/hf \
    GROQ_CACHE_DIR=/data/groq_cache

WORKDIR /app

RUN useradd --system --create-home app \
    && mkdir -p /data/hf /data/groq_cache \
    && chown -R app:app /data

COPY requirements.txt constraints.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.14.0 --index-url ${TORCH_INDEX_URL} \
    && pip install -r requirements.txt -c constraints.txt

# torch tren Linux chay mot so phep (vd bmm trong reranker) bang kernel triton; lan dau triton phai bien dich
# module driver bang C compiler -> thieu gcc se loi "Failed to find C compiler" (Windows khong co triton nen khong gap)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# .env KHONG duoc copy vao image (xem .dockerignore): config.py goi load_dotenv(override=True),
# neu co .env trong image se de len bien moi truong cua compose (vd QDRANT_URL).
COPY --chown=app:app . .

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# 1 worker: model nam trong bo nho/GPU cua tien trinh, nhieu worker se nap model nhieu lan
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
