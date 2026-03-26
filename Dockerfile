# Single-stage: build + run in one image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps: Python 3.11 + build tools (build tools kept for potential CUDA extensions in rfdetr)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    build-essential ninja-build curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# PyTorch (CUDA 12.1 wheels)
RUN python3.11 -m pip install -U pip setuptools wheel && \
    python3.11 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.1.0+cu121 torchvision==0.16.0+cu121

# App deps
RUN python3.11 -m pip install --no-cache-dir \
      "numpy>=1.24.0,<2.0.0" \
      "opencv-python-headless>=4.8.0" \
      "pillow>=9.5.0" \
      "matplotlib>=3.7.0" \
      fastapi uvicorn[standard] python-multipart psutil \
      rfdetr

# --- App files ---
COPY dinoAPI.py ./
COPY ai_dev/ ./ai_dev/

RUN mkdir -p uploads outputs

EXPOSE 8080
ENV HOST=0.0.0.0 PORT=8080 PYTHONUNBUFFERED=1
ENV WEIGHTS_PATH=/app/ai_dev/checkpoint_best_ema.pth
ENV THRESHOLDS_PATH=/app/ai_dev/per_class_thresholds_CL4-Martin.json

CMD ["python3.11", "dinoAPI.py"]
