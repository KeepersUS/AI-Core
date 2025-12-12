# Single-stage: build + run in one image (most reliable)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# System deps: Python 3.11 + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3.11-distutils \
    git build-essential ninja-build curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# PyTorch (CUDA 12.1 wheels) — using 2.1.0 which is the earliest available for cu121
RUN python3.11 -m pip install -U pip setuptools wheel && \
    python3.11 -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.1.0+cu121 torchvision==0.16.0+cu121

# App deps - pin numpy<2 for PyTorch 2.1.0 compatibility, and transformers compatible with torch 2.1
RUN python3.11 -m pip install --no-cache-dir \
      "numpy>=1.24.0,<2.0.0" \
      "transformers>=4.35.0,<4.50.0" \
      "opencv-python-headless>=4.8.0" \
      "pillow>=9.5.0" \
      "matplotlib>=3.7.0" \
      fastapi uvicorn[standard] python-multipart

# System metrics dependency for /metrics endpoint
RUN python3.11 -m pip install --no-cache-dir psutil>=5.9.0

# GroundingDINO from source - clone and build with proper CUDA support
ARG GD_REF=856dde20aee659246248e20734ef9ba5214f5e44
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO && \
    cd /tmp/GroundingDINO && \
    git checkout ${GD_REF} && \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX" python3.11 -m pip install --no-cache-dir --no-build-isolation . && \
    cd /app && \
    rm -rf /tmp/GroundingDINO

# Force numpy<2 after GroundingDINO (its deps try to upgrade to numpy 2.x)
RUN python3.11 -m pip install --no-cache-dir "numpy>=1.24.0,<2.0.0"

# --- Your app files ---
COPY dinoAPI.py grounding_dino.py model_tests.py COCO_CLASSES.py ./

# Pre-download BERT model to avoid HuggingFace rate limits at runtime
RUN python3.11 -c "from transformers import BertModel, BertTokenizer; \
    BertTokenizer.from_pretrained('bert-base-uncased'); \
    BertModel.from_pretrained('bert-base-uncased'); \
    print('BERT model cached successfully')"

RUN mkdir -p uploads outputs

# Sanity check: verify CUDA, Torch, and the _C extension
RUN python3.11 -c "import importlib, torch, sys; \
print('=' * 80); \
print('CUDA EXTENSIONS VERIFICATION'); \
print('=' * 80); \
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'CUDA avail:', torch.cuda.is_available()); \
m = importlib.import_module('groundingdino.models.GroundingDINO.ms_deform_attn'); \
print('GroundingDINO _C present:', hasattr(m, '_C'), 'at:', m.__file__); \
(print('✅ CUDA extensions compiled successfully!') if hasattr(m, '_C') else (print('❌ CUDA extensions NOT compiled - model will fail on GPU!'), sys.exit(1)))"

EXPOSE 8080
ENV HOST=0.0.0.0 PORT=8080 PYTHONUNBUFFERED=1

#HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/docs').getcode()" || exit 1

CMD ["python3.11", "dinoAPI.py"]  
