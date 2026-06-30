# CPU image for the OpenAI Privacy Filter (opf) token-classification model.
# Model weights are mounted as a verified modelwrap (MWP) read-only
# filesystem at boot — no HuggingFace download or egress required.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (from the PyTorch CPU index, not PyPI which serves the
# 2GB+ CUDA wheel by default on Linux). opf declares torch as a dependency;
# pre-installing it here means pip install -r requirements.txt skips it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-populate tiktoken cache so o200k_base doesn't need network egress.
# The cache file is named by sha1(url) — deterministic, never changes.
COPY tiktoken_cache /app/tiktoken_cache

COPY server.py /app/server.py

WORKDIR /app

ENV OPF_DEVICE=cpu \
    TIKTOKEN_CACHE_DIR=/app/tiktoken_cache

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
