# CPU image for the OpenAI Privacy Filter (opf) token-classification model.
# The model weights are NOT baked into the image — they download from
# HuggingFace on first boot and are cached on the hf-cache volume.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (from the PyTorch CPU index, not PyPI which serves the
# 2GB+ CUDA wheel by default on Linux). opf declares torch as a dependency;
# pre-installing it here means pip install -r requirements.txt skips it.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY server.py /app/server.py

WORKDIR /app

ENV HF_HOME=/hf-cache \
    OPF_CHECKPOINT=/hf-cache/privacy-filter \
    OPF_DEVICE=cpu

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
