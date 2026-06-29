# CPU image for the OpenAI Privacy Filter (opf) token-classification model.
# The model weights are NOT baked into the image — they download from
# HuggingFace on first boot and are cached on the hf-cache volume.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    https://github.com/openai/privacy-filter/archive/refs/heads/main.tar.gz \
    fastapi \
    "uvicorn[standard]"

COPY server.py /app/server.py

WORKDIR /app

ENV HF_HOME=/hf-cache \
    OPF_CHECKPOINT=/hf-cache/privacy-filter \
    OPF_DEVICE=cpu

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
