"""OpenAI Privacy Filter inference server.

Exposes POST /redact for PII span detection and redaction using the
openai/privacy-filter token-classification model. The model downloads
from HuggingFace on first boot and is cached on a persistent volume.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger("privacy-filter")
logging.basicConfig(level=logging.INFO)

MODEL_ID = "openai/privacy-filter"
CHECKPOINT_DIR = os.environ.get("OPF_CHECKPOINT", "/hf-cache/privacy-filter")
DEVICE = os.environ.get("OPF_DEVICE", "cpu")

_opf = None


def load_model():
    global _opf
    from huggingface_hub import snapshot_download
    from opf import OPF

    if not os.path.isdir(CHECKPOINT_DIR) or not os.listdir(CHECKPOINT_DIR):
        log.info("Downloading %s to %s ...", MODEL_ID, CHECKPOINT_DIR)
        # Skip onnx/ (~12.7 GB) and original/ (~2.8 GB duplicate) — the OPF
        # runtime only needs config.json + *.safetensors + tokenizer + viterbi.
        snapshot_download(
            MODEL_ID,
            local_dir=CHECKPOINT_DIR,
            ignore_patterns=["onnx/*", "original/*"],
        )
    else:
        log.info("Loading model from %s", CHECKPOINT_DIR)

    _opf = OPF(model=CHECKPOINT_DIR, device=DEVICE)
    log.info("Model loaded on %s", DEVICE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Privacy Filter", lifespan=lifespan)


class RedactRequest(BaseModel):
    text: str


class Span(BaseModel):
    label: str
    start: int
    end: int
    text: str
    placeholder: str


class RedactResponse(BaseModel):
    schema_version: int
    summary: dict
    text: str
    detected_spans: list[Span]
    redacted_text: str
    warning: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/redact", response_model=RedactResponse)
def redact(req: RedactRequest):
    result = _opf.redact(req.text)
    return result.to_dict()
