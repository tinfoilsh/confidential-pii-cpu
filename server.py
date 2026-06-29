"""OpenAI Privacy Filter inference server.

Exposes POST /redact for PII span detection and redaction using the
openai/privacy-filter token-classification model. The model is mounted
as a verified modelwrap (MWP) read-only filesystem at boot — no
HuggingFace download or egress required.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

log = logging.getLogger("privacy-filter")
logging.basicConfig(level=logging.INFO)

CHECKPOINT_DIR = os.environ.get("OPF_CHECKPOINT", "/tinfoil/mwp/privacy-filter")
DEVICE = os.environ.get("OPF_DEVICE", "cpu")

_opf = None


def load_model():
    global _opf
    from opf import OPF

    log.info("Loading model from %s", CHECKPOINT_DIR)
    _opf = OPF(model=CHECKPOINT_DIR, device=DEVICE)
    # Force eager weight loading — OPF() is lazy, so run a dummy redaction
    # to load tensors into memory before serving requests.
    _opf.redact("warmup")
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
