"""OpenAI Privacy Filter inference server.

Exposes POST /redact for PII span detection and redaction using the
openai/privacy-filter token-classification model. The model is mounted
as a verified modelwrap (MWP) read-only filesystem at boot — no
HuggingFace download or egress required.
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

log = logging.getLogger("privacy-filter")
logging.basicConfig(level=logging.INFO)

CHECKPOINT_DIR = os.environ.get("OPF_CHECKPOINT", "/tinfoil/mpk/privacy-filter")
DEVICE = os.environ.get("OPF_DEVICE", "cpu")

_opf = None
_semaphore = None

REQUESTS_TOTAL = Counter(
    "pii_filter_requests_total",
    "Total redact requests processed",
    ["status"],
)
REQUEST_DURATION = Histogram(
    "pii_filter_request_duration_seconds",
    "Time spent processing a redact request",
    ["status"],
)
REQUESTS_RUNNING = Gauge(
    "pii_filter_requests_running",
    "Number of redact requests currently in flight",
)
SPANS_DETECTED = Counter(
    "pii_filter_spans_detected_total",
    "Total PII spans detected by label",
    ["label"],
)
CHARS_PROCESSED = Counter(
    "pii_filter_chars_processed_total",
    "Total characters processed by the redact endpoint",
)
MODEL_LOADED = Gauge(
    "pii_filter_model_loaded",
    "1 if the model is loaded and ready, 0 otherwise",
)


def load_model():
    global _opf, _semaphore
    import torch

    n_threads = int(os.environ.get("OPF_NUM_THREADS", str(os.cpu_count() or 1)))
    torch.set_num_threads(n_threads)

    from opf import OPF

    log.info("Loading model from %s (threads=%d)", CHECKPOINT_DIR, n_threads)
    _opf = OPF(model=CHECKPOINT_DIR, device=DEVICE)
    # Force eager weight loading — OPF() is lazy, so run a dummy redaction
    # to load tensors into memory before serving requests.
    _opf.redact("warmup")
    max_concurrency = int(os.environ.get("OPF_MAX_CONCURRENCY", "1"))
    _semaphore = asyncio.Semaphore(max_concurrency)
    MODEL_LOADED.set(1)
    log.info("Model loaded on %s (max_concurrency=%d)", DEVICE, max_concurrency)


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


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/redact", response_model=RedactResponse)
async def redact(req: RedactRequest):
    REQUESTS_RUNNING.inc()
    start = time.time()
    status = "success"
    try:
        async with _semaphore:
            result = await run_in_threadpool(_opf.redact, req.text)
        result_dict = result.to_dict()
        for span in result_dict.get("detected_spans", []):
            SPANS_DETECTED.labels(label=span.get("label", "unknown")).inc()
        CHARS_PROCESSED.inc(len(req.text))
        return result_dict
    except Exception:
        status = "error"
        raise
    finally:
        REQUESTS_TOTAL.labels(status=status).inc()
        REQUEST_DURATION.labels(status=status).observe(time.time() - start)
        REQUESTS_RUNNING.dec()
