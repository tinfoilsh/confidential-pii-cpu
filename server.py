"""OpenAI Privacy Filter inference server.

Exposes POST /redact for PII span detection and redaction using the
openai/privacy-filter token-classification model. The model is mounted
as a verified modelwrap (MWP) read-only filesystem at boot — no
HuggingFace download or egress required.

Each process plays two roles:

- coordinator: /redact tokenizes the input, cuts it into overlapping
  windows (chunking.py) and runs the windows concurrently — through the
  gateway's /forward when OPF_FORWARD_URL is set, so every worker's forward
  lane can take part, or in-process otherwise — then stitches the per-token
  scores together and runs the unchanged global decode.
- forward lane: /forward runs one model forward over one window, guarded
  by the OPF_MAX_CONCURRENCY semaphore. This is the only CPU-heavy path.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from chunking import assemble, plan_windows

log = logging.getLogger("privacy-filter")
logging.basicConfig(level=logging.INFO)

CHECKPOINT_DIR = os.environ.get("OPF_CHECKPOINT", "/tinfoil/models/privacy-filter/original")
DEVICE = os.environ.get("OPF_DEVICE", "cpu")
# Windows of WINDOW_TOKENS tokens overlapping by 2 x WINDOW_OVERLAP; the
# overlap must be at least the model's attention band (128) so every kept
# token sees the same neighbours it would in one long forward pass.
WINDOW_TOKENS = int(os.environ.get("OPF_WINDOW_TOKENS", "1024"))
WINDOW_OVERLAP = int(os.environ.get("OPF_WINDOW_OVERLAP", "128"))
# The gateway's forward dispatcher. Unset means run windows in this process.
FORWARD_URL = os.environ.get("OPF_FORWARD_URL") or None
FORWARD_TIMEOUT_SECONDS = float(os.environ.get("OPF_FORWARD_TIMEOUT", "120"))
KEY_HASH_HEADER = "X-Tinfoil-Key-Hash"

_opf = None
_runtime = None
_decoder = None
_semaphore = None
_encoding = None
_forward_client = None

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
WINDOWS_TOTAL = Counter(
    "pii_filter_windows_total",
    "Total model windows dispatched by redact requests coordinated here",
)
WINDOW_ROUNDTRIP = Histogram(
    "pii_filter_window_roundtrip_seconds",
    "Time from dispatching a window to receiving its scores, queueing included",
)
FORWARD_DURATION = Histogram(
    "pii_filter_forward_duration_seconds",
    "Model forward time for one window on this process",
)
FORWARDS_RUNNING = Gauge(
    "pii_filter_forwards_running",
    "Number of model forwards currently running on this process",
)
MODEL_LOADED = Gauge(
    "pii_filter_model_loaded",
    "1 if the model is loaded and ready, 0 otherwise",
)


def load_model():
    global _opf, _runtime, _decoder, _semaphore, _encoding, _forward_client
    import torch

    n_threads = int(os.environ.get("OPF_NUM_THREADS", str(os.cpu_count() or 1)))
    torch.set_num_threads(n_threads)

    from opf import OPF

    plan_windows(0, WINDOW_TOKENS, WINDOW_OVERLAP)  # validate the geometry early
    log.info("Loading model from %s (threads=%d)", CHECKPOINT_DIR, n_threads)
    _opf = OPF(model=CHECKPOINT_DIR, device=DEVICE)
    # Force eager weight loading — OPF() is lazy, so run a dummy redaction
    # to load tensors into memory before serving requests.
    _opf.redact("warmup")
    _runtime, _decoder = _opf.get_prediction_components()
    # The checkpoint ships mixed bf16/fp32 tensors; on CPU every forward pass
    # then pays thousands of per-layer dtype-conversion copies (profiled at
    # ~56% of inference time). Converting the whole model once at load makes
    # every pass single-dtype. Opt out with OPF_COMPUTE_DTYPE=checkpoint.
    compute_dtype = os.environ.get("OPF_COMPUTE_DTYPE", "float32").lower()
    if DEVICE == "cpu" and compute_dtype == "float32":
        start = time.time()
        _runtime.model.float()
        log.info("Converted model to float32 in %.1fs", time.time() - start)
    elif compute_dtype not in ("checkpoint", "float32"):
        raise ValueError(f"Unsupported OPF_COMPUTE_DTYPE: {compute_dtype!r}")
    _encoding = _runtime.encoding
    max_concurrency = int(os.environ.get("OPF_MAX_CONCURRENCY", "1"))
    _semaphore = asyncio.Semaphore(max_concurrency)
    if FORWARD_URL is not None:
        _forward_client = httpx.AsyncClient(
            timeout=FORWARD_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )
    MODEL_LOADED.set(1)
    log.info(
        "Model loaded on %s (max_concurrency=%d, window=%d, overlap=%d, forward=%s)",
        DEVICE,
        max_concurrency,
        WINDOW_TOKENS,
        WINDOW_OVERLAP,
        FORWARD_URL or "in-process",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    if _forward_client is not None:
        await _forward_client.aclose()


app = FastAPI(title="Privacy Filter", lifespan=lifespan)


class RedactRequest(BaseModel):
    text: str


class ForwardRequest(BaseModel):
    tokens: list[int]


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


def forward_tokens(tokens: list[int]) -> list[list[float]]:
    """One model forward: per-token class log-probabilities for one window."""
    from opf._core.runtime import forward_window

    return forward_window(_runtime, tokens).tolist()


def finish(text: str, tokens: list[int], rows: list[list[float]]) -> dict:
    """Decode stitched per-token scores into the redaction result, exactly as
    OPF.redact does after its own window loop."""
    import torch
    from opf._api import RedactionResult, _redact_text, _warning_for_prediction
    from opf._common.constants import SCHEMA_VERSION
    from opf._core.runtime import (
        PredictionResult,
        build_detection_summary,
        decode_token_scores,
    )

    if tokens:
        scores = torch.tensor(rows, dtype=torch.float32)
        prediction = decode_token_scores(
            _runtime, text, tokens, range(len(tokens)), scores, decoder=_decoder
        )
    else:
        prediction = PredictionResult(text=text, spans=(), decoded_mismatch=False)
    redacted_text = _redact_text(prediction.text, prediction.spans)
    summary = build_detection_summary(
        output_mode=_runtime.output_mode,
        labels=[span.label for span in prediction.spans],
        decoded_mismatch=prediction.decoded_mismatch,
    )
    return RedactionResult(
        schema_version=SCHEMA_VERSION,
        summary=summary,
        text=prediction.text,
        detected_spans=tuple(prediction.spans),
        redacted_text=redacted_text,
        warning=_warning_for_prediction(prediction),
    ).to_dict()


async def run_window(tokens: list[int], key_hash: str) -> list[list[float]]:
    """Score one window: on this process, or via the gateway's fair queue."""
    start = time.time()
    try:
        if FORWARD_URL is None:
            async with _semaphore:
                return await run_in_threadpool(forward_tokens, tokens)
        try:
            response = await _forward_client.post(
                FORWARD_URL,
                content=json.dumps({"tokens": tokens}),
                headers={"Content-Type": "application/json", KEY_HASH_HEADER: key_hash},
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="forward dispatch failed") from exc
        if response.status_code == 503:
            raise HTTPException(status_code=503, detail="privacy filter is at capacity")
        if response.status_code != 200:
            raise HTTPException(
                status_code=502, detail=f"forward failed with status {response.status_code}"
            )
        return response.json()["logprobs"]
    finally:
        WINDOW_ROUNDTRIP.observe(time.time() - start)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/forward")
async def forward(req: ForwardRequest):
    if not req.tokens:
        raise HTTPException(status_code=422, detail="tokens must not be empty")
    async with _semaphore:
        FORWARDS_RUNNING.inc()
        start = time.time()
        try:
            rows = await run_in_threadpool(forward_tokens, req.tokens)
        finally:
            FORWARD_DURATION.observe(time.time() - start)
            FORWARDS_RUNNING.dec()
    # json.dumps is markedly faster than pydantic serialization for a
    # window's worth of floats, and repr round-trips float32 values exactly.
    return Response(content=json.dumps({"logprobs": rows}), media_type="application/json")


@app.post("/redact", response_model=RedactResponse)
async def redact(req: RedactRequest, request: Request):
    REQUESTS_RUNNING.inc()
    start = time.time()
    status = "success"
    try:
        tokens = [int(tok) for tok in _encoding.encode(req.text, allowed_special="all")]
        plans = plan_windows(len(tokens), WINDOW_TOKENS, WINDOW_OVERLAP)
        WINDOWS_TOTAL.inc(len(plans))
        key_hash = request.headers.get(KEY_HASH_HEADER, "")
        try:
            # A TaskGroup cancels the remaining windows when one fails, so a
            # refused window does not leave the rest burning worker time.
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(run_window(tokens[plan.start : plan.end], key_hash))
                    for plan in plans
                ]
        except* HTTPException as failures:
            raise failures.exceptions[0]
        rows = assemble(plans, [task.result() for task in tasks], len(tokens))
        result_dict = await run_in_threadpool(finish, req.text, tokens, rows)
        for span in result_dict.get("detected_spans", []):
            SPANS_DETECTED.labels(label=span.get("label", "unknown")).inc()
        CHARS_PROCESSED.inc(len(req.text))
        return result_dict
    except HTTPException as exc:
        status = "overloaded" if exc.status_code == 503 else "error"
        raise
    except Exception:
        status = "error"
        raise
    finally:
        REQUESTS_TOTAL.labels(status=status).inc()
        REQUEST_DURATION.labels(status=status).observe(time.time() - start)
        REQUESTS_RUNNING.dec()
