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
import logging
import os
import sys
import time
from array import array
from contextlib import asynccontextmanager
from typing import Sequence

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

from chunking import kept_outputs, plan_windows

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
# A window may wait in the gateway queue for up to the request ceiling.
FORWARD_TIMEOUT_SECONDS = float(os.environ.get("OPF_MAX_REQUEST_SECONDS", "900"))
# How many windows one request keeps in flight to the gateway at once. The
# gateway queue does the real scheduling; this just keeps a 250-window
# request from opening 250 sockets. Default: twice the forward lanes.
FORWARD_FANOUT = int(
    os.environ.get("OPF_FORWARD_FANOUT")
    or 2 * int(os.environ.get("OPF_WORKERS", "1")) * int(os.environ.get("OPF_MAX_CONCURRENCY", "1"))
)
KEY_HASH_HEADER = "X-Tinfoil-Key-Hash"
# Input token count returned to the gateway on every /redact response; the
# gateway meters it on the usage event and strips it from the client response.
INPUT_TOKENS_HEADER = "X-Tinfoil-Input-Tokens"
# /forward speaks raw native little-endian int32 / float32; the shape of the
# returned score matrix travels in this header as "rows,classes".
SHAPE_HEADER = "X-Tinfoil-Shape"
OCTET_STREAM = "application/octet-stream"
if sys.byteorder != "little" or array("i").itemsize != 4:
    raise RuntimeError("forward protocol assumes little-endian 4-byte ints")

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
INPUT_TOKENS = Counter(
    "pii_filter_input_tokens_total",
    "Total input tokens processed by the redact endpoint",
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
FORWARD_FAILURES = Counter(
    "pii_filter_forward_failures_total",
    "Windows this coordinator could not get scored, by gateway status",
    ["status"],
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
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
        )
    MODEL_LOADED.set(1)
    log.info(
        "Model loaded on %s (max_concurrency=%d, window=%d, overlap=%d, forward=%s, fanout=%d)",
        DEVICE,
        max_concurrency,
        WINDOW_TOKENS,
        WINDOW_OVERLAP,
        FORWARD_URL or "in-process",
        FORWARD_FANOUT,
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


def forward_tokens(tokens: Sequence[int]) -> tuple[bytes, int]:
    """One model forward: the window's per-token class log-probabilities as
    raw float32 bytes (row-major, native little-endian) plus the class count."""
    from opf._core.runtime import forward_window

    scores = forward_window(_runtime, tokens).contiguous()
    return scores.numpy().tobytes(), int(scores.shape[1])


def finish(text: str, tokens: list[int], scores: bytes, classes: int) -> dict:
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
        stacked = torch.frombuffer(bytearray(scores), dtype=torch.float32).reshape(
            len(tokens), classes
        )
        prediction = decode_token_scores(
            _runtime, text, tokens, range(len(tokens)), stacked, decoder=_decoder
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


def scores_view(payload: bytes, rows: int, classes: int) -> memoryview:
    """View float32 score bytes as a `[rows, classes]` memoryview, which
    slices by row without copying or decoding."""
    if classes <= 0 or len(payload) != rows * classes * 4:
        raise ValueError(f"{len(payload)} score bytes for {rows}x{classes} float32")
    return memoryview(payload).cast("f", shape=(rows, classes))


async def run_window(tokens: list[int], key_hash: str, fanout: asyncio.Semaphore) -> memoryview:
    """Score one window: on this process, or via the gateway's fair queue."""
    start = time.time()
    try:
        if FORWARD_URL is None:
            async with _semaphore:
                payload, classes = await run_in_threadpool(forward_tokens, tokens)
            return scores_view(payload, len(tokens), classes)
        try:
            async with fanout:
                response = await _forward_client.post(
                    FORWARD_URL,
                    content=array("i", tokens).tobytes(),
                    headers={"Content-Type": OCTET_STREAM, KEY_HASH_HEADER: key_hash},
                )
        except httpx.HTTPError as exc:
            FORWARD_FAILURES.labels(status="transport").inc()
            raise HTTPException(status_code=502, detail="forward dispatch failed") from exc
        if response.status_code != 200:
            FORWARD_FAILURES.labels(status=str(response.status_code)).inc()
            if response.status_code == 503:
                raise HTTPException(status_code=503, detail="privacy filter is at capacity")
            # Includes 504: a forward exceeded the gateway's per-window
            # deadline. The whole request fails; nothing partial is returned.
            raise HTTPException(
                status_code=502, detail=f"forward failed with status {response.status_code}"
            )
        try:
            rows, classes = (int(v) for v in response.headers[SHAPE_HEADER].split(","))
            if rows != len(tokens):
                raise ValueError(f"{rows} rows for {len(tokens)} tokens")
            return scores_view(response.content, rows, classes)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"malformed forward response: {exc}") from exc
    finally:
        WINDOW_ROUNDTRIP.observe(time.time() - start)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/forward")
async def forward(request: Request):
    """One model forward over one window.

    Request body: the window's token ids as native little-endian int32.
    Response body: `[tokens, classes]` float32 log-probabilities, row-major,
    with the shape in X-Tinfoil-Shape as `rows,classes`. Binary rather than
    JSON because a window's scores are ~1024 x classes floats, which JSON
    would spend tens of milliseconds encoding and decoding per window.
    """
    body = await request.body()
    if not body or len(body) % 4:
        raise HTTPException(status_code=422, detail="tokens must be non-empty int32 bytes")
    tokens = array("i")
    tokens.frombytes(body)
    async with _semaphore:
        FORWARDS_RUNNING.inc()
        start = time.time()
        try:
            payload, classes = await run_in_threadpool(forward_tokens, tokens)
        finally:
            FORWARD_DURATION.observe(time.time() - start)
            FORWARDS_RUNNING.dec()
    return Response(
        content=payload,
        media_type=OCTET_STREAM,
        headers={SHAPE_HEADER: f"{len(tokens)},{classes}"},
    )


@app.post("/redact", response_model=RedactResponse)
async def redact(req: RedactRequest, request: Request, response: Response):
    REQUESTS_RUNNING.inc()
    start = time.time()
    status = "success"
    try:
        tokens = [int(tok) for tok in _encoding.encode(req.text, allowed_special="all")]
        plans = plan_windows(len(tokens), WINDOW_TOKENS, WINDOW_OVERLAP)
        WINDOWS_TOTAL.inc(len(plans))
        key_hash = request.headers.get(KEY_HASH_HEADER, "")
        fanout = asyncio.Semaphore(FORWARD_FANOUT)
        try:
            # A TaskGroup cancels the remaining windows when one fails, so a
            # refused window does not leave the rest burning worker time.
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(run_window(tokens[plan.start : plan.end], key_hash, fanout))
                    for plan in plans
                ]
        except* HTTPException as failures:
            raise failures.exceptions[0]
        outputs = [task.result() for task in tasks]
        classes = outputs[0].shape[1] if outputs else 0
        if any(out.shape[1] != classes for out in outputs):
            raise HTTPException(status_code=502, detail="windows disagree on class count")
        scores = b"".join(part.tobytes() for part in kept_outputs(plans, outputs, len(tokens)))
        result_dict = await run_in_threadpool(finish, req.text, tokens, scores, classes)
        for span in result_dict.get("detected_spans", []):
            SPANS_DETECTED.labels(label=span.get("label", "unknown")).inc()
        CHARS_PROCESSED.inc(len(req.text))
        INPUT_TOKENS.inc(len(tokens))
        response.headers[INPUT_TOKENS_HEADER] = str(len(tokens))
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
