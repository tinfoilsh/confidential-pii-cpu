"""OpenAI Privacy Filter inference server.

Exposes POST /redact for PII span detection and redaction using the
openai/privacy-filter token-classification model. The model is mounted
as a verified modelwrap (MWP) read-only filesystem at boot — no
HuggingFace download or egress required.

Each process plays two roles:

- coordinator: /chunk (where the gateway proxies a client's /redact)
  tokenizes the input, cuts it into overlapping windows (chunking.py) and
  posts every window at once — to the gateway's /queue when OPF_QUEUE_URL
  is set, which schedules them fairly across callers and every worker's
  lane, or in-process otherwise — then
  stitches the per-token scores together and runs the unchanged global
  decode. When the gateway ends the request (answered, timed out, client
  gone) it refuses the request's remaining windows and the coordinator
  stops.
- forward lane: /forward runs one model forward over one window at a time
  (one interpreter, one torch pool, one rotary offset). This is the only
  CPU-heavy path.
"""

import asyncio
import json
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

from chunking import check_geometry, kept_outputs, plan_windows, required_overlap

log = logging.getLogger("privacy-filter")
logging.basicConfig(level=logging.INFO)
# httpx logs one INFO line per request: that is one per window.
logging.getLogger("httpx").setLevel(logging.WARNING)

CHECKPOINT_DIR = os.environ.get(
    "OPF_CHECKPOINT", "/tinfoil/models/privacy-filter/original"
)
DEVICE = os.environ.get("OPF_DEVICE", "cpu")
# Windows of WINDOW_TOKENS tokens overlapping by 2 x WINDOW_OVERLAP. The
# overlap must cover the model's receptive field (layers x attention
# context, 1024 for the pinned checkpoint; enforced at load from the
# checkpoint config) so every kept token's scores equal a single long pass.
WINDOW_TOKENS = int(os.environ.get("OPF_WINDOW_TOKENS", "6144"))
WINDOW_OVERLAP = int(os.environ.get("OPF_WINDOW_OVERLAP", "1024"))
# Largest input accepted, in tokens; longer inputs are refused with 413 and
# the limit in the message. Chunking is exact at any length; the cap keeps
# absolute positions well inside the range the model was validated at
# (exact and clean to 50k in the lab; the position table runs to 131k).
MAX_INPUT_TOKENS = int(os.environ.get("OPF_MAX_INPUT_TOKENS", "80000"))
# The gateway's forward dispatcher. Unset means run windows in this process.
QUEUE_URL = os.environ.get("OPF_QUEUE_URL") or None
# A window may wait in the gateway queue for up to the request ceiling.
FORWARD_TIMEOUT_SECONDS = float(os.environ.get("OPF_MAX_REQUEST_SECONDS", "900"))
KEY_HASH_HEADER = "X-Tinfoil-Key-Hash"
# The gateway's id for the /redact this coordinator is serving; echoed on
# every /queue post so the gateway can drop the request's queue when it ends.
REQUEST_ID_HEADER = "X-Tinfoil-Request-ID"
# Input token count returned to the gateway on every /redact response; the
# gateway meters it on the usage event and strips it from the client response.
INPUT_TOKENS_HEADER = "X-Tinfoil-Input-Tokens"
# Total tokens of the owning request, sent on each /queue post so the gateway
# can prioritise short requests.
REQUEST_TOKENS_HEADER = "X-Tinfoil-Request-Tokens"
# /forward speaks raw native little-endian int32 / float32; the shape of the
# returned score matrix travels in this header as "rows,classes".
SHAPE_HEADER = "X-Tinfoil-Shape"
# A window's absolute token offset, sent with each window. The worker runs
# the window at its true positions so the rotary angles are bit-for-bit
# those of an unchunked pass; without it every window would restart at
# position 0 and differ from the single-pass result by float32 angle
# rounding (measured ~1e-2 in log-probability — the same size as prepending
# one token to the document, but not zero).
POSITION_OFFSET_HEADER = "X-Tinfoil-Position-Offset"
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
# A histogram: _sum is the total tokens processed, _bucket the traffic shape.
INPUT_TOKENS = Histogram(
    "pii_filter_input_tokens",
    "Input tokens per redact request",
    buckets=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
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

    with open(os.path.join(CHECKPOINT_DIR, "config.json"), encoding="utf-8") as handle:
        checkpoint_config = json.load(handle)
    required = required_overlap(checkpoint_config)
    check_geometry(WINDOW_TOKENS, WINDOW_OVERLAP, required)
    log.info(
        "Window geometry %d/%d (receptive field %d)",
        WINDOW_TOKENS,
        WINDOW_OVERLAP,
        required,
    )
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
    _init_rotary_offsets(_runtime.model, int(checkpoint_config["num_hidden_layers"]))
    for retired in ("OPF_MAX_CONCURRENCY", "OPF_FORWARD_FANOUT", "OPF_FORWARD_URL"):
        if os.environ.get(retired):
            raise ValueError(f"{retired} is no longer a setting (one forward at a time per worker; windows go to OPF_QUEUE_URL)")
    _semaphore = asyncio.Semaphore(1)
    if QUEUE_URL is not None:
        _forward_client = httpx.AsyncClient(
            timeout=FORWARD_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=64),
        )
    MODEL_LOADED.set(1)
    log.info(
        "Model loaded on %s (window=%d, overlap=%d, forward=%s)",
        DEVICE,
        WINDOW_TOKENS,
        WINDOW_OVERLAP,
        QUEUE_URL or "in-process",
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


_rotary_modules: list = []


def _init_rotary_offsets(model, layers: int) -> None:
    """Find every rotary module so a window can be run at an absolute
    position offset (see POSITION_OFFSET_HEADER), and keep each module's
    original tables to restore between forwards."""
    global _rotary_modules
    _rotary_modules = [
        m for m in model.modules() if type(m).__name__ == "RotaryEmbedding"
    ]
    if len(_rotary_modules) != layers:
        # Every attention layer owns one; a miss would silently run windows
        # at position 0 and break exactness.
        raise RuntimeError(
            f"found {len(_rotary_modules)} RotaryEmbedding modules, expected {layers}"
        )
    for rope in _rotary_modules:
        rope._tinfoil_cos_default = rope.cos_cache
        rope._tinfoil_sin_default = rope.sin_cache
    log.info("Rotary position offsets enabled on %d module(s)", len(_rotary_modules))


def _rotary_rows(rope, offset: int, length: int):
    """cos/sin rows for absolute positions [offset, offset+length): the same
    elementwise t·inv_freq the module's own table holds, computed for just
    these rows so nothing grows with the largest position ever seen."""
    import torch

    concentration, inv_freq = rope._compute_concentration_and_inv_freq(
        device=rope._tinfoil_cos_default.device
    )
    t = torch.arange(offset, offset + length, dtype=torch.float32, device=inv_freq.device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return (freqs.cos() * concentration).to(rope.dtype), (freqs.sin() * concentration).to(rope.dtype)


def _set_rotary_offset(offset: int, length: int) -> None:
    """Point the rotary caches at rows for [offset, offset+length). opf
    slices `cos_cache[:num_tokens]`, so a table whose row 0 is position
    `offset` yields the true absolute positions. One forward runs at a time
    per process, so the module state is not shared with another forward."""
    for rope in _rotary_modules:
        if offset == 0 and length <= rope._tinfoil_cos_default.shape[0]:
            rope.cos_cache = rope._tinfoil_cos_default
            rope.sin_cache = rope._tinfoil_sin_default
        else:
            rope.cos_cache, rope.sin_cache = _rotary_rows(rope, offset, length)


def forward_tokens(
    tokens: Sequence[int], position_offset: int = 0
) -> tuple[bytes, int]:
    """One model forward: the window's per-token class log-probabilities as
    raw float32 bytes (row-major, native little-endian) plus the class count.
    `position_offset` is the window's absolute token index in its request."""
    from opf._core.runtime import forward_window

    _set_rotary_offset(position_offset, len(tokens))
    try:
        scores = forward_window(_runtime, tokens).contiguous()
    finally:
        _set_rotary_offset(0, 0)
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


async def run_window(
    tokens: list[int],
    key_hash: str,
    request_id: str,
    request_tokens: int,
    position_offset: int = 0,
) -> memoryview:
    """Score one window: on this process, or via the gateway's fair queue."""
    start = time.time()
    try:
        if QUEUE_URL is None:
            async with _semaphore:
                payload, classes = await run_in_threadpool(
                    forward_tokens, tokens, position_offset
                )
            return scores_view(payload, len(tokens), classes)
        # A transport failure, a malformed reply (missing shape header, short
        # body) or a wedged worker (504) is retried once: the window is
        # idempotent and cheap next to failing a whole multi-window request.
        # 503 (capacity) and 409 (request ended) are not retried.
        last_error = None
        for attempt in range(2):
            try:
                response = await _forward_client.post(
                    QUEUE_URL,
                    content=array("i", tokens).tobytes(),
                    headers={
                        "Content-Type": OCTET_STREAM,
                        KEY_HASH_HEADER: key_hash,
                        REQUEST_ID_HEADER: request_id,
                        REQUEST_TOKENS_HEADER: str(request_tokens),
                        POSITION_OFFSET_HEADER: str(position_offset),
                    },
                )
            except httpx.HTTPError as exc:
                FORWARD_FAILURES.labels(status="transport").inc()
                last_error = f"forward dispatch failed: {type(exc).__name__}: {exc}"
                log.warning(
                    "window @%d (%d tokens) attempt %d: %s",
                    position_offset,
                    len(tokens),
                    attempt + 1,
                    last_error,
                )
                continue
            if response.status_code != 200:
                FORWARD_FAILURES.labels(status=str(response.status_code)).inc()
                if response.status_code == 503:
                    raise HTTPException(
                        status_code=503, detail="privacy filter is at capacity"
                    )
                if response.status_code == 409:
                    # The gateway ended this request (client gone, deadline);
                    # fail fast so the task group cancels the rest.
                    raise HTTPException(status_code=409, detail="request no longer active")
                if response.status_code == 504:
                    # The worker that took this window wedged; the gateway is
                    # replacing it and routes a re-queued window elsewhere.
                    last_error = "forward timed out (worker being replaced)"
                    log.warning("window @%d (%d tokens) attempt %d: %s", position_offset, len(tokens), attempt + 1, last_error)
                    continue
                raise HTTPException(
                    status_code=502,
                    detail=f"forward failed with status {response.status_code}",
                )
            try:
                rows, classes = (
                    int(v) for v in response.headers[SHAPE_HEADER].split(",")
                )
                if rows != len(tokens):
                    raise ValueError(f"{rows} rows for {len(tokens)} tokens")
                return scores_view(response.content, rows, classes)
            except (KeyError, ValueError, TypeError) as exc:
                FORWARD_FAILURES.labels(status="malformed").inc()
                last_error = (
                    f"malformed forward response: {type(exc).__name__}: {exc} "
                    f"(status {response.status_code}, {len(response.content)} bytes, "
                    f"shape={response.headers.get(SHAPE_HEADER)!r})"
                )
                log.warning(
                    "window @%d (%d tokens) attempt %d: %s",
                    position_offset,
                    len(tokens),
                    attempt + 1,
                    last_error,
                )
        FORWARD_FAILURES.labels(status="gave_up").inc()
        raise HTTPException(status_code=502, detail=last_error or "forward failed")
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
        raise HTTPException(
            status_code=422, detail="tokens must be non-empty int32 bytes"
        )
    tokens = array("i")
    tokens.frombytes(body)
    try:
        position_offset = int(request.headers.get(POSITION_OFFSET_HEADER, "0"))
        if position_offset < 0:
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid position offset") from None
    async with _semaphore:
        FORWARDS_RUNNING.inc()
        start = time.time()
        try:
            payload, classes = await run_in_threadpool(
                forward_tokens, tokens, position_offset
            )
        finally:
            FORWARD_DURATION.observe(time.time() - start)
            FORWARDS_RUNNING.dec()
    return Response(
        content=payload,
        media_type=OCTET_STREAM,
        headers={SHAPE_HEADER: f"{len(tokens)},{classes}"},
    )


@app.post("/chunk", response_model=RedactResponse)
async def redact(req: RedactRequest, request: Request, response: Response):
    REQUESTS_RUNNING.inc()
    start = time.time()
    status = "success"
    try:
        tokens = [int(tok) for tok in _encoding.encode(req.text, allowed_special="all")]
        if len(tokens) > MAX_INPUT_TOKENS:
            raise HTTPException(
                status_code=413,
                detail=f"input is {len(tokens)} tokens; the limit is {MAX_INPUT_TOKENS}",
            )
        plans = plan_windows(len(tokens), WINDOW_TOKENS, WINDOW_OVERLAP)
        WINDOWS_TOTAL.inc(len(plans))
        key_hash = request.headers.get(KEY_HASH_HEADER, "")
        request_id = request.headers.get(REQUEST_ID_HEADER, "")
        try:
            # Every window goes to the gateway queue at once; it schedules
            # them. A TaskGroup cancels the remaining windows when one fails
            # (including the gateway refusing them because the request
            # ended), so nothing of a dead request keeps running here.
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(
                        run_window(
                            tokens[plan.start : plan.end],
                            key_hash,
                            request_id,
                            len(tokens),
                            position_offset=plan.start,
                        )
                    )
                    for plan in plans
                ]
        except* HTTPException as failures:
            raise failures.exceptions[0]
        outputs = [task.result() for task in tasks]
        classes = outputs[0].shape[1] if outputs else 0
        if any(out.shape[1] != classes for out in outputs):
            raise HTTPException(
                status_code=502, detail="windows disagree on class count"
            )
        scores = b"".join(
            part.tobytes() for part in kept_outputs(plans, outputs, len(tokens))
        )
        result_dict = await run_in_threadpool(finish, req.text, tokens, scores, classes)
        for span in result_dict.get("detected_spans", []):
            SPANS_DETECTED.labels(label=span.get("label", "unknown")).inc()
        CHARS_PROCESSED.inc(len(req.text))
        INPUT_TOKENS.observe(len(tokens))
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
