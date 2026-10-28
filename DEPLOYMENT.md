# Deployment

Release with the `Tinfoil Release` workflow, which writes the measured image digest into `tinfoil-config.yml`. Provision `USAGE_REPORTER_SECRET` before deploying; startup fails without it.

## Long inputs and the forward queue

A `/redact` request is proxied to one worker process, which tokenizes the text and cuts it into windows of `OPF_WINDOW_TOKENS` tokens (default 1024) that overlap by `2 × OPF_WINDOW_OVERLAP` (default 128 each side), so the effective stride is `OPF_WINDOW_TOKENS − 2 × OPF_WINDOW_OVERLAP`; an input that fits in one window is not masked at all. Only the tokens with `OPF_WINDOW_OVERLAP` tokens of context on both sides are kept from each window, except at the sequence ends, so every token is scored by exactly one window with its attention band intact; the stitched scores then go through upstream `opf`'s unchanged global Viterbi decode. The windows are posted concurrently to the gateway's internal `POST /forward` (raw little-endian int32 token ids in, row-major float32 log-probabilities out with the `rows,classes` shape in `X-Tinfoil-Shape`; binary because a window's scores are ~1024 × classes floats), which holds one FIFO queue per caller, round-robins across callers and runs each window on the worker with the fewest forwards in flight, with at most `OPF_WORKERS × OPF_MAX_CONCURRENCY` forwards running. A long document therefore uses every worker, and a short request from another caller waits for about one window, not for the whole document.

`/forward` is internal: the gateway only accepts it from loopback and the shim does not expose it. The Python worker's `/forward` listens on loopback only. The coordinating worker reaches the gateway through `OPF_FORWARD_URL`; when it is unset, windows run in-process (single-worker development).

## Limits

Requests are capped at 4 MiB (≈250 windows at the default geometry), and that is a normal request, not an overload. There is no token limit; load is bounded in windows and time:

- A `/redact` request may run for 30 s plus `OPF_WINDOW_TIMEOUT_SECONDS` (default 5, generous against the ~0.8 s measured per window) per window, estimated by the gateway from the body size (one token per four bytes), capped at `OPF_MAX_REQUEST_SECONDS` (default 900). The ceiling also bounds how long a window may wait in the forward queue and the server's read and idle timeouts.
- The timeouts above are size allowances, not wedge detectors. The wedge detector is per dispatch: each gateway→worker forward must finish within `OPF_FORWARD_TIMEOUT_SECONDS` (default 30, ~40× a window). A forward that does not returns 504 to the coordinator, which fails the whole `/redact` (no partial result, no billing event), increments `pii_forward_timeouts_total` on the gateway and `pii_filter_forward_failures_total{status="504"}` on the coordinator, and marks that worker unhealthy: `/health` then fails until the container is replaced.
- When more than `OPF_MAX_QUEUED_WINDOWS` windows (default 4096) are waiting for a forward lane, new windows get 503 and the request that owns them fails with 503, cancelling its remaining windows. A window that cannot start within the ceiling is also refused with 503.
- The coordinating worker keeps at most `OPF_FORWARD_FANOUT` windows (default 2 × `OPF_WORKERS` × `OPF_MAX_CONCURRENCY`) in flight to the gateway per request, enough to keep every lane busy while the gateway queue does the scheduling.

On shutdown the front end drains for up to 30 seconds, then flushes queued usage events.

## Usage reporting

Each HTTP 200 from `/redact` queues one `pii-filter:redact` usage event for the authenticated caller and returns `X-Tinfoil-Billable-Requests: 1`, which websearch and the router use as the billing receipt. The event carries an `input_tokens` meter with the request's tokenized length, which the coordinating worker returns on an internal `X-Tinfoil-Input-Tokens` response header; the gateway strips the header before the response reaches the client and refuses (502) a 200 that lacks it rather than return an unmetered result. Pricing the meter is a controlplane change; until then only the per-request charge applies. The header acknowledges a queued event, not settlement: failed deliveries are dropped, so monitor `pii_usage_failed_total` and `pii_usage_dropped_total`. Delivery needs egress to `api.tinfoil.sh`. The gateway's `/metrics` also reports `pii_forward_queued` and `pii_forward_inflight` alongside worker 0's model metrics (`pii_filter_input_tokens_total`, `pii_filter_windows_total`, `pii_filter_window_roundtrip_seconds`, `pii_filter_forward_duration_seconds`).
