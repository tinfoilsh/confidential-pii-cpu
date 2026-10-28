# Deployment

Release with the `Tinfoil Release` workflow, which writes the measured image digest into `tinfoil-config.yml`. Provision `USAGE_REPORTER_SECRET` before deploying; startup fails without it.

## Long inputs and the forward queue

A `/redact` request is proxied to one worker process, which tokenizes the text and cuts it into windows of `OPF_WINDOW_TOKENS` tokens (default 1024) that overlap by `2 × OPF_WINDOW_OVERLAP` (default 128 each side). Only the tokens with `OPF_WINDOW_OVERLAP` tokens of context on both sides are kept from each window, except at the sequence ends, so every token is scored by exactly one window with its attention band intact; the stitched scores then go through upstream `opf`'s unchanged global Viterbi decode. The windows are posted concurrently to the gateway's internal `POST /forward`, which holds one FIFO queue per caller, round-robins across callers and runs each window on the worker with the fewest forwards in flight, with at most `OPF_WORKERS × OPF_MAX_CONCURRENCY` forwards running. A long document therefore uses every worker, and a short request from another caller waits for about one window, not for the whole document.

`/forward` is internal: the gateway only accepts it from loopback and the shim does not expose it. The Python worker's `/forward` listens on loopback only. The coordinating worker reaches the gateway through `OPF_FORWARD_URL`; when it is unset, windows run in-process (single-worker development).

## Limits

Requests are capped at 4 MiB and inference times out after two minutes. There is no token limit: load is bounded in windows instead. When more than `OPF_MAX_QUEUED_WINDOWS` windows (default 512, roughly 512 × 768 ≈ 390k tokens of pending work) are waiting for a forward lane, new windows get 503 and the request that owns them fails with 503, cancelling its remaining windows. A window that cannot start within the inference timeout is also refused with 503. On shutdown the front end drains for up to 30 seconds, then flushes queued usage events.

## Usage reporting

Each HTTP 200 from `/redact` queues one `pii-filter:redact` usage event for the authenticated caller and returns `X-Tinfoil-Billable-Requests: 1`, which websearch and the router use as the billing receipt. The event carries an `input_tokens` meter with the request's tokenized length, which the coordinating worker returns on an internal `X-Tinfoil-Input-Tokens` response header; the gateway strips the header before the response reaches the client and refuses (502) a 200 that lacks it rather than return an unmetered result. Pricing the meter is a controlplane change; until then only the per-request charge applies. The header acknowledges a queued event, not settlement: failed deliveries are dropped, so monitor `pii_usage_failed_total` and `pii_usage_dropped_total`. Delivery needs egress to `api.tinfoil.sh`. The gateway's `/metrics` also reports `pii_forward_queued` and `pii_forward_inflight` alongside worker 0's model metrics (`pii_filter_input_tokens_total`, `pii_filter_windows_total`, `pii_filter_window_roundtrip_seconds`, `pii_filter_forward_duration_seconds`).
