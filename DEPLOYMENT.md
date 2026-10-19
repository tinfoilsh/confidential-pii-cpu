# Deployment

Release with the `Tinfoil Release` workflow, which writes the measured image digest into `tinfoil-config.yml`. Provision `USAGE_REPORTER_SECRET` before deploying; startup fails without it.

## Limits

Requests are capped at 4 MiB and inference times out after two minutes. Inference cannot be interrupted once started, so Python rejects inputs over `OPF_MAX_INPUT_TOKENS` with 413 and returns 503 once `OPF_MAX_QUEUE_DEPTH` requests are running or waiting. Both are set in [tinfoil-config.yml](tinfoil-config.yml). On shutdown the front end drains for up to 30 seconds, then flushes queued usage events.

## Health and restarts

`GET /health` returns `{"status": "ok"}` once the model is loaded and 503 while the request queue is full. The Go front end probes it every 30 seconds and exits after three consecutive failures, so `restart: always` replaces a wedged inference process. Failures before the first successful probe are ignored so model loading never counts.

## Usage reporting

Each HTTP 200 from `/redact` queues one `pii-filter:redact` usage event for the authenticated caller and returns `X-Tinfoil-Billable-Requests: 1`, which websearch and the router use as the billing receipt. The header acknowledges a queued event, not settlement: failed deliveries are dropped, so monitor `pii_usage_failed_total` and `pii_usage_dropped_total`. Delivery needs egress to `api.tinfoil.sh`.
