# Endpoint-owned privacy-filter billing

## Contract

- The shim authenticates `/redact` and admin-only `/metrics`. Python listens only on loopback port 8002; the Go front end is the shim upstream on port 8001.
- Each HTTP 200 from inference queues one `pii-filter:redact` usage event with the authenticated caller's credential, regardless of detected spans. Each accepted request gets a server-generated identity; caller-supplied IDs and usage-context headers cannot suppress charges.
- The controlplane prices that operation at $0.005. API keys use ordinary billing rules; chat subscription and billing exemptions remain unchanged.
- `X-Tinfoil-Billable-Requests: 1` acknowledges queued usage, not ledger settlement. The shared reporter buffers in memory and drops failed deliveries rather than retrying. Monitor `pii_usage_failed_total` and `pii_usage_dropped_total`; this change does not promise durable or exactly-once delivery.
- Websearch forwards the customer credential over its attested connection, never its service credential for privacy filtering. Prompt-injection inference still uses the service credential.
- Websearch emits only its session charge. It carries the endpoint receipt as `pii_filter_requests` in MCP structured content, including errors after a completed filter call. `null` means an unknown outcome, such as a lost response.
- The router reports known receipts and prices in `other_cost_usd`; it withholds totals when billing is unknown. It does not emit another charge for direct `pii-filter` proxy requests.

## Rollout order

1. Deploy the controlplane configuration with both catalog price and `meter_pricing.json` entry at $0.005. Verify the `pii-filter:redact` display name and normal account billing rules. The existing all-zero rate-limit rows do not implement endpoint admission; no new database migration is needed here.
2. Deploy the receipt-aware router. It must also skip its own billing for the endpoint-owned `pii-filter` model. Drain older router replicas before enabling endpoint reporting to avoid duplicate charges on direct proxy requests.
3. Deploy the websearch change that removes `ReportPIICheck` and forwards the customer credential. Drain **all** websearch replicas running the caller-billing implementation from PR #208. Against the old PII endpoint, this version reports zero filter charges. A short unbilled transition is safer than double billing.
4. Provision `USAGE_REPORTER_SECRET` for the PII enclave from the existing authorized reporting secret. Verify the host is enrolled for the shim's key-validation endpoint. Build and release this image using the repository's release workflow so the new digest is included in attestation; do not deploy the old pinned image with the new command.
5. Deploy the PII enclave. Verify health and attestation, then use a designated test account for a direct clean-text request and a websearch request. Check exactly one delivered `pii-filter:redact` event per completed inference, the correct customer account, and a $0.005 API charge. Also check a search-provider failure after successful inference, invalid input with no charge, and unauthenticated access rejection.
6. Publish the catalog/docs describing endpoint-owned billing after the services are live. Confirm the router reports the same filter request count through streaming and non-streaming paths and that no caller-side duplicate events remain.

These steps require production credentials and release approval. Unit tests and merged PRs alone do not prove that production secrets, pricing, or replicas have been updated.

## Rollback

Roll back the PII endpoint first to stop endpoint-owned charges, and drain it before reverting websearch to any version that reports filter usage itself. Keep the receipt-aware router when possible; it handles old endpoints without inventing charges. Never run the endpoint reporter and websearch's old filter reporter together. Catalog entries can remain during rollback, but suspend or update public pricing claims if metering is disabled.

## Operational limits

The request-body cap is 4 MiB, inference timeout is two minutes, and Python concurrency remains controlled by `OPF_MAX_CONCURRENCY`. Inference cannot be interrupted once started, so Python also rejects inputs over `OPF_MAX_INPUT_TOKENS` with 413 and sheds load with 503 once `OPF_MAX_QUEUE_DEPTH` requests are running or waiting; a single long input could otherwise hold every lane while callers behind it time out. The Go front end drains for up to 30 seconds on shutdown and then flushes its reporter with a bounded timeout. A client disconnect or lost response does not undo completed inference; if no receipt arrives, callers must treat its billing outcome as unknown rather than retrying under a free-request assumption.

## Health and restarts

`GET /health` returns `{"status": "ok"}` once the model is loaded and 503 while the request queue is full. Docker's restart policy only fires when the process exits, so the Go front end probes `/health` every 30 seconds and exits after three consecutive failures, letting `restart: always` replace a wedged inference process. Failures before the first successful probe are ignored so model loading never counts.

## Release and configuration

The `Tinfoil Release` workflow builds the image, writes its measured digest into `tinfoil-config.yml`, and publishes the attested release. `USAGE_REPORTER_SECRET` must be provisioned before deployment; startup fails if it is absent. `CONTROL_PLANE_URL` defaults to `https://api.tinfoil.sh` and the reporter ID is `pii-filter`. The container needs egress to `api.tinfoil.sh` for usage delivery; without the `networks` allowlist in `tinfoil-config.yml` every batch is dropped and filter requests go unbilled.

## Tests

- `go test -race ./...` exercises the front end with local HTTP fixtures and the real signing and batching client, without loading model weights.
- `pip install -r requirements-test.txt && python -m pytest test_server.py` exercises admission control and health with a stubbed model.
