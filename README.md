# confidential-pii-cpu

CPU enclave serving the [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) — a bidirectional token-classification model for PII span detection and redaction.

The model returns structured spans and redacted text without generative prompting. Websearch applies its own removal policy to these spans; the model endpoint does not apply that policy.

## Architecture

The authenticated shim forwards to a Go front end on port 8001, which proxies to the Python inference process on loopback port 8002. The front end owns usage reporting through `usage-reporting-go`; Python only loads the model and performs inference. Neither request text nor detected spans enter billing events.

## API

### `POST /redact`

Requires a Tinfoil bearer credential validated by the shim. Each HTTP 200 inference queues one `pii-filter:redact` event attributed to that credential and returns `X-Tinfoil-Billable-Requests: 1`. No detection is required to incur the charge. Invalid requests, failed inference, health checks, and metrics are not charged. Authentication and inference stay inside the enclave; the Python process does not receive the credential.

```json
{"text": "Call John Smith at 555-867-5309"}
```

```json
{
  "schema_version": 1,
  "summary": {"output_mode": "typed", "span_count": 2, "by_label": {"private_person": 1, "private_phone": 1}, "decoded_mismatch": false},
  "text": "Call John Smith at 555-867-5309",
  "detected_spans": [
    {"label": "private_person", "start": 5, "end": 15, "text": "John Smith", "placeholder": "<PRIVATE_PERSON>"},
    {"label": "private_phone", "start": 20, "end": 32, "text": "555-867-5309", "placeholder": "<PRIVATE_PHONE>"}
  ],
  "redacted_text": "Call <PRIVATE_PERSON> at <PRIVATE_PHONE>"
}
```

### `GET /health`

Returns `{"status": "ok"}` once the model is loaded.

## Model

- **Weights**: `openai/privacy-filter` (Apache 2.0, 1.5B total / 50M active params)
- **Verified model mount**: loaded from the read-only modelwrap filesystem pinned in `tinfoil-config.yml`
- **CPU inference**: `OPF_DEVICE=cpu`

## Deployment

The release workflow builds the image and updates its measured digest. `USAGE_REPORTER_SECRET` must be provisioned before deployment; startup fails if it is absent. `CONTROL_PLANE_URL` defaults to `https://api.tinfoil.sh`. The reporter ID is `pii-filter`.

See [DEPLOYMENT.md](DEPLOYMENT.md) for the cross-service rollout and rollback order. Run `go test -race ./...` for the front-end tests; they use local HTTP fixtures and the real signing/batching client without loading model weights.

Full API and pricing documentation: [Privacy filter](https://docs.tinfoil.sh/guides/privacy-filter).
