# Confidential Privacy Filter

Tinfoil's privacy filter serves the [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) inside a CPU secure enclave. It detects and redacts personally identifiable information in text, returning typed spans and placeholder-redacted output. Web search calls it to mask outgoing queries; API clients can call it directly. Available on `pii-filter.tinfoil.sh`.

## How it works

For each `POST /redact` request the enclave:

1. Authenticates the caller's bearer credential at the shim
2. Rejects inputs that are too long or arrive while the request queue is full
3. Runs the token-classification model over the text
4. Returns the detected spans and redacted text, and records one billable request

Client-facing behavior (request and response shape, span labels, limits, and pricing) is documented at [docs.tinfoil.sh](https://docs.tinfoil.sh):

- [Privacy filter](https://docs.tinfoil.sh/guides/privacy-filter)
- [PII protection in web search](https://docs.tinfoil.sh/guides/web-search#pii-protection)
- [Safety models](https://docs.tinfoil.sh/models/safety)

## Running locally

```bash
export OPF_CHECKPOINT="/path/to/privacy-filter/original"
export USAGE_REPORTER_SECRET="your-usage-reporter-secret"

go run .
```

The front end starts the Python inference server itself; install `requirements.txt` and a CPU build of torch first. See [DEPLOYMENT.md](DEPLOYMENT.md) for release, rollout, and operational notes.

## Architecture Overview

- **[main.go](main.go)**: Entry point; starts the inference process and the HTTP front end
- **[gateway.go](gateway.go)**: Authenticated reverse proxy that owns usage reporting and the billable-request receipt
- **[health.go](health.go)**: Probes inference and exits so Docker restarts a wedged process
- **[server.py](server.py)**: FastAPI inference server wrapping the `opf` model with admission control
- **[constants.go](constants.go)**: Ports, paths, timeouts, and limits shared by the front end
- **[tinfoil-config.yml](tinfoil-config.yml)**: Attested enclave configuration, model mount, and resources

## Reporting Vulnerabilities

Please report security vulnerabilities by either:

- Emailing [security@tinfoil.sh](mailto:security@tinfoil.sh)
- Opening an issue on GitHub on this repository

We aim to respond to (legitimate) security reports within 24 hours.
