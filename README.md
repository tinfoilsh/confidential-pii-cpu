# Confidential Privacy Filter

Tinfoil's privacy filter serves the [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) inside a CPU secure enclave. It detects and redacts personally identifiable information in text, returning typed spans and placeholder-redacted output. Web search calls it to mask outgoing queries; API clients can call it directly. Available on `pii-filter.tinfoil.sh`.

Client-facing behavior (request and response shape, span labels, and limits) is documented at [docs.tinfoil.sh](https://docs.tinfoil.sh):

- [Privacy filter](https://docs.tinfoil.sh/guides/privacy-filter)
- [PII protection in web search](https://docs.tinfoil.sh/guides/web-search#pii-protection)
- [Safety models](https://docs.tinfoil.sh/models/safety)

## Development

Run the tests without model weights:

```bash
go test -race ./...
pip install -r requirements-test.txt && python -m pytest test_server.py test_chunking.py
```

Run locally after installing `requirements.txt` and a CPU build of torch; the front end starts the Python inference server itself:

```bash
export OPF_CHECKPOINT="/path/to/privacy-filter/original"
export USAGE_REPORTER_SECRET="your-usage-reporter-secret"

go run .
```

Long inputs are cut into overlapping windows that the Go gateway's internal `/queue` spreads across every worker process, fairly per caller; see [DEPLOYMENT.md](DEPLOYMENT.md) for the window geometry, limits, health checks, and usage reporting. Upstream `opf` changes are carried as [patches](patches/README.md) applied at image build.

## Reporting Vulnerabilities

Please report security vulnerabilities by either:

- Emailing [security@tinfoil.sh](mailto:security@tinfoil.sh)
- Opening an issue on GitHub on this repository

We aim to respond to (legitimate) security reports within 24 hours.
