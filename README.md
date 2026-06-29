# confidential-pii-cpu

CPU enclave serving the [OpenAI Privacy Filter](https://huggingface.co/openai/privacy-filter) — a bidirectional token-classification model for PII span detection and redaction.

Unlike the existing `gpt-oss-safeguard-120b` safeguard (an LLM that classifies whole queries as PII/not-PII), this model returns structured spans: `[{label, start, end, text, placeholder}]` plus a redacted string. That granularity lets the websearch PII guard redact sensitive substrings instead of blocking the entire query.

## API

### `POST /redact`

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
- **Downloaded at startup** from HuggingFace, cached on the `hf-cache` volume
- **CPU inference**: `OPF_DEVICE=cpu`

## Deployment

Custom Docker image built in CI (`tinfoil-release.yml`). The `tinfoil-config.yml` image digest is a placeholder until the first release populates it.
