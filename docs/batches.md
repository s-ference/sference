# Batch API

Submit bulk inference with `POST /v1/batches`. Base URL: **`https://api.sference.com`** (Bearer API key).

**Completion windows:** `"15m"`, `"1h"`, or `"24h"` on every batch (default `"24h"`). For sync realtime inference, use `/v1/chat/completions`, `/v1/messages`, or blocking `/v1/responses` instead.

OpenAPI: [api.sference.com/openapi.json](https://api.sference.com/openapi.json)

## Quick flow

1. `POST /v1/batches` — inline `requests[]` + `window: "15m"`, `"1h"`, or `"24h"`.
2. `GET /v1/batches/{id}` — poll status.
3. `GET /v1/batches/{id}/results.jsonl` — download per-row outcomes (`application/x-ndjson`).

## Request example

```json
{
  "window": "24h",
  "requests": [
    {
      "custom_id": "row-a",
      "body": {
        "model": "moonshotai/Kimi-K2.6",
        "messages": [{"role": "user", "content": "Hello"}]
      }
    }
  ]
}
```

## Row `body` shapes

Each `requests[].body` must include **`model`**, and **every row in a batch must use the same model**. At create time the API **normalizes** the body to chat-completions format, **validates** it, then persists — invalid rows return **HTTP 400** with `requests[i]` and optional `custom_id` (nothing is enqueued).

### Chat completions

```json
{
  "model": "moonshotai/Kimi-K2.6",
  "messages": [{"role": "user", "content": "Summarize this."}],
  "temperature": 0.2,
  "max_tokens": 512
}
```

`messages` must be a non-empty array. Same optional fields as `POST /v1/chat/completions` (`tools`, `tool_choice`, …).

### Responses API (normalized at create)

Same fields as `POST /v1/responses`. The API converts `input` → `messages`, `max_output_tokens` → `max_tokens`, etc., before enqueue:

```json
{
  "model": "moonshotai/Kimi-K2.6",
  "input": [{"role": "user", "content": "Summarize this."}],
  "instructions": "Reply in one sentence.",
  "max_output_tokens": 512
}
```

String `input` shorthand is supported. Stored rows always contain `messages`; workers never see raw Responses shape.

### Rejected at create

| Case | Example error |
|------|----------------|
| Missing `messages` and `input` | `body must include a non-empty messages list or Responses API input` |
| Empty `messages: []` | `body.messages must be a non-empty list` |
| Invalid Responses payload | Field-level validation on `input`, etc. |
| `background: true` in row body | `background is not supported in batch request bodies` |

## JSONL (SDK / CLI)

Two line formats in `submit_batch` / `sference batch submit`:

**OpenAI-style envelope** — only `custom_id` + inner `body` are sent; `method` / `url` are ignored:

```jsonl
{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"…","messages":[{"role":"user","content":"hi"}]}}
{"custom_id":"b","method":"POST","url":"/v1/responses","body":{"model":"…","input":[{"role":"user","content":"hi"}]}}
```

**Content-only** — requires global `model=` on submit:

```jsonl
{"content":"Classify this log line."}
```

See [CLI README](../cli/README.md) and [SDK README](../sdk-python/README.md).

## Not OpenAI Batch API

sference batches are **not** OpenAI’s file-upload batch flow:

- No `POST /v1/files`.
- Create uses inline `requests[]`, not a uploaded JSONL file id.
- Results use `result_json` / `error_json`, not OpenAI’s batch result envelope.

For per-request async Responses without a batch id, use `POST /v1/responses` with `background: true`.

## SDK quick start

```python
from sference_sdk import SferenceClient

client = SferenceClient(api_key="sk_...")
batch = client.submit_batch(input_file="workload.jsonl", model="moonshotai/Kimi-K2.6", window="24h")
done = client.wait_for_completion(batch.id, poll_interval=5.0, timeout=86_400.0)
client.download_results_jsonl(done.id, out="./out.jsonl")
```

```bash
sference batch submit --input-file workload.jsonl --model moonshotai/Kimi-K2.6 --window 24h
sference batch wait --batch-id <id>
```
