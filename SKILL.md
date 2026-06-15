---
name: sference-sdk
description: >-
  Writes Python that calls the sference inference API with the sference-sdk
  package (SferenceClient, AsyncSferenceClient). Default to the Responses API
  with background=True. Use when a coding agent (Claude Code, Cursor, etc.) needs
  to send prompts, group requests under a stream, run bulk JSONL batches, or read
  completions — or when the user mentions sference, sference-sdk, SFERENCE_API_KEY,
  or the PyPI package sference-sdk. Not for changing this repo or the platform API.
disable-model-invocation: false
---

# sference SDK (Python)

For coding agents writing application code that **uses** sference. Use the **`sference-sdk`** package only — do **not** read the SDK source, hit raw HTTP, or clone the platform monorepo.

**Vibe coding?** Copy the short rules from **[PROMPT.txt](PROMPT.txt)** first ([raw](https://raw.githubusercontent.com/s-ference/sference/main/PROMPT.txt)). Use this file when you need examples, stream/batch detail, and a method map—or for a Cursor skill install (see **[AGENTS.md](AGENTS.md)**).

**Default to the Responses API with `background=True`** and poll for completion. Reach for batches or streams only when the situation below calls for them.

## Install

```bash
uv add sference-sdk
# or
pip install sference-sdk
```

Import name is `sference_sdk`.

## Credentials

| Variable | Meaning |
|----------|---------|
| `SFERENCE_API_KEY` | **Required** — secret key `sk_...` from Console → API keys (https://app.sference.com) |

Read from the environment or pass explicitly. Keep keys in env or a gitignored `.env` — never hardcode/commit them.

```python
from sference_sdk import SferenceClient

client = SferenceClient()                  # reads SFERENCE_API_KEY
client = SferenceClient(api_key="sk_...")  # explicit
```

Use `api_key`/env, **not** `client.login()`, in agent-generated code.

## Choosing an approach

| Situation | Use | Notes |
|-----------|-----|-------|
| **Default** — one or many independent prompts | **Responses API, `background=True`** | Submit, then poll `wait_for_response` |
| Many requests that belong to the **same logical group** | **Stream** (a user-created bucket) + responses tagged with its `stream_id` | The stream **must be created first** |
| A fixed JSONL file run once for bulk output | **Batch** | `submit_batch` → `wait_for_completion` → results |

### Completion window

**Async** workloads use **`"15m"`**, **`"1h"`**, or **`"24h"`** completion windows (default `"24h"`; shorter windows are prioritized to finish sooner). Set via `metadata={"completion_window": ...}` on background responses, and via `window=` on `create_stream` and `submit_batch`.

**Realtime** sync endpoints (`POST /v1/chat/completions`, `POST /v1/messages`, blocking `POST /v1/responses` without `background: true`) do not take a completion window — they return when inference finishes.

## Default: Responses API (background)

`background=True` returns immediately with `status="in_progress"`; poll until terminal.

```python
import os
from sference_sdk import SferenceClient

client = SferenceClient(api_key=os.environ["SFERENCE_API_KEY"])

resp = client.create_response(
    model="<model-id>",                       # a model deployed on the user's sference
    input="Summarize this incident report.",  # str, or [{"role": "user", "content": "..."}]
    background=True,
    metadata={"completion_window": "24h"},     # "15m" / "1h" / "24h" (background only)
)

done = client.wait_for_response(resp.id, poll_interval=2.0, timeout=3600.0)
print(done.status)                             # completed | failed | cancelled
```

`wait_for_response` defaults to `timeout=3600.0`; pass a larger value for long jobs.

### Reading the output text

`done.output` is a list of items. Final text lives in `message` items as `output_text` parts:

```python
def response_text(resp) -> str:
    return "".join(
        part.text
        for item in (resp.output or [])
        if item.type == "message"
        for part in item.content
    )
```

`done.usage` holds `input_tokens` / `output_tokens` / `total_tokens`; `done.error` is set when `status == "failed"`.

## Streams (a user-created "preset" bucket)

A **stream** groups many responses that share the same purpose and window — think of it like a **preset**: e.g. one stream for "system-log analysis" under which you submit a stream of related prompts and watch their completions roll in.

**The user creates the stream first**, then every related response carries that `stream_id` in its metadata.

```python
import os
from sference_sdk import SferenceClient

client = SferenceClient(api_key=os.environ["SFERENCE_API_KEY"])

stream = client.create_stream(name="syslog-analysis", window="24h")  # "15m" / "1h" / "24h"

for line in log_lines:
    client.create_response(
        model="<model-id>",
        input=f"Classify this log line: {line}",
        background=True,
        metadata={"stream_id": stream.id, "completion_window": "24h"},
    )

# Tail completions as they finish (cursor-based; checkpoint=False = don't persist a cursor)
for ev in client.iter_responses_events(stream_id=stream.id, checkpoint=False):
    print(ev.completion_id, ev.status, ev.total_tokens)
```

`get_stream(stream.id)` returns counters (`completed_items`, `pending_items`, `completion_ratio`, …). When done, `cancel_stream` stops new work; `archive_stream` finalizes it.

## Batches (bulk JSONL, secondary)

Use when the user already has a JSONL file and wants one job with bulk results.

```python
batch = client.submit_batch(input_file="./prompts.jsonl", model="<model-id>", window="24h")
done = client.wait_for_completion(batch.id, poll_interval=5.0, timeout=86_400.0)
client.download_results_jsonl(done.id, out="./out.jsonl")   # arg is `out`
```

`wait_for_completion` defaults to **`timeout=30.0`s** — always pass an explicit `timeout` for real batches.

**Row bodies:** each `requests[].body` may use chat completions (`messages`) or Responses API shape (`input`, `max_output_tokens`, …). The API normalizes Responses fields to chat format at create, validates, then enqueues — invalid rows return **400** with `requests[i]` and `custom_id` (no late worker failures). Do not set `background: true` on batch row bodies.

JSONL line shapes (both accepted; OpenAI envelope sends only `custom_id` + inner `body`):

```jsonl
{"custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":{"model":"<model-id>","messages":[{"role":"user","content":"hi"}]}}
{"custom_id":"req-2","method":"POST","url":"/v1/responses","body":{"model":"<model-id>","input":[{"role":"user","content":"hi"}]}}
{"content":"content-only line — requires model= on submit_batch"}
```

## Async

`AsyncSferenceClient` mirrors the sync API with `await`; same method names and arguments.

```python
import asyncio, os
from sference_sdk import AsyncSferenceClient

async def main() -> None:
    async with AsyncSferenceClient(api_key=os.environ["SFERENCE_API_KEY"]) as client:
        resp = await client.create_response(
            model="<model-id>", input="hi", background=True,
            metadata={"completion_window": "24h"},
        )
        done = await client.wait_for_response(resp.id, timeout=3600.0)
        print(done.status)

asyncio.run(main())
```

## SDK method map (sync; async = same with `await`)

| Task | Methods |
|------|---------|
| Responses (default) | `create_response` (set `background=True`), `wait_for_response`, `get_response`, `cancel_response`, `list_responses` |
| Completion events | `list_responses_events` (`stream_id`, `starting_after`, `wait_ms` long-poll); `iter_responses_events` (`stream_id`, `checkpoint`, `from_latest`) |
| Streams | `create_stream(name, window=)`, `get_stream`, `list_streams`, `cancel_stream`, `archive_stream` |
| Batches | `submit_batch`, `wait_for_completion`, `get_results`, `download_results_jsonl`, `get_batch`, `list_batches`, `cancel_batch` |

`create_response` / `submit_batch` / `create_stream` / `*_events` take **keyword arguments**. Response status: `in_progress → completed | failed | cancelled`. Batch status: `pending → running → completed | failed | cancelled`.

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Defaulting to synchronous create | Prefer `create_response(background=True)` then `wait_for_response` |
| Submitting stream responses before the stream exists | `create_stream(...)` first, then pass `metadata={"stream_id": ...}` |
| `download_results_jsonl(..., path=...)` | Argument is **`out`** |
| `wait_for_completion(batch.id)` with default timeout | Default is 30s — pass an explicit `timeout` |
| Inventing window values (`"2h"`, `"30m"`) | Async windows are `"15m"` / `"1h"` / `"24h"` only (responses, streams, batches) |
| Content-only JSONL without a model | Pass `model=` to `submit_batch` |
| Responses-shaped batch row without `messages` | Use `input` in `body` (normalized at create) or chat `messages` — not `background: true` |
| Guessing JSON field names | See `contract/openapi.json` |
| Re-running `create_response` on Prefect retry after success | Split **submit** and **wait** tasks; retry wait only when you already have a `response_id` |

## Framework integrations

Runnable examples live under **[examples/](examples/)** (`uv sync --group dev --group examples` from this repo).

### Prefect

Use a **two-stage flow**: enqueue with `background=True`, then poll in a separate task. Prefect owns retries and UI observability; the SDK owns HTTP and `wait_for_response` polling.

Pattern (see **[examples/prefect/ai_data_analyst_batch_responses.py](examples/prefect/ai_data_analyst_batch_responses.py)**):

1. **Prepare** — build prompts locally (pandas, files, etc.); no inference in this stage.
2. **Submit** (`@task`, `retries=…`) — `create_response(..., background=True, metadata={"completion_window": "1h"})` per item; return `response.id`.
3. **Wait** (`@task`, `retries=…`) — `wait_for_response(response_id)`; parse `done.output` message / `output_text` parts.
4. **Fan-out** — `create_response_request.map(prompts)` then `wait_for_response_completion.map(response_id_futures)` so each prompt is its own task in the Prefect UI.

```python
from prefect import flow, task
from sference_sdk import SferenceClient

client = SferenceClient()  # SFERENCE_API_KEY from env

@task(name="create-response-request", retries=2)
def create_response_request(prompt: AnalysisPrompt) -> str:
    created = client.create_response(
        model="...",
        input=[{"role": "user", "content": prompt.user_content}],
        background=True,
        metadata={"completion_window": "1h"},  # "15m" / "1h" / "24h"
    )
    return created.id

@task(name="wait-for-response-completion", retries=2)
def wait_for_response_completion(response_id: str) -> dict:
    done = client.wait_for_response(response_id)
    # extract text from done.output (message → output_text parts)
    return {"id": done.id, "status": done.status, "text": "..."}

@flow
def my_flow() -> None:
    prompts = build_analysis_prompts(prepare_sample_dataset())
    ids = create_response_request.map(prompts)
    results = wait_for_response_completion.map(ids)
    ordered = [f.result() for f in results]
```

Run: `uv run python examples/prefect/ai_data_analyst_batch_responses.py` — add `--serve` for a Prefect deployment. Details: **[examples/prefect/README.md](examples/prefect/README.md)**.

**Why not in-process LLMs in the flow?** Background responses decouple orchestration from GPU work: workers stay thin, failed **wait** tasks can retry without re-submitting, and `completion_window` matches batch SLA semantics.

## Reference

- SDK README: [sdk-python/README.md](sdk-python/README.md)
- API contract: [contract/openapi.json](contract/openapi.json)
- Examples index: [examples/README.md](examples/README.md)
