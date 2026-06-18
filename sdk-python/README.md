# sference Python SDK

Installable package: `sference-sdk` (import: `sference_sdk`). Used by the `sference` CLI and your own automation.

## Install

```bash
uv add sference-sdk
```

Fallback:

```bash
pip install sference-sdk
```

From a clone of this repo:

```bash
uv sync --package sference-sdk
```

## Usage

Set `SFERENCE_API_KEY`, or pass `api_key=` to the client.

**Completion windows:** async workloads use `"15m"`, `"1h"`, or `"24h"` on background responses (`metadata.completion_window`), streams (`window=`), and batches (`window=`). Sync realtime endpoints (`/v1/chat/completions`, `/v1/messages`, blocking `/v1/responses`) do not take a window.

### `./workload.jsonl`

Batch APIs take a JSONL file: one JSON object per line. OpenAI-compatible lines include `custom_id`, `method`, `url`, and `body` (only `custom_id` + inner `body` are sent to `POST /v1/batches`; `method`/`url` are ignored). Content-only lines are `{"content": "..."}` (then pass `model=` on submit).

Inner `body` accepts **chat completions** (`messages`) or **Responses** (`input`, `max_output_tokens`, …). Responses fields are normalized to chat format at create and validated before enqueue. Invalid rows return HTTP 400 with `requests[i]` and optional `custom_id`.

Example `workload.jsonl`:

```jsonl
{"custom_id":"example-1","method":"POST","url":"/v1/chat/completions","body":{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"user","content":"Say hello in exactly one word."}]}}
{"custom_id":"example-2","method":"POST","url":"/v1/chat/completions","body":{"model":"Qwen/Qwen3.6-35B-A3B","messages":[{"role":"system","content":"You reply with one short sentence only."},{"role":"user","content":"What is 2+2?"}]}}
{"custom_id":"example-3","method":"POST","url":"/v1/responses","body":{"model":"Qwen/Qwen3.6-35B-A3B","input":[{"role":"user","content":"Reply with one word."}],"max_output_tokens":32}}
```

### Batches (sync)

Best for a **fixed JSONL workload**: one submit, poll until terminal, then fetch structured results or download JSONL via the API.

```python
from sference_sdk import SferenceClient

client = SferenceClient(api_key="sk_...")

batch = client.submit_batch(
    input_file="./workload.jsonl",
    model="Qwen/Qwen3.6-35B-A3B",
    window="24h",
)
done = client.wait_for_completion(batch.id, poll_interval=2.0, timeout=3600.0)
results = client.get_results(done.id)
by_id = results.index_by_custom_id()
print(by_id["row-a"].completion_text)
```

Or in one call:

```python
by_id = client.get_results_indexed(done.id)
```

Build chat rows without hand-assembling `body.messages`:

```python
from sference_sdk.models import InferenceRequest

req = InferenceRequest.chat(
    custom_id="row-a",
    user_content="Summarize this.",
    system_content="One sentence only.",
    model="Qwen/Qwen3.6-35B-A3B",
    temperature=0,
)
batch = client.submit_batch(requests=[req], window="24h")
```

Use a `model` supported by your sference deployment.

### OpenAI-compatible responses (sync)

Standalone or stream-associated jobs via `POST /v1/responses`. Keys need `responses:read` and `responses:write` (default on newly issued keys).

```python
from sference_sdk import SferenceClient

client = SferenceClient(api_key="sk_...")

created = client.create_response(
    model="Qwen/Qwen3.6-35B-A3B",
    input=[{"role": "user", "content": "Hello"}],
    metadata={"completion_window": "24h"},
)
row = client.get_response(created.id)
```

For a stream, add `stream_id` inside `metadata` next to `completion_window`.

### OpenAI Python SDK (`openai` package)

If you already use the official OpenAI client, point it at sference’s **`/v1`** endpoint and the same API key (with `responses:read` and `responses:write`).

```bash
pip install openai
```

```python
import asyncio
import os

from openai import AsyncOpenAI


async def main() -> None:
    client = AsyncOpenAI(
        base_url="https://api.sference.com/v1",
        api_key=os.environ["SFERENCE_API_KEY"],
    )

    response = await client.responses.create(
        model="Qwen/Qwen3.6-35B-A3B",
        input=[{"role": "user", "content": "Hello, world!"}],
        background=True,
    )
    # Poll GET /v1/responses/{id} until terminal; your openai version may expose
    # something like await client.responses.retrieve(response.id), or use
    # AsyncSferenceClient.get_response(response.id) with the same API key.


asyncio.run(main())
```

**Metadata:** to set `completion_window` or `stream_id` like the native SDK, pass them in the request body your `openai` version supports (for example `metadata=` on `create`, or `extra_body={"metadata": {...}}` if the helper does not list those fields yet).

### Async client — batches

`AsyncSferenceClient` uses `httpx.AsyncClient` so batch polling can run alongside other async I/O without blocking threads.

**Use case:** You already know the full set of prompts (for example a JSONL file) and want one scheduled unit of work with a clear terminal state and bulk results.

**Benefits:** Simple lifecycle (submit → wait → fetch results), fits large static workloads and JSONL-heavy pipelines.

```python
import asyncio

from sference_sdk import AsyncSferenceClient


async def main() -> None:
    async with AsyncSferenceClient(api_key="sk_...") as client:
        batch = await client.submit_batch(
            input_file="./workload.jsonl",
            model="Qwen/Qwen3.6-35B-A3B",
            window="24h",
        )
        done = await client.wait_for_completion(batch.id, poll_interval=2.0, timeout=3600.0)
        results = await client.get_results(done.id)
        print(results.status, results.output_url)


asyncio.run(main())
```

### Async client — streams

Stream-associated jobs use `create_response(..., metadata={"stream_id": ..., "completion_window": "24h"})`. Consume completions with `list_responses_events` / `iter_responses_events` (optional `stream_id`, `wait_ms` long-poll; optional checkpoints align with CLI `sference responses tail`).

**Use case:** Work arrives over time, or you want one id to group many responses and observe completions as they land.

**Benefits:** Independent submits with aggregated progress, stream-level status in the API/UI, and efficient event tailing.

```python
import asyncio

from sference_sdk import AsyncSferenceClient


async def main() -> None:
    async with AsyncSferenceClient(api_key="sk_...") as client:
        stream = await client.create_stream(name="sdk-demo", window="24h")
        await client.create_response(
            model="Qwen/Qwen3.6-35B-A3B",
            input=[{"role": "user", "content": "Hello"}],
            metadata={"stream_id": stream.id, "completion_window": "24h"},
        )
        async for ev in client.iter_responses_events(stream_id=stream.id, checkpoint=False):
            print(ev.completion_id, ev.status)


asyncio.run(main())
```

## CLI

For `sference batch …` and `sference stream …` commands, see the [CLI README](../cli/README.md).
