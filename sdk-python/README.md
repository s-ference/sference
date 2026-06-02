# sference Python SDK

Installable package: `sference-sdk` (import: `sference_sdk`). Used by the `sference` CLI and your own automation.

## Install

```bash
pip install sference-sdk
# or: uv add sference-sdk
```

## Usage

Set `SFERENCE_API_KEY` and optional `SFERENCE_BASE_URL` (default `https://api.sference.com`), or pass `api_key=` / `base_url=` to the client.

### Batches (sync)

Best for a **fixed JSONL workload**: one submit, poll until terminal, then fetch structured results or download JSONL via the API.

```python
from sference_sdk import SferenceClient

client = SferenceClient(api_key="sk_...", base_url="https://api.sference.com")

batch = client.submit_batch(
    input_file="./workload.jsonl",
    model="Qwen/Qwen3.6-35B-A3B",
    window="24h",
)
done = client.wait_for_completion(batch.id, poll_interval=2.0, timeout=3600.0)
results = client.get_results(done.id)
print(results.status, results.output_url)
```

Use a `model` supported by your sference deployment.

### OpenAI-compatible responses (sync)

Standalone or stream-associated jobs via `POST /v1/responses`. Keys need `responses:read` and `responses:write` (default on newly issued keys).

```python
from sference_sdk import SferenceClient

client = SferenceClient(api_key="sk_...", base_url="https://api.sference.com")

created = client.create_response(
    model="Qwen/Qwen3.6-35B-A3B",
    input=[{"role": "user", "content": "Hello"}],
    metadata={"completion_window": "24h"},
)
row = client.get_response(created.id)
```

For a stream, add `stream_id` inside `metadata` next to `completion_window`.

### OpenAI Python SDK (`openai` package)

If you already use the official OpenAI client, point it at a sference-compatible **`/v1`** base URL and the same API key (with `responses:read` and `responses:write`).

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
    # AsyncSferenceClient.get_response(response.id) with the same host and key.


asyncio.run(main())
```

**Self-hosted** (local API): use `base_url="http://127.0.0.1:8000/v1"` (or your `SFERENCE_BASE_URL` + `"/v1"`). **`model`** must match a model your inference workers consume.

**Metadata:** to set `completion_window` or `stream_id` like the native SDK, pass them in the request body your `openai` version supports (for example `metadata=` on `create`, or `extra_body={"metadata": {...}}` if the helper does not list those fields yet).

### Async client — batches

`AsyncSferenceClient` uses `httpx.AsyncClient` so batch polling can run alongside other async I/O without blocking threads.

**Use case:** You already know the full set of prompts (for example a JSONL file) and want one scheduled unit of work with a clear terminal state and bulk results.

**Benefits:** Simple lifecycle (submit → wait → fetch results), fits large static workloads and JSONL-heavy pipelines.

```python
import asyncio

from sference_sdk import AsyncSferenceClient


async def main() -> None:
    async with AsyncSferenceClient(api_key="sk_...", base_url="https://api.sference.com") as client:
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
    async with AsyncSferenceClient(api_key="sk_...", base_url="https://api.sference.com") as client:
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

## cURL (same API the SDK calls)

API keys need `responses:read` and `responses:write` (default on newly issued keys). `X-API-Key` or `Authorization: Bearer sk_...` are both accepted.

```bash
export TOKEN=sk_...
BASE_URL=https://api.sference.com

RID=$(curl -sS -X POST "${BASE_URL}/v1/responses" \
  -H "X-API-Key: $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B",
    "input": [{"role": "user", "content": "Hello"}],
    "metadata": {"completion_window": "24h"}
  }' | jq -r '.id')

curl -sS "${BASE_URL}/v1/responses/${RID}" \
  -H "X-API-Key: $TOKEN"
```

For self-hosted APIs, set `BASE_URL` to your API origin (no `/v1` suffix on `BASE_URL` here—the paths already include `/v1`). Without `jq`, read `id` from the POST JSON and substitute it in the GET URL.

## CLI

For `sference batch …` and `sference stream …` commands, see the [CLI README](../cli/README.md).
