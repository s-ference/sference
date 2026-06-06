# Prefect + sference batch responses

[`ai_data_analyst_batch_responses.py`](ai_data_analyst_batch_responses.py) is a two-stage [Prefect](https://www.prefect.io/) flow inspired by Prefect’s [AI data analyst + pydantic-ai example](https://github.com/PrefectHQ/prefect/blob/main/examples/ai_data_analyst_with_pydantic_ai.py), rewritten for sference:

1. **Submit** — `create_response(..., background=True, metadata={"completion_window": "1h"})` per analysis prompt (Prefect task map).
2. **Wait** — `wait_for_response(id)` until each job is terminal (second task map).

Prefect handles orchestration, retries, and run state; the SDK handles HTTP and polling.

## Setup

From the **oss** repository root:

```bash
uv sync --group dev --group examples
export SFERENCE_API_KEY=sk_...
export SFERENCE_MODEL=Qwen/Qwen3.6-35B-A3B            # must match your deployment
```

## Run

```bash
uv run python examples/prefect/ai_data_analyst_batch_responses.py
```

Serve as a Prefect deployment (worker + UI):

```bash
uv run python examples/prefect/ai_data_analyst_batch_responses.py --serve
```

Then trigger **ai-data-analyst-batch-responses** from the Prefect UI or CLI.

## See also

- [SDK README — OpenAI-compatible responses](../../sdk-python/README.md)
- [sference docs](https://sference.com) — Responses (async / background)
