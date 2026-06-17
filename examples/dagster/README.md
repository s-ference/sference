# Dagster + sference batch

[`document_enrichment_batch.py`](document_enrichment_batch.py) defines three [Dagster](https://dagster.io/) assets:

1. **`source_documents`** — local demo corpus (swap for a warehouse table or file asset).
2. **`sference_batch_id`** — `submit_batch(..., window="24h")` with one row per document.
3. **`enriched_documents`** — waits for terminal status, joins batch results back by `custom_id`.

Dagster owns lineage and materialization; sference runs GPU inference on the scheduler.

## Setup

```bash
uv sync --group dev --group examples
export SFERENCE_API_KEY=sk_...
export SFERENCE_MODEL=Qwen/Qwen3.6-35B-A3B   # optional
```

## Run

```bash
uv run python examples/dagster/document_enrichment_batch.py
```

For a persistent Dagster deployment, move `defs` into a `definitions.py` and run `dagster dev`.
