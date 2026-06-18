# PySpark + sference batch

[`enrich_dataframe_batch.py`](enrich_dataframe_batch.py) demonstrates the enterprise pattern:

1. Spark SQL / DataFrame transforms tickets on executors.
2. Driver collects stable `ticket_id` + prompt text (or writes JSONL for huge jobs).
3. One `submit_batch` call (same model per batch) → poll → join summaries back.

For millions of rows, write content-only JSONL to S3/MinIO from Spark, then `submit_batch(input_file=..., model=...)` from a thin orchestration step.

## Setup

```bash
uv sync --group dev --group examples
uv pip install pyspark
export SFERENCE_API_KEY=sk_...
```

## Run

```bash
uv run python examples/spark/enrich_dataframe_batch.py
```

Requires Java 8+ on `PATH` for Spark local mode.

## Test (local only)

Not run in CI. From the repo root:

```bash
uv sync --group dev --group examples-test --group examples-spark
uv run pytest examples/tests -k spark -q
```
