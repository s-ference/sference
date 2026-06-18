# Ray Data + sference batch

[`classify_reviews_batch.py`](classify_reviews_batch.py) uses [Ray Data](https://docs.ray.io/en/latest/data/data.html) for tabular input and sference for inference:

1. Load a small pandas table into `ray.data.from_pandas`.
2. Build one `submit_batch` request per row (`custom_id` = business key).
3. Poll until terminal, join `result_json` back onto each row.

At scale, Ray partitions preprocessing (`map_batches`, writes to object storage); each partition can emit JSONL chunks that you submit as separate batches or merge on the driver. This demo keeps the join on the driver for clarity.

When running via `uv run`, the script sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` so Ray does not try to replicate the uv environment to workers for this small driver-side demo.

## Setup

```bash
uv sync --group dev --group examples
export SFERENCE_API_KEY=sk_...
```

## Run

```bash
uv run python examples/ray_data/classify_reviews_batch.py
```
