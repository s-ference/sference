# Hugging Face Datasets + sference batch

[`map_splits_to_batch.py`](map_splits_to_batch.py) shows the usual HF + hosted inference pattern:

1. Build a `datasets.Dataset` (replace with `load_dataset(...)` in production).
2. Serialize rows to `InferenceRequest` objects with stable `custom_id` values.
3. `submit_batch` → wait → `Dataset.map` to attach model output columns.

Use this for offline eval sets, synthetic QA generation, or labeling prep — not for per-row `model.generate()` on your own GPU.

## Setup

```bash
uv sync --group dev --group examples
export SFERENCE_API_KEY=sk_...
```

## Run

```bash
uv run python examples/hf_datasets/map_splits_to_batch.py
```
