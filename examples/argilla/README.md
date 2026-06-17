# Argilla + sference batch (suggestions)

[`suggestion_batch.py`](suggestion_batch.py) drafts **offline suggestions** for human review:

1. Read [`sample_records.json`](sample_records.json) (minimal Argilla record shape).
2. Batch-generate reply drafts on sference.
3. Write [`sample_suggestions.json`](sample_suggestions.json) for inspection.

Optionally push to a live Argilla server when `ARGILLA_API_URL` and `ARGILLA_API_KEY` are set.

## Setup

```bash
uv sync --group dev --group examples
export SFERENCE_API_KEY=sk_...
# optional live push:
# uv pip install argilla
# export ARGILLA_API_URL=http://localhost:6900
# export ARGILLA_API_KEY=...
# export ARGILLA_DATASET=sference-support-suggestions
```

## Run

```bash
uv run python examples/argilla/suggestion_batch.py
```

Human annotators correct drafts in Argilla; nothing is sent to customers automatically.
