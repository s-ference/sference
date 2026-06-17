# LlamaIndex (batch ingestion) + sference

[`metadata_extraction_batch.py`](metadata_extraction_batch.py) mirrors the LangChain example for [LlamaIndex](https://docs.llamaindex.ai/):

1. `Document` → `SentenceSplitter` nodes (index-build preprocessing).
2. Each node becomes one batch row with a stable `custom_id`.
3. Model output is attached as metadata before you embed / upsert to a vector store.

This is **offline ingestion**, not `chat_engine` or agent tool loops.

## Setup

```bash
uv sync --group dev --group examples
export SFERENCE_API_KEY=sk_...
```

## Run

```bash
uv run python examples/llamaindex/metadata_extraction_batch.py
```
