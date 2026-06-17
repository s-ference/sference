# LangChain (batch indexing) + sference

[`split_and_summarize_batch.py`](split_and_summarize_batch.py) is an **offline index-build** pattern — not an agent loop:

1. `RecursiveCharacterTextSplitter` produces chunks (`Document` objects).
2. `ChatPromptTemplate` formats each chunk into chat messages.
3. Messages are sent via `submit_batch` instead of `ChatOpenAI.invoke()`.

Use this when building RAG corpora, digesting PDFs, or nightly summarization. Pair with your vector DB loader after batch results land.

## Setup

```bash
uv sync --group dev --group examples
export SFERENCE_API_KEY=sk_...
```

## Run

```bash
uv run python examples/langchain/split_and_summarize_batch.py
```
