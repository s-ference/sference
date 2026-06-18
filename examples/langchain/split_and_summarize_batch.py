# ---
# LangChain (offline): split documents, batch-summarize chunks on sference — not agents.
#
# Install:
#   uv sync --group dev --group examples
#   export SFERENCE_API_KEY=sk_...
#
# Run:
#   uv run python examples/langchain/split_and_summarize_batch.py
# ---

from __future__ import annotations

import os
import sys

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sference_sdk import SferenceClient
from sference_sdk.models import InferenceRequest

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
COMPLETION_WINDOW = "24h"
BATCH_POLL_INTERVAL_S = float(os.environ.get("SFERENCE_BATCH_POLL_INTERVAL_S", "5.0"))
BATCH_WAIT_TIMEOUT_S = float(os.environ.get("SFERENCE_BATCH_WAIT_TIMEOUT_S", "86400.0"))


def require_api_key() -> None:
    if not os.getenv("SFERENCE_API_KEY"):
        print("Error: SFERENCE_API_KEY is not set", file=sys.stderr)
        sys.exit(1)


def model_id() -> str:
    return os.environ.get("SFERENCE_MODEL", DEFAULT_MODEL)


client = SferenceClient()

RAW_TEXT = """
Sference batches are designed for workloads that tolerate minutes-to-hours of latency.
Jobs declare a completion window of 15 minutes, one hour, or twenty-four hours.
The scheduler prioritizes shorter windows when capacity is constrained.
Every request carries a custom_id for correlation with upstream systems.
Results are available as structured JSON or downloadable JSONL.
""".strip()

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Summarize the chunk in one short bullet. No preamble."),
        ("user", "{chunk}"),
    ]
)


def main() -> None:
    require_api_key()
    splitter = RecursiveCharacterTextSplitter(chunk_size=180, chunk_overlap=40)
    docs = splitter.split_documents([Document(page_content=RAW_TEXT)])
    print(f"LangChain produced {len(docs)} chunks\n")

    requests = []
    for idx, doc in enumerate(docs):
        messages = PROMPT.format_messages(chunk=doc.page_content)
        user_content = next(m.content for m in messages if m.type == "human")
        system_content = next((m.content for m in messages if m.type == "system"), None)
        requests.append(
            InferenceRequest.chat(
                custom_id=f"chunk-{idx}",
                user_content=str(user_content),
                system_content=str(system_content) if system_content else None,
                model=model_id(),
            )
        )

    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = client.wait_for_completion(
        batch.id,
        poll_interval=BATCH_POLL_INTERVAL_S,
        timeout=BATCH_WAIT_TIMEOUT_S,
    )
    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = client.get_results_indexed(batch.id)

    print("Chunk summaries:")
    for idx in range(len(docs)):
        row = by_id.get(f"chunk-{idx}")
        print(f"  [{idx}] {row.completion_text if row else ''}")


if __name__ == "__main__":
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    main()
