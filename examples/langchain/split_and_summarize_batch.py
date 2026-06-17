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

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from _common import (
    COMPLETION_WINDOW,
    chat_batch_request,
    completion_text_from_row,
    index_results_by_custom_id,
    model_id,
    require_api_key,
    wait_for_batch_terminal,
)
from sference_sdk import SferenceClient

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
            chat_batch_request(
                custom_id=f"chunk-{idx}",
                user_content=str(user_content),
                system_content=str(system_content) if system_content else None,
            )
        )

    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = wait_for_batch_terminal(client, batch.id)
    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = index_results_by_custom_id(client.get_results(batch.id).results)

    print("Chunk summaries:")
    for idx in range(len(docs)):
        row = by_id.get(f"chunk-{idx}", {})
        print(f"  [{idx}] {completion_text_from_row(row)}")


if __name__ == "__main__":
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    main()
