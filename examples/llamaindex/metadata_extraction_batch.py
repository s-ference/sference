# ---
# LlamaIndex (offline): parse nodes, batch-extract metadata on sference — not chat agents.
#
# Install:
#   uv sync --group dev --group examples
#   export SFERENCE_API_KEY=sk_...
#
# Run:
#   uv run python examples/llamaindex/metadata_extraction_batch.py
# ---

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

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

POLICY_DOCS = [
    Document(
        text=(
            "Section 4.2: Customer data is processed in EU regions only. "
            "Subprocessors must sign SCCs before onboarding."
        ),
        metadata={"doc_id": "policy-a", "section": "4.2"},
    ),
    Document(
        text=(
            "Section 7.1: Incident response begins within one hour of P1 detection. "
            "Customers receive status updates every four hours."
        ),
        metadata={"doc_id": "policy-b", "section": "7.1"},
    ),
]

SYSTEM = (
    "Extract metadata as one line: THEME: <short label> | ENTITIES: comma-separated nouns"
)


def main() -> None:
    require_api_key()
    parser = SentenceSplitter(chunk_size=256, chunk_overlap=32)
    nodes = parser.get_nodes_from_documents(POLICY_DOCS)
    print(f"LlamaIndex produced {len(nodes)} nodes\n")

    requests = []
    for idx, node in enumerate(nodes):
        doc_id = node.metadata.get("doc_id", "unknown")
        custom_id = f"{doc_id}-node-{idx}"
        requests.append(
            chat_batch_request(
                custom_id=custom_id,
                user_content=f"Clause:\n{node.get_content()}",
                system_content=SYSTEM,
            )
        )

    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = wait_for_batch_terminal(client, batch.id)
    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = index_results_by_custom_id(client.get_results(batch.id).results)

    print("Node metadata:")
    for idx, node in enumerate(nodes):
        doc_id = node.metadata.get("doc_id", "unknown")
        custom_id = f"{doc_id}-node-{idx}"
        row = by_id.get(custom_id, {})
        print(f"  {custom_id}: {completion_text_from_row(row)}")


if __name__ == "__main__":
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    main()
