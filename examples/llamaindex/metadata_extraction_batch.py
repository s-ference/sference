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

import os
import sys

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

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
            InferenceRequest.chat(
                custom_id=custom_id,
                user_content=f"Clause:\n{node.get_content()}",
                system_content=SYSTEM,
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

    print("Node metadata:")
    for idx, node in enumerate(nodes):
        doc_id = node.metadata.get("doc_id", "unknown")
        custom_id = f"{doc_id}-node-{idx}"
        row = by_id.get(custom_id)
        print(f"  {custom_id}: {row.completion_text if row else ''}")


if __name__ == "__main__":
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    main()
