# ---
# Dagster assets: build prompts locally, submit one sference batch, materialize enriched rows.
#
# Install (oss repo root):
#   uv sync --group dev --group examples
#   export SFERENCE_API_KEY=sk_...
#
# Run:
#   uv run python examples/dagster/document_enrichment_batch.py
# ---

from __future__ import annotations

import os
import sys
from typing import Any

from dagster import Definitions, asset, materialize

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

SAMPLE_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "doc-1",
        "text": "Acme Corp reported Q3 revenue up 12% year over year, citing strong enterprise demand.",
    },
    {
        "id": "doc-2",
        "text": "The municipal council voted to delay the bridge renovation until next spring.",
    },
    {
        "id": "doc-3",
        "text": "Researchers published a preprint on protein folding benchmarks using synthetic data.",
    },
]

SYSTEM = (
    "You extract structured metadata. Reply with exactly one line: "
    "TOPIC: <short label> | SENTIMENT: positive|neutral|negative"
)


@asset
def source_documents() -> list[dict[str, str]]:
    return SAMPLE_DOCUMENTS


@asset
def sference_batch_id(source_documents: list[dict[str, str]]) -> str:
    requests = [
        InferenceRequest.chat(
            custom_id=doc["id"],
            user_content=f"Document:\n{doc['text']}",
            system_content=SYSTEM,
            model=model_id(),
        )
        for doc in source_documents
    ]
    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    return batch.id


@asset
def enriched_documents(
    source_documents: list[dict[str, str]],
    sference_batch_id: str,
) -> list[dict[str, Any]]:
    terminal = client.wait_for_completion(
        sference_batch_id,
        poll_interval=BATCH_POLL_INTERVAL_S,
        timeout=BATCH_WAIT_TIMEOUT_S,
    )
    print(f"Batch {sference_batch_id} finished with status={terminal.status}")
    if terminal.status != "completed":
        raise RuntimeError(f"Batch {sference_batch_id} ended as {terminal.status}")

    by_id = client.get_results_indexed(sference_batch_id)

    enriched: list[dict[str, Any]] = []
    for doc in source_documents:
        row = by_id.get(doc["id"])
        enriched.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata_line": row.completion_text if row else "",
                "status": row.status if row else None,
            }
        )
    return enriched


defs = Definitions(assets=[source_documents, sference_batch_id, enriched_documents])


if __name__ == "__main__":
    require_api_key()
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    result = materialize([source_documents, sference_batch_id, enriched_documents])
    enriched = result.output_for_node("enriched_documents")
    print("\nEnriched documents:")
    for row in enriched:
        print(f"  {row['id']}: {row['metadata_line']}")
