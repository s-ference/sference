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

import sys
from pathlib import Path
from typing import Any

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from dagster import AssetExecutionContext, Definitions, asset, materialize

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
        chat_batch_request(
            custom_id=doc["id"],
            user_content=f"Document:\n{doc['text']}",
            system_content=SYSTEM,
        )
        for doc in source_documents
    ]
    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    return batch.id


@asset
def enriched_documents(
    context: AssetExecutionContext,
    source_documents: list[dict[str, str]],
    sference_batch_id: str,
) -> list[dict[str, Any]]:
    terminal = wait_for_batch_terminal(client, sference_batch_id)
    context.log.info("Batch %s finished with status=%s", sference_batch_id, terminal.status)
    if terminal.status != "completed":
        raise RuntimeError(f"Batch {sference_batch_id} ended as {terminal.status}")

    payload = client.get_results(sference_batch_id)
    by_id = index_results_by_custom_id(payload.results)

    enriched: list[dict[str, Any]] = []
    for doc in source_documents:
        row = by_id.get(doc["id"], {})
        enriched.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "metadata_line": completion_text_from_row(row),
                "status": row.get("status"),
            }
        )
    return enriched


defs = Definitions(assets=[source_documents, sference_batch_id, enriched_documents])


if __name__ == "__main__":
    require_api_key()
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    result = materialize([enriched_documents])
    enriched = result.output_for_node("enriched_documents")
    print("\nEnriched documents:")
    for row in enriched:
        print(f"  {row['id']}: {row['metadata_line']}")
