# ---
# Hugging Face Datasets: map split rows to a sference batch, add a completion column.
#
# Install:
#   uv sync --group dev --group examples
#   export SFERENCE_API_KEY=sk_...
#
# Run:
#   uv run python examples/hf_datasets/map_splits_to_batch.py
# ---

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from datasets import Dataset

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

SAMPLE = Dataset.from_dict(
    {
        "id": ["ex-1", "ex-2", "ex-3"],
        "passage": [
            "Photosynthesis converts light energy into chemical energy in plants.",
            "The treaty established a demilitarized zone along the river.",
            "A mutex prevents two threads from entering a critical section at once.",
        ],
    }
)

SYSTEM = "Write one exam-style question about the passage. No answer."


def main() -> None:
    require_api_key()
    print(f"Dataset: {SAMPLE}")
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}\n")

    requests = [
        chat_batch_request(
            custom_id=row["id"],
            user_content=f"Passage:\n{row['passage']}",
            system_content=SYSTEM,
        )
        for row in SAMPLE
    ]

    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = wait_for_batch_terminal(client, batch.id)
    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = index_results_by_custom_id(client.get_results(batch.id).results)

    def attach_question(example: dict) -> dict:
        row = by_id.get(example["id"], {})
        return {"question": completion_text_from_row(row)}

    enriched = SAMPLE.map(attach_question)
    print("Enriched dataset:")
    for row in enriched:
        print(f"  [{row['id']}] {row['question'][:120]}...")


if __name__ == "__main__":
    main()
