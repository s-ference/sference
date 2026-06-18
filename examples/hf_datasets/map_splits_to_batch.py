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

import os
import sys

from datasets import Dataset

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
        InferenceRequest.chat(
            custom_id=row["id"],
            user_content=f"Passage:\n{row['passage']}",
            system_content=SYSTEM,
            model=model_id(),
        )
        for row in SAMPLE
    ]

    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = client.wait_for_completion(
        batch.id,
        poll_interval=BATCH_POLL_INTERVAL_S,
        timeout=BATCH_WAIT_TIMEOUT_S,
    )
    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = client.get_results_indexed(batch.id)

    def attach_question(example: dict) -> dict:
        row = by_id.get(example["id"])
        return {"question": row.completion_text if row else ""}

    enriched = SAMPLE.map(attach_question)
    print("Enriched dataset:")
    for row in enriched:
        print(f"  [{row['id']}] {row['question'][:120]}...")


if __name__ == "__main__":
    main()
