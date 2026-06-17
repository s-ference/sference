# ---
# Ray Data: partition rows locally, submit one sference batch, join completions back.
#
# Install:
#   uv sync --group dev --group examples
#   export SFERENCE_API_KEY=sk_...
#
# Run:
#   uv run python examples/ray_data/classify_reviews_batch.py
# ---

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import pandas as pd
import ray
from ray.data import from_pandas

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

REVIEWS = pd.DataFrame(
    {
        "review_id": ["r1", "r2", "r3", "r4"],
        "text": [
            "Battery life is amazing but the screen flickers sometimes.",
            "Shipped fast. Packaging was crushed yet the item works.",
            "Support ignored my ticket for two weeks.",
            "Best keyboard I have used in years.",
        ],
    }
)

SYSTEM = (
    "Classify product review sentiment. Reply with exactly one word: "
    "positive, negative, or mixed."
)


def main() -> None:
    require_api_key()
    ray.init(ignore_reinit_error=True)

    ds = from_pandas(REVIEWS)
    print(f"Ray dataset: {ds.count()} rows\n")

    # Ray handles partitioning / scale-out preprocessing; inference is one batch job.
    rows = ds.take_all()
    requests = [
        chat_batch_request(
            custom_id=str(row["review_id"]),
            user_content=f"Review:\n{row['text']}",
            system_content=SYSTEM,
        )
        for row in rows
    ]

    print(f"Submitting batch ({len(requests)} rows, window={COMPLETION_WINDOW!r})...")
    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = wait_for_batch_terminal(client, batch.id)
    print(f"Batch {batch.id} → {terminal.status}\n")

    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    payload = client.get_results(batch.id)
    by_id = index_results_by_custom_id(payload.results)

    enriched = []
    for row in rows:
        result_row = by_id.get(str(row["review_id"]), {})
        enriched.append(
            {
                **row,
                "sentiment": completion_text_from_row(result_row).strip().lower(),
                "inference_status": result_row.get("status"),
            }
        )

    result_ds = from_pandas(pd.DataFrame(enriched))
    print("Results:")
    print(result_ds.take_all())
    ray.shutdown()


if __name__ == "__main__":
    print(f"Model: {model_id()}")
    main()
