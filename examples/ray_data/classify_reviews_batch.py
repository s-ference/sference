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

import os

# Ray reads this at import time, so it must be set before `import ray` below.
# It disables uv-run worker env propagation, which otherwise makes
# `uv run python examples/ray_data/classify_reviews_batch.py` hang while Ray
# tries to replicate the driver's uv environment to workers for this demo.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

import sys

import pandas as pd
import ray
from ray.data import from_pandas

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
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="error")

    ds = from_pandas(REVIEWS)
    print(f"Ray dataset: {ds.count()} rows\n")

    # Ray handles partitioning / scale-out preprocessing; inference is one batch job.
    rows = ds.take_all()
    requests = [
        InferenceRequest.chat(
            custom_id=str(row["review_id"]),
            user_content=f"Review:\n{row['text']}",
            system_content=SYSTEM,
            model=model_id(),
            temperature=0,
        )
        for row in rows
    ]

    print(f"Submitting batch ({len(requests)} rows, window={COMPLETION_WINDOW!r})...")
    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = client.wait_for_completion(
        batch.id,
        poll_interval=BATCH_POLL_INTERVAL_S,
        timeout=BATCH_WAIT_TIMEOUT_S,
    )
    print(f"Batch {batch.id} → {terminal.status}\n")

    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = client.get_results_indexed(batch.id)

    enriched = []
    for row in rows:
        result_row = by_id.get(str(row["review_id"]))
        enriched.append(
            {
                **row,
                "sentiment": (result_row.completion_text if result_row else "").strip().lower(),
                "inference_status": result_row.status if result_row else None,
            }
        )

    result_ds = from_pandas(pd.DataFrame(enriched))
    print("Results:")
    print(result_ds.take_all())
    ray.shutdown()


if __name__ == "__main__":
    print(f"Model: {model_id()}")
    main()
