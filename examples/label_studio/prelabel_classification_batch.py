# ---
# Label Studio: pre-label text classification tasks with a sference batch.
#
# Reads a minimal LS export JSON, writes predictions JSON for import.
#
# Run:
#   uv run python examples/label_studio/prelabel_classification_batch.py
# ---

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

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

HERE = Path(__file__).resolve().parent
TASKS_PATH = HERE / "sample_tasks.json"
PREDICTIONS_PATH = HERE / "sample_predictions.json"

LABELS = ["billing", "shipping", "product_quality", "account"]
SYSTEM = (
    "Classify the customer message into exactly one label: "
    + ", ".join(LABELS)
    + ". Reply with the label only."
)


def load_tasks() -> list[dict[str, Any]]:
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def build_prediction(task: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "data": task["data"],
        "predictions": [
            {
                "model_version": f"sference-batch/{model_id()}",
                "score": 0.5,
                "result": [
                    {
                        "from_name": "topic",
                        "to_name": "text",
                        "type": "choices",
                        "value": {"choices": [label]},
                    }
                ],
            }
        ],
    }


def main() -> None:
    require_api_key()
    tasks = load_tasks()
    print(f"Loaded {len(tasks)} Label Studio tasks from {TASKS_PATH.name}\n")

    requests = [
        InferenceRequest.chat(
            custom_id=str(task["id"]),
            user_content=f"Message:\n{task['data']['text']}",
            system_content=SYSTEM,
            model=model_id(),
            temperature=0,
        )
        for task in tasks
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

    predictions: list[dict[str, Any]] = []
    for task in tasks:
        row = by_id.get(str(task["id"]))
        raw = (row.completion_text if row else "").strip().lower()
        label = raw if raw in LABELS else "account"
        predictions.append(build_prediction(task, label))

    PREDICTIONS_PATH.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Wrote {len(predictions)} pre-annotations → {PREDICTIONS_PATH}")
    print("Import in Label Studio: Project → Import → upload sample_predictions.json")


if __name__ == "__main__":
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    main()
