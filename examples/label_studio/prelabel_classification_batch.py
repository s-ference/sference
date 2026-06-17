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
import sys
from pathlib import Path
from typing import Any

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

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
        chat_batch_request(
            custom_id=str(task["id"]),
            user_content=f"Message:\n{task['data']['text']}",
            system_content=SYSTEM,
        )
        for task in tasks
    ]

    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = wait_for_batch_terminal(client, batch.id)
    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = index_results_by_custom_id(client.get_results(batch.id).results)

    predictions: list[dict[str, Any]] = []
    for task in tasks:
        row = by_id.get(str(task["id"]), {})
        raw = completion_text_from_row(row).strip().lower()
        label = raw if raw in LABELS else "account"
        predictions.append(build_prediction(task, label))

    PREDICTIONS_PATH.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    print(f"Wrote {len(predictions)} pre-annotations → {PREDICTIONS_PATH}")
    print("Import in Label Studio: Project → Import → upload sample_predictions.json")


if __name__ == "__main__":
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    main()
