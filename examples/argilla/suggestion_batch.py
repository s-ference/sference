# ---
# Argilla: generate text suggestions for human feedback records via sference batch.
#
# Extra install to push records to a server:
#   uv pip install argilla
#
# Run (writes local JSON either way):
#   uv run python examples/argilla/suggestion_batch.py
# ---

from __future__ import annotations

import json
import os
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
RECORDS_PATH = HERE / "sample_records.json"
SUGGESTIONS_PATH = HERE / "sample_suggestions.json"

SYSTEM = (
    "You are drafting a concise, empathetic support reply. "
    "Write 2 sentences maximum. Do not invent order numbers."
)


def main() -> None:
    require_api_key()
    records: list[dict[str, Any]] = json.loads(RECORDS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} Argilla-style records\n")

    requests = [
        chat_batch_request(
            custom_id=record["id"],
            user_content=f"Customer message:\n{record['fields']['message']}",
            system_content=SYSTEM,
        )
        for record in records
    ]

    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    terminal = wait_for_batch_terminal(client, batch.id)
    if terminal.status != "completed":
        raise RuntimeError(f"Batch ended as {terminal.status}")

    by_id = index_results_by_custom_id(client.get_results(batch.id).results)

    suggestions = []
    for record in records:
        row = by_id.get(record["id"], {})
        suggestions.append(
            {
                "id": record["id"],
                "fields": record["fields"],
                "suggestion": completion_text_from_row(row),
                "model": model_id(),
            }
        )

    SUGGESTIONS_PATH.write_text(json.dumps(suggestions, indent=2), encoding="utf-8")
    print(f"Wrote suggestions → {SUGGESTIONS_PATH}")

    if os.getenv("ARGILLA_API_URL") and os.getenv("ARGILLA_API_KEY"):
        push_to_argilla(suggestions)
    else:
        print("Set ARGILLA_API_URL and ARGILLA_API_KEY to push suggestions to a live dataset.")


def push_to_argilla(suggestions: list[dict[str, Any]]) -> None:
    import argilla as rg

    dataset_name = os.environ.get("ARGILLA_DATASET", "sference-support-suggestions")
    workspace = os.environ.get("ARGILLA_WORKSPACE", "admin")

    client_rg = rg.Argilla(api_url=os.environ["ARGILLA_API_URL"], api_key=os.environ["ARGILLA_API_KEY"])
    try:
        dataset = client_rg.datasets(name=dataset_name, workspace=workspace)
    except Exception:
        settings = rg.Settings(
            guidelines="Review AI-drafted replies before sending to customers.",
            fields=[rg.TextField(name="message"), rg.TextField(name="draft_reply")],
            questions=[rg.TextQuestion(name="approved_reply", title="Approved reply")],
        )
        dataset = client_rg.datasets.create(name=dataset_name, settings=settings, workspace=workspace)

    records = [
        rg.Record(
            id=item["id"],
            fields={"message": item["fields"]["message"], "draft_reply": item["suggestion"]},
            suggestions=[
                rg.Suggestion(
                    question_name="approved_reply",
                    value=item["suggestion"],
                    agent=model_id(),
                )
            ],
        )
        for item in suggestions
    ]
    dataset.records.log(records)
    print(f"Pushed {len(records)} records to Argilla dataset {dataset_name!r}")


if __name__ == "__main__":
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    main()
