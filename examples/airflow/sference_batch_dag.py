# ---
# Airflow TaskFlow: submit sference batch, wait, publish enriched JSON artifact.
#
# Extra install:
#   uv pip install "apache-airflow>=3.0"
#
# Run locally (no scheduler):
#   uv run python examples/airflow/sference_batch_dag.py
# ---

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow.sdk import dag, task

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

PRODUCT_BLURBS = [
    {"sku": "sku-a", "description": "Waterproof trail running shoe, wide toe box."},
    {"sku": "sku-b", "description": "Ceramic pour-over kettle with gooseneck spout."},
    {"sku": "sku-c", "description": "USB-C hub with 2.5GbE and SD card reader."},
]

SYSTEM = "Write a 12-word marketing tagline. No quotes."
ARTIFACT_PATH = Path("/tmp/sference_airflow_taglines.json")


def build_request_dicts() -> list[dict[str, Any]]:
    return [
        InferenceRequest.chat(
            custom_id=item["sku"],
            user_content=f"Product:\n{item['description']}",
            system_content=SYSTEM,
            model=model_id(),
        ).model_dump()
        for item in PRODUCT_BLURBS
    ]


def submit_batch_from_dicts(request_dicts: list[dict[str, Any]]) -> str:
    requests = [InferenceRequest.model_validate(r) for r in request_dicts]
    batch = client.submit_batch(requests=requests, window=COMPLETION_WINDOW)
    return batch.id


def enrich_taglines(batch_id: str) -> list[dict[str, str]]:
    terminal = client.wait_for_completion(
        batch_id,
        poll_interval=BATCH_POLL_INTERVAL_S,
        timeout=BATCH_WAIT_TIMEOUT_S,
    )
    if terminal.status != "completed":
        raise RuntimeError(f"Batch {batch_id} ended as {terminal.status}")

    by_id = client.get_results_indexed(batch_id)
    return [
        {
            "sku": item["sku"],
            "tagline": (by_id[item["sku"]].completion_text if item["sku"] in by_id else ""),
        }
        for item in PRODUCT_BLURBS
    ]


def write_tagline_artifact(rows: list[dict[str, str]]) -> str:
    ARTIFACT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return str(ARTIFACT_PATH)


@dag(
    dag_id="sference_product_taglines_batch",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["sference", "batch"],
)
def sference_product_taglines_batch():
    @task
    def build_requests() -> list[dict[str, Any]]:
        return build_request_dicts()

    @task
    def submit_batch_task(request_dicts: list[dict[str, Any]]) -> str:
        return submit_batch_from_dicts(request_dicts)

    @task
    def wait_and_enrich(batch_id: str) -> list[dict[str, str]]:
        return enrich_taglines(batch_id)

    @task
    def write_artifact(rows: list[dict[str, str]]) -> str:
        return write_tagline_artifact(rows)

    serialized = build_requests()
    batch_id = submit_batch_task(serialized)
    enriched = wait_and_enrich(batch_id)
    write_artifact(enriched)


dag = sference_product_taglines_batch()


if __name__ == "__main__":
    require_api_key()
    print(f"Model: {model_id()}  window: {COMPLETION_WINDOW}")
    request_dicts = build_request_dicts()
    batch_id = submit_batch_from_dicts(request_dicts)
    enriched = enrich_taglines(batch_id)
    write_tagline_artifact(enriched)
    print(f"Wrote {ARTIFACT_PATH}")
