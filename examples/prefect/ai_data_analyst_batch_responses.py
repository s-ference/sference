# ---
# Why this pattern (vs realtime / in-process LLM agents)
# ---
#
# Adapted from Prefect's "AI-Powered Data Analyst" example:
#   https://github.com/PrefectHQ/prefect/blob/main/examples/ai_data_analyst_with_pydantic_ai.py
#
# The original keeps the model in the hot path: every tool call and LLM round-trip
# blocks the flow, bills GPU time while you orchestrate, and retries duplicate
# expensive inference unless you add your own idempotency.
#
# This example instead uses sference **batch responses** (`POST /v1/responses` with
# `background=true` and `metadata.completion_window="1h"`):
#
# - **Submit then wait** — Stage 1 only enqueues work and records response IDs; Stage 2
#   polls until terminal. Prefect owns orchestration, retries, and run history; the SDK
#   hides poll loops (`create_response`, `wait_for_response`).
# - **Cheaper at scale** — Inference runs on the platform scheduler, not inside your
#   Prefect worker. Workers stay thin (HTTP + state), so you can scale orchestration
#   separately from GPU capacity.
# - **Durable stages** — If Stage 2 fails mid-poll, Prefect can retry waiting without
#   re-submitting jobs that already have an ID. Map each prompt to its own tasks for
#   per-item observability in the UI.
# - **Same SLA semantics as production** — `1h` completion window matches how customers
#   run large, non-interactive analysis workloads on sference.
#
# Install (from the oss repo root):
#   uv sync --group dev --group examples
#   export SFERENCE_API_KEY=sk_...
#
# Run locally:
#   uv run python examples/prefect/ai_data_analyst_batch_responses.py
#
# Deploy for durable runs (Prefect UI / workers):
#   uv run python examples/prefect/ai_data_analyst_batch_responses.py --serve
# ---

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd
from prefect import flow, task

from sference_sdk import SferenceClient

COMPLETION_WINDOW = "1h"
DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"

# One shared client for the whole example (reads SFERENCE_API_KEY).
client = SferenceClient()


@dataclass(frozen=True)
class AnalysisPrompt:
    custom_id: str
    user_content: str


@task(name="prepare-sample-dataset")
def prepare_sample_dataset() -> pd.DataFrame:
    """Small sales table with deliberate outliers (demo input)."""
    data = {
        "product": ["Widget", "Gadget", "Doohickey", "Widget", "Gadget"] * 20,
        "sales": [100, 150, 200, 110, 145] * 19 + [100, 150, 200, 1000, 2000],
        "region": ["North", "South", "East", "West", "Central"] * 20,
        "month": [1, 2, 3, 4, 5] * 20,
    }
    return pd.DataFrame(data)


@task(name="build-analysis-prompts")
def build_analysis_prompts(df: pd.DataFrame) -> list[AnalysisPrompt]:
    """Turn local pandas stats into LLM prompts (no inference in this task)."""
    sales = df["sales"]
    summary = {
        "rows": len(df),
        "sales_mean": float(sales.mean()),
        "sales_std": float(sales.std()),
        "sales_max": float(sales.max()),
        "regions": sorted(df["region"].unique().tolist()),
    }
    return [
        AnalysisPrompt(
            custom_id="dataset-overview",
            user_content=(
                "You are a data analyst. Given this dataset summary JSON, write a short "
                f"executive summary (3–5 sentences). Summary: {summary!r}"
            ),
        ),
        AnalysisPrompt(
            custom_id="anomaly-hunt",
            user_content=(
                "You are a data analyst. The sales column may contain outliers. "
                f"Using mean={summary['sales_mean']:.1f} and std={summary['sales_std']:.1f}, "
                "explain which kinds of rows are likely anomalies and what to investigate next. "
                "Reply in bullet points."
            ),
        ),
        AnalysisPrompt(
            custom_id="recommendations",
            user_content=(
                "You are a data analyst. Regions in the dataset: "
                f"{summary['regions']}. "
                "Suggest 3 actionable follow-up analyses (not generic advice)."
            ),
        ),
    ]


@task(name="create-response-request", retries=2, retry_delay_seconds=5)
def create_response_request(prompt: AnalysisPrompt) -> str:
    """Stage 1 — enqueue one background response; return its id."""
    created = client.create_response(
        model=os.environ.get("SFERENCE_MODEL", DEFAULT_MODEL),
        input=[{"role": "user", "content": prompt.user_content}],
        background=True,
        metadata={
            "completion_window": COMPLETION_WINDOW,
            "custom_id": prompt.custom_id,
        },
    )
    return created.id


@task(name="wait-for-response-completion", retries=2, retry_delay_seconds=10)
def wait_for_response_completion(response_id: str) -> dict[str, Any]:
    """Stage 2 — wait until terminal; the SDK handles polling internally."""
    done = client.wait_for_response(response_id)
    text_parts: list[str] = []
    if done.output:
        for item in done.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "output_text" and part.text:
                        text_parts.append(part.text)
    return {
        "id": done.id,
        "status": done.status,
        "model": done.model,
        "text": "\n".join(text_parts),
        "usage": done.usage.model_dump() if done.usage else None,
        "error": done.error.model_dump() if done.error else None,
    }


@task(name="print-analysis-report")
def print_analysis_report(
    results: list[dict[str, Any]],
    prompts: list[AnalysisPrompt],
) -> None:
    print("\n" + "=" * 80)
    print("BATCH RESPONSE ANALYSIS (1h completion window)")
    print("=" * 80)
    for prompt, row in zip(prompts, results, strict=True):
        print(f"\n--- {prompt.custom_id} ({row['status']}) ---")
        if row["error"]:
            print(f"Error: {row['error']}")
        else:
            print(row["text"] or "(no text output)")
    print("=" * 80 + "\n")


@flow(name="ai-data-analyst-batch-responses", log_prints=True)
def analyze_dataset_with_batch_responses() -> list[dict[str, Any]]:
    """Two-stage Prefect flow: submit background responses, then wait for completion."""
    print("Preparing dataset and prompts...")
    df = prepare_sample_dataset()
    prompts = build_analysis_prompts(df)
    print(f"Dataset shape: {df.shape}; {len(prompts)} LLM jobs to enqueue.\n")

    print(f"Stage 1 — create background responses (completion_window={COMPLETION_WINDOW!r})...")
    # .map() fans out one task per prompt and returns futures (not values yet).
    response_id_futures = create_response_request.map(prompts)
    print(f"Enqueued {len(prompts)} background response(s).\n")

    print("Stage 2 — wait for completions (SDK polling)...")
    # Futures chain directly into the next mapped task without blocking here.
    result_futures = wait_for_response_completion.map(response_id_futures)

    # Resolve futures to dicts in the flow (map preserves prompt order).
    ordered = [future.result() for future in result_futures]
    print_analysis_report(ordered, prompts)

    failed = [r for r in ordered if r["status"] != "completed"]
    if failed:
        raise RuntimeError(f"{len(failed)} response(s) did not complete successfully")

    return ordered


if __name__ == "__main__":
    if not os.getenv("SFERENCE_API_KEY"):
        print("Error: SFERENCE_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    if "--serve" in sys.argv:
        analyze_dataset_with_batch_responses.serve(
            name="ai-data-analyst-batch-responses",
            tags=["sference", "batch-responses", "1h", "prefect"],
        )
    else:
        analyze_dataset_with_batch_responses()
