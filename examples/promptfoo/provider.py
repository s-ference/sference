# ---
# promptfoo custom provider that evaluates prompts on a sference-hosted model, one
# synchronous `POST /v1/responses` per row. promptfoo renders each prompt + test case
# and calls `call_api(prompt, options, context)`; this provider forwards the rendered
# prompt and returns the completion, so promptfoo's concurrency parallelizes the calls.
#
# For very large offline sweeps, generate the outputs with a `submit_batch` script
# separately and have promptfoo score the precomputed results (see the README).
#
# Wire it up from promptfooconfig.yaml:
#   providers:
#     - id: file://provider.py
#       config: { temperature: 0, enable_thinking: false }
#
# Install:
#   uv sync --group dev --group examples
#   export SFERENCE_API_KEY=sk_...
#
# Run the eval (needs Node / npx; point promptfoo at uv's venv — see README):
#   export PROMPTFOO_PYTHON="$PWD/.venv/bin/python3"
#   npx promptfoo@latest eval -c examples/promptfoo/promptfooconfig.yaml
#
# Run the standalone self-check (no Node, exercises call_api directly):
#   uv run python examples/promptfoo/provider.py
# ---

from __future__ import annotations

import os
import sys
from typing import Any

from sference_sdk import SferenceClient
from sference_sdk.models import Response

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
# Reasoning models put short answers in `reasoning_content` and leave `content` empty,
# so thinking is OFF by default (override per-provider via config.enable_thinking).
ENABLE_THINKING = os.environ.get("SFERENCE_ENABLE_THINKING", "false").lower() == "true"
MAX_OUTPUT_TOKENS = int(os.environ.get("SFERENCE_MAX_OUTPUT_TOKENS", "256"))

client = SferenceClient()  # reused across promptfoo's many call_api invocations


def require_api_key() -> None:
    if not os.getenv("SFERENCE_API_KEY"):
        print("Error: SFERENCE_API_KEY is not set", file=sys.stderr)
        sys.exit(1)


def model_id() -> str:
    return os.environ.get("SFERENCE_MODEL", DEFAULT_MODEL)


def _response_text(response: Response) -> str:
    """Concatenate the assistant's output_text parts (skips reasoning items)."""
    parts: list[str] = []
    for item in response.output or []:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text" and part.text:
                    parts.append(part.text)
    return "\n".join(parts)


def call_api(
    prompt: str,
    options: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """promptfoo provider entrypoint. Returns {"output": str} or {"error": str}."""
    config = (options or {}).get("config") or {}
    model = config.get("model") or model_id()

    try:
        response = client.create_response(
            model=model,
            input=[{"role": "user", "content": prompt}],
            instructions=config.get("instructions"),
            temperature=config.get("temperature"),
            top_p=config.get("top_p"),
            max_output_tokens=config.get("max_output_tokens", MAX_OUTPUT_TOKENS),
            enable_thinking=config.get("enable_thinking", ENABLE_THINKING),
        )
    except Exception as exc:  # surface transport/auth errors to promptfoo
        return {"error": str(exc)}

    if response.status != "completed":
        return {"error": f"response ended as {response.status}"}

    result: dict[str, Any] = {"output": _response_text(response)}
    if response.usage:
        result["tokenUsage"] = {
            "prompt": response.usage.input_tokens,
            "completion": response.usage.output_tokens,
            "total": response.usage.total_tokens,
        }
    return result


# A tiny copy of promptfooconfig.yaml's prompt + cases so the provider is runnable
# (and testable) without a Node/promptfoo install.
_SELF_CHECK_PROMPT = (
    "Classify the sentiment of the review as POSITIVE, NEGATIVE, or NEUTRAL.\n"
    "Reply with a single word.\n\nReview: {review}"
)
_SELF_CHECK_CASES = (
    {"label": "positive-review", "review": "Absolutely love it, fast and reliable."},
    {"label": "negative-review", "review": "Terrible experience, it broke on day one."},
    {"label": "neutral-review", "review": "It is fine, nothing special either way."},
)


def _self_check() -> None:
    require_api_key()

    model = model_id()
    print(f"Sference promptfoo provider — local self-check ({len(_SELF_CHECK_CASES)} cases)")
    print(f"Model: {model}\n")

    produced = 0
    for case in _SELF_CHECK_CASES:
        prompt = _SELF_CHECK_PROMPT.format(review=case["review"])
        result = call_api(prompt, options={"config": {"model": model, "temperature": 0}})
        if "error" in result:
            print(f"[{case['label']}] ERROR: {result['error']}")
            continue
        produced += 1
        print(f"[{case['label']}] {result['output']}")

    print(f"\nSelf-check complete: {produced}/{len(_SELF_CHECK_CASES)} produced output")
    if produced != len(_SELF_CHECK_CASES):
        sys.exit(1)


if __name__ == "__main__":
    _self_check()
