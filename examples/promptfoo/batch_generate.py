# ---
# Offline batch for promptfoo: submit the whole eval as ONE sference batch, then write
# the precomputed outputs into a promptfoo tests file so promptfoo can score them with
# its built-in `echo` provider — no live calls during the eval. This is the clean way to
# get batch cost/throughput without pushing batching into a (synchronous) provider.
#
# Run:
#   export SFERENCE_API_KEY=sk_...
#   uv run python examples/promptfoo/batch_generate.py                       # writes batch_tests.json
#   npx promptfoo@latest eval -c examples/promptfoo/promptfooconfig.batch.yaml   # scores it
# ---

from __future__ import annotations

import json
import os
import sys

from sference_sdk import SferenceClient
from sference_sdk.models import InferenceRequest

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
OUT_PATH = os.environ.get(
    "BATCH_TESTS_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_tests.json"),
)

PROMPT = (
    "Classify the sentiment of the review as POSITIVE, NEGATIVE, or NEUTRAL.\n"
    "Reply with a single word.\n\nReview: {review}"
)
# Each eval case: (review, expected sentiment for the assertion).
CASES = [
    ("Absolutely love it, fast and reliable.", "positive"),
    ("Terrible experience, it broke on day one.", "negative"),
    ("It is fine, nothing special either way.", "neutral"),
]

client = SferenceClient()


def require_api_key() -> None:
    if not os.getenv("SFERENCE_API_KEY"):
        print("Error: SFERENCE_API_KEY is not set", file=sys.stderr)
        sys.exit(1)


def model_id() -> str:
    return os.environ.get("SFERENCE_MODEL", DEFAULT_MODEL)


def main() -> None:
    require_api_key()
    model = model_id()

    # Build rows in the RESPONSES shape (`input` + `enable_thinking`), not chat
    # (`messages`): the API drops unknown keys on chat rows, so `enable_thinking` would
    # be lost and reasoning models would return empty `content`.
    requests = [
        InferenceRequest(
            custom_id=str(i),
            body={
                "model": model,
                "input": [{"role": "user", "content": PROMPT.format(review=review)}],
                "enable_thinking": False,
                "max_output_tokens": 256,
            },
        )
        for i, (review, _) in enumerate(CASES)
    ]

    batch = client.submit_batch(requests=[r.model_dump() for r in requests], window="15m")
    print(f"Submitted batch {batch.id} ({len(requests)} rows); waiting ...")
    client.wait_for_completion(batch.id, poll_interval=5.0, timeout=3600.0)
    indexed = client.get_results_indexed(batch.id)

    # promptfoo tests file: the precomputed completion becomes the `output` var, scored
    # by the echo provider (see promptfooconfig.batch.yaml).
    tests = [
        {
            "vars": {"review": review, "output": indexed[str(i)].completion_text},
            "assert": [{"type": "icontains", "value": label}],
        }
        for i, (review, label) in enumerate(CASES)
    ]
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(tests, fh, indent=2)
    print(f"Wrote {len(tests)} precomputed rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
