# Label Studio + sference batch (pre-labeling)

[`prelabel_classification_batch.py`](prelabel_classification_batch.py) automates **offline pre-annotations**:

1. Load [`sample_tasks.json`](sample_tasks.json) (minimal Label Studio export).
2. Submit one batch row per task (`custom_id` = task `id`).
3. Map completions to Label Studio prediction JSON → [`sample_predictions.json`](sample_predictions.json).

Annotators review suggestions in the UI instead of starting from scratch.

## Label config (example)

Use a single-choice taxonomy named `topic` on field `text`:

```xml
<View>
  <Text name="text" value="$text"/>
  <Choices name="topic" toName="text" choice="single">
    <Choice value="billing"/>
    <Choice value="shipping"/>
    <Choice value="product_quality"/>
    <Choice value="account"/>
  </Choices>
</View>
```

## Setup

```bash
uv sync --group dev --group examples
export SFERENCE_API_KEY=sk_...
```

## Run

```bash
uv run python examples/label_studio/prelabel_classification_batch.py
```

No running Label Studio server required for the demo script.
