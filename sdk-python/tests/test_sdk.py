import json
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import sference_sdk
from sference_sdk.client import ApiError, SferenceClient
from sference_sdk.models import InferenceRequest, reasoning_summary_plaintext

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_sdk_version_matches_metadata() -> None:
    from importlib.metadata import version

    assert sference_sdk.__version__ == version("sference-sdk")
    assert len(sference_sdk.__version__.split(".")) >= 2


def test_parse_openai_jsonl_format(tmp_path: Path) -> None:
    p = tmp_path / "openai.jsonl"
    p.write_text(
        '{"custom_id":"request-1","method":"POST","url":"/v1/chat/completions","body":{"model":"deepseek-ai/DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"hi"}]}}\n',
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="model argument is ignored"):
        parsed = SferenceClient.parse_inference_requests_jsonl(p, model="ignored")
    assert len(parsed) == 1
    assert parsed[0].custom_id == "request-1"


def test_parse_openai_jsonl_no_model_no_warning(tmp_path: Path) -> None:
    p = tmp_path / "openai.jsonl"
    p.write_text(
        '{"custom_id":"request-1","method":"POST","url":"/v1/chat/completions","body":{"model":"deepseek-ai/DeepSeek-V4.1-Flash-1017","messages":[{"role":"user","content":"hi"}]}}\n',
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        parsed = SferenceClient.parse_inference_requests_jsonl(p, model=None)
    assert len(w) == 0
    assert len(parsed) == 1


def test_parse_openai_jsonl_model_warns_once_per_file(tmp_path: Path) -> None:
    line = '{"custom_id":"r","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"x"}]}}\n'
    p = tmp_path / "openai.jsonl"
    p.write_text(line * 3, encoding="utf-8")
    with pytest.warns(UserWarning, match="model argument is ignored") as record:
        parsed = SferenceClient.parse_inference_requests_jsonl(p, model="extra")
    assert len(record) == 1
    assert len(parsed) == 3


def test_parse_content_only_jsonl_format(tmp_path: Path) -> None:
    p = tmp_path / "content.jsonl"
    p.write_text('{"content":"What is the capital of France?"}\n', encoding="utf-8")
    parsed = SferenceClient.parse_inference_requests_jsonl(
        p, model="deepseek-ai/DeepSeek-V4.1-Flash-1017"
    )
    assert len(parsed) == 1
    assert parsed[0].body["model"] == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert parsed[0].body["messages"][0]["content"] == "What is the capital of France?"


def test_parse_content_only_requires_model(tmp_path: Path) -> None:
    p = tmp_path / "content.jsonl"
    p.write_text('{"content":"x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="model is required"):
        SferenceClient.parse_inference_requests_jsonl(p, model=None)


def test_submit_batch_raises_api_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/batches":
            return httpx.Response(status_code=401, json={"detail": "Unauthorized"})
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with (
        SferenceClient(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ApiError, match="401"),
    ):
        client.submit_batch(
            requests=[
                {
                    "custom_id": "request-1",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                }
            ]
        )


def test_submit_batch_rejects_unknown_window() -> None:
    with (
        SferenceClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(500)), api_key="tok"
        ) as client,
        pytest.raises(ValueError, match='"24h"'),
    ):
        client.submit_batch(
            requests=[{"custom_id": "r1", "body": {"model": "m", "messages": []}}],
            window="2h",
        )


def test_submit_batch_sends_selected_window() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/batches":
            body = json.loads(request.content.decode("utf-8"))
            seen["window"] = body["window"]
            payload = json.loads(
                (FIXTURES / "V1BatchesCreateBatch" / "201.json").read_text(
                    encoding="utf-8"
                )
            )
            payload["window"] = body["window"]
            return httpx.Response(status_code=201, json=payload)
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        batch = client.submit_batch(
            requests=[{"custom_id": "r1", "body": {"model": "m", "messages": []}}],
            window="24h",
        )
        assert seen["window"] == "24h"
        assert batch.window == "24h"


def test_download_results_jsonl_via_httpx_mock_transport(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == "/v1/batches/batch_abc/results.jsonl"
        ):
            assert request.headers.get("authorization") == "Bearer tok"
            return httpx.Response(status_code=200, content=b'{"x": 1}\n{"x": 2}\n')
        return httpx.Response(status_code=404, json={"detail": "not found"})

    out_path = tmp_path / "out.jsonl"
    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        client.download_results_jsonl("batch_abc", out_path)
    assert out_path.read_text(encoding="utf-8").splitlines() == ['{"x": 1}', '{"x": 2}']


def test_get_response_parses_reasoning_output_item() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/responses/resp_1":
            return httpx.Response(
                status_code=200,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "created_at": 1712345678,
                    "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                    "status": "completed",
                    "output": [
                        {
                            "type": "reasoning",
                            "id": "rs_reasoning_0",
                            "summary": [
                                {"type": "summary_text", "text": "internal reasoning"}
                            ],
                        },
                        {
                            "type": "message",
                            "id": "msg_0",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "The answer is 42.",
                                    "annotations": [],
                                }
                            ],
                        },
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        resp = client.get_response("resp_1")
        assert resp.status == "completed"
        assert resp.output is not None
        assert resp.output[0].type == "reasoning"
        assert reasoning_summary_plaintext(resp.output[0]) == "internal reasoning"
        assert resp.output[1].type == "message"
        assert resp.output[1].content[0].text == "The answer is 42."


_TOOLS = [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]


@pytest.mark.parametrize(
    ("options", "body", "idempotency_key"),
    [
        # A bare string is forwarded as-is; the server expands it to a user message.
        ({"input": "grade this"}, {"input": "grade this"}, None),
        # Falsy settings are sent, not dropped as if unset.
        (
            {"include_reasoning": False, "enable_thinking": True},
            {"include_reasoning": False, "enable_thinking": True},
            None,
        ),
        (
            {
                "tools": _TOOLS,
                "tool_choice": "auto",
                "reasoning": {"effort": "low"},
                "service_tier": "flex",
                "max_queue_wait_s": 60,
                "idempotency_key": "key-1",
            },
            {
                "tools": _TOOLS,
                "tool_choice": "auto",
                "reasoning": {"effort": "low"},
                "service_tier": "flex",
                "max_queue_wait_s": 60,
            },
            "key-1",
        ),
    ],
)
def test_create_response_sends_only_the_options_given(
    options: dict[str, Any], body: dict[str, Any], idempotency_key: str | None
) -> None:
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["json"] = json.loads(request.content)
        sent["idempotency_key"] = request.headers.get("idempotency-key")
        return httpx.Response(
            201,
            json={"id": "r", "created_at": 1, "model": "m", "status": "in_progress"},
        )

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        client.create_response(**{"model": "m", "input": "hi", **options})

    assert sent["json"] == {"model": "m", "input": "hi", **body}
    assert sent["idempotency_key"] == idempotency_key


def test_wait_for_response_treats_incomplete_as_terminal() -> None:
    """max_output_tokens truncation surfaces as status="incomplete"; the poll loop
    must parse it and stop instead of polling until timeout."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/responses/resp_trunc":
            calls["count"] += 1
            return httpx.Response(
                status_code=200,
                json={
                    "id": "resp_trunc",
                    "object": "response",
                    "created_at": 1712345678,
                    "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_0",
                            "role": "assistant",
                            "status": "incomplete",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "partial",
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        resp = client.wait_for_response("resp_trunc", poll_interval=0.01, timeout=1.0)
        assert calls["count"] == 1
        assert resp.status == "incomplete"
        assert resp.incomplete_details is not None
        assert resp.incomplete_details.reason == "max_output_tokens"


def test_create_embeddings_single_string_input() -> None:
    captured_json: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_json
        if request.method == "POST" and request.url.path == "/v1/embeddings":
            captured_json = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                status_code=200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [0.1, 0.2, 0.3],
                        }
                    ],
                    "model": captured_json.get("model", "m"),
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        resp = client.create_embeddings(model="Ettin/Ettin-Encoder-7B", input="hello")

    assert captured_json == {
        "model": "Ettin/Ettin-Encoder-7B",
        "input": "hello",
        "encoding_format": "float",
    }
    assert resp.data[0].embedding == [0.1, 0.2, 0.3]
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 4


@pytest.mark.parametrize(
    "images",
    [
        None,
        [
            "data:image/png;base64,eA==",
            {"content_type": "image/jpeg", "base64": "eA=="},
        ],
    ],
)
def test_create_decision_posts_questions_and_parses_answers(images) -> None:
    from sference_sdk.models import ChoiceQuestion

    captured_json: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_json
        if request.method == "POST" and request.url.path == "/v1/decisions":
            captured_json = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                status_code=200,
                json=json.loads(
                    (FIXTURES / "createDecision" / "200.json").read_text(
                        encoding="utf-8"
                    )
                ),
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        resp = client.create_decision(
            model="Cloudflare/clef",
            state="I was charged twice, please refund me",
            images=images,
            questions={
                "route": ChoiceQuestion(
                    criteria={"billing": "Payments", "support": "Everything else"}
                ),
                "urgency": {"type": "score", "criteria": ["Low", "High"]},
                "refund": {
                    "type": "noul",
                    "instructions": "Does the customer ask for a refund?",
                },
            },
        )

    assert captured_json["model"] == "Cloudflare/clef"
    if images is None:
        assert "images" not in captured_json
    else:
        assert captured_json["images"] == images
    assert captured_json["state"] == "I was charged twice, please refund me"
    assert captured_json["questions"]["route"]["type"] == "choice"
    assert captured_json["questions"]["urgency"]["criteria"] == ["Low", "High"]
    assert (
        captured_json["questions"]["refund"]["instructions"]
        == "Does the customer ask for a refund?"
    )
    assert resp.answers["route"].choice == "billing"
    assert resp.answers["urgency"].score == 0.75
    assert resp.answers["refund"].noul == 0.9
    assert resp.usage.input_tokens == 512


def test_create_decision_rejects_unknown_question_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not reach the API")

    with (
        SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client,
        pytest.raises(ValueError),
    ):
        client.create_decision(
            model="Cloudflare/clef", state="x", questions={"q": {"type": "rank"}}
        )


def test_checkpoint_roundtrip(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from sference_sdk.checkpoint import (
        clear_checkpoint,
        load_checkpoint,
        save_checkpoint,
    )

    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "cp.json"))
    assert load_checkpoint("http://127.0.0.1:8000", "sid", "c1") is None
    save_checkpoint("http://127.0.0.1:8000", "sid", "c1", "e1")
    assert load_checkpoint("http://127.0.0.1:8000", "sid", "c1") == "e1"
    clear_checkpoint("http://127.0.0.1:8000", "sid", "c1")
    assert load_checkpoint("http://127.0.0.1:8000", "sid", "c1") is None


def _completion_event_row(completion_id: str) -> dict:
    return {
        "completion_id": completion_id,
        "response_id": None,
        "custom_id": None,
        "status": "completed",
        "result": {},
        "error": None,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "completed_at": "2026-04-04T10:12:09+00:00",
    }


def test_iter_responses_events_uses_starting_after_from_saved_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sference_sdk.checkpoint import save_checkpoint

    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "cp.json"))
    base = "http://iter-checkpoint.local"
    saved_cursor = "019d58a7-2ece-7742-bc3e-69ba44168279"
    next_id = "019d58a7-2ece-7742-bc3e-69ba44168280"
    save_checkpoint(base, "__responses_events__", "consumer-a", saved_cursor)

    starting_after_log: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/responses/events":
            return httpx.Response(status_code=404, json={"detail": "not found"})
        params = dict(request.url.params)
        starting_after_log.append(params.get("starting_after"))
        assert params.get("starting_after") == saved_cursor
        body = {
            "object": "list",
            "data": [_completion_event_row(next_id)],
            "has_more": False,
        }
        return httpx.Response(status_code=200, json=body)

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok", base_url=base
    ) as client:
        events = list(
            client.iter_responses_events(consumer_name="consumer-a", checkpoint=True)
        )
    assert len(events) == 1
    assert events[0].completion_id == next_id
    assert starting_after_log == [saved_cursor]


def test_iter_responses_events_multipage_updates_checkpoint_to_last_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sference_sdk.checkpoint import load_checkpoint, save_checkpoint

    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "cp.json"))
    base = "http://iter-pages.local"
    c0 = "019d58a7-2ece-7742-bc3e-69ba44168277"
    c1 = "019d58a7-2ece-7742-bc3e-69ba44168278"
    c2 = "019d58a7-2ece-7742-bc3e-69ba44168279"
    save_checkpoint(base, "__responses_events__", "tail", c0)

    starting_after_log: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/responses/events":
            return httpx.Response(status_code=404, json={"detail": "not found"})
        params = dict(request.url.params)
        starting_after_log.append(params.get("starting_after"))
        sa = params.get("starting_after")
        if sa == c0:
            body = {
                "object": "list",
                "data": [_completion_event_row(c1)],
                "has_more": True,
            }
        elif sa == c1:
            body = {
                "object": "list",
                "data": [_completion_event_row(c2)],
                "has_more": False,
            }
        else:
            return httpx.Response(
                status_code=500, json={"detail": f"unexpected starting_after={sa!r}"}
            )
        return httpx.Response(status_code=200, json=body)

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok", base_url=base
    ) as client:
        events = list(
            client.iter_responses_events(consumer_name="tail", checkpoint=True)
        )
    assert [e.completion_id for e in events] == [c1, c2]
    assert starting_after_log == [c0, c1]
    assert load_checkpoint(base, "__responses_events__", "tail") == c2


async def test_iter_responses_events_async_same_multipage_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sference_sdk.async_client import AsyncSferenceClient
    from sference_sdk.checkpoint import load_checkpoint, save_checkpoint

    monkeypatch.setenv("SFERENCE_STREAM_CHECKPOINTS", str(tmp_path / "cp-async.json"))
    base = "http://iter-async.local"
    c0 = "019d58a7-2ece-7742-bc3e-69ba44168281"
    c1 = "019d58a7-2ece-7742-bc3e-69ba44168282"
    c2 = "019d58a7-2ece-7742-bc3e-69ba44168283"
    save_checkpoint(base, "__responses_events__", "async-c", c0)

    starting_after_log: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/responses/events":
            return httpx.Response(status_code=404, json={"detail": "not found"})
        params = dict(request.url.params)
        starting_after_log.append(params.get("starting_after"))
        sa = params.get("starting_after")
        if sa == c0:
            body = {
                "object": "list",
                "data": [_completion_event_row(c1)],
                "has_more": True,
            }
        elif sa == c1:
            body = {
                "object": "list",
                "data": [_completion_event_row(c2)],
                "has_more": False,
            }
        else:
            return httpx.Response(
                status_code=500, json={"detail": f"unexpected starting_after={sa!r}"}
            )
        return httpx.Response(status_code=200, json=body)

    async with AsyncSferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok", base_url=base
    ) as client:
        out: list[str] = []
        async for ev in client.iter_responses_events(
            consumer_name="async-c", checkpoint=True
        ):
            out.append(ev.completion_id)
    assert out == [c1, c2]
    assert starting_after_log == [c0, c1]
    assert load_checkpoint(base, "__responses_events__", "async-c") == c2


def test_inference_request_chat_builds_messages_body() -> None:
    req = InferenceRequest.chat(
        custom_id="row-1",
        user_content="hello",
        system_content="be brief",
        model="deepseek-ai/DeepSeek-V4.1-Flash-1017",
        temperature=0,
    )
    assert req.custom_id == "row-1"
    assert req.body["model"] == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert req.body["temperature"] == 0
    assert req.body["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_batch_result_row_completion_text() -> None:
    from sference_sdk.models import BatchResultRow

    ok = BatchResultRow(
        custom_id="a",
        status="completed",
        result_json={"choices": [{"message": {"content": "positive"}}]},
    )
    assert ok.completion_text == "positive"

    failed = BatchResultRow(custom_id="b", status="failed", error_json={"code": "x"})
    assert failed.completion_text == "[failed] {'code': 'x'}"


def test_batch_results_index_by_custom_id() -> None:
    from sference_sdk.models import BatchResults

    payload = BatchResults(
        batch_id="batch_1",
        status="completed",
        results=[
            {
                "custom_id": "doc-1",
                "status": "completed",
                "result_json": {"choices": []},
            },
            {
                "custom_id": "doc-2",
                "status": "completed",
                "result_json": {"choices": []},
            },
        ],
    )
    by_id = payload.index_by_custom_id()
    assert set(by_id) == {"doc-1", "doc-2"}
    assert by_id["doc-1"].custom_id == "doc-1"


def test_get_results_indexed_via_mock_transport() -> None:
    from sference_sdk.models import BatchResultRow

    def handler(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == "/v1/batches/batch_1/results"
        ):
            return httpx.Response(
                status_code=200,
                json={
                    "batch_id": "batch_1",
                    "status": "completed",
                    "output_url": None,
                    "completed_at": "2025-01-01T00:00:01+00:00",
                    "results": [
                        {
                            "custom_id": "r1",
                            "status": "completed",
                            "result_json": {
                                "choices": [{"message": {"content": "ok"}}]
                            },
                        }
                    ],
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        by_id = client.get_results_indexed("batch_1")
    assert isinstance(by_id["r1"], BatchResultRow)
    assert by_id["r1"].completion_text == "ok"


def test_set_api_key_swaps_the_credential_without_rebuilding() -> None:
    """A key rotated mid-session applies to subsequent requests on the same
    client, so callers keep their connection pool."""
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(
            status_code=200, json={"username": "dev", "role": "admin"}
        )

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="old"
    ) as client:
        client.get_me()
        client.set_api_key("new")
        client.get_me()

    assert seen == ["Bearer old", "Bearer new"]


def test_api_error_exposes_openai_error_fields() -> None:
    body = {
        "error": {
            "message": "Insufficient credits",
            "type": "billing_error",
            "param": None,
            "code": "insufficient_credits",
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json=body)

    with (
        SferenceClient(transport=httpx.MockTransport(handler), api_key="tok") as client,
        pytest.raises(ApiError) as excinfo,
    ):
        client.create_response(model="m", input="hi")

    err = excinfo.value
    assert (err.status_code, err.type, err.code, err.message) == (
        402,
        "billing_error",
        "insufficient_credits",
        "Insufficient credits",
    )
    assert err.body == body
    assert str(err).startswith("402: ")


def _frame(*events: dict[str, Any]) -> bytes:
    """Frame events the way the API does: ``event:`` + ``data:``, with keep-alive pings."""
    frames = [": ping\n\n"]
    frames += [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events]
    # The final frame drops its blank line to prove the decoder flushes at EOF.
    return "".join(frames).removesuffix("\n\n").encode()


_SHELL = {
    "id": "resp_1",
    "object": "response",
    "created_at": 1,
    "model": "m",
    "status": "in_progress",
    "output": [],
}
_STREAM_EVENTS = [
    {"type": "response.created", "sequence_number": 0, "response": _SHELL},
    {
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        },
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 2,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "Hel",
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 3,
        "item_id": "msg_1",
        "output_index": 0,
        "content_index": 0,
        "delta": "lo",
    },
    {
        "type": "response.completed",
        "sequence_number": 4,
        "response": {
            **_SHELL,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        },
    },
]


def test_stream_response_yields_typed_events_until_completed() -> None:
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["json"] = json.loads(request.content)
        sent["extra"] = request.headers.get("x-extra")
        return httpx.Response(
            200,
            content=_frame(*_STREAM_EVENTS),
            headers={"content-type": "text/event-stream"},
        )

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        with client.extra_headers({"x-extra": "1"}):
            stream = client.stream_response(model="m", input="hi")
        events = list(stream)

    assert sent["json"] == {"model": "m", "input": "hi", "stream": True}
    # Headers are bound when the call is made, not when iteration starts.
    assert sent["extra"] == "1"
    assert [e.sequence_number for e in events] == [0, 1, 2, 3, 4]
    assert "".join(e.delta or "" for e in events) == "Hello"
    final = events[-1].response
    assert final is not None and final.status == "completed" and final.usage is not None


def test_stream_response_raises_mid_stream_error_frame() -> None:
    # After HTTP 200 a failure can only arrive as an SSE ``error`` frame.
    body = _frame(
        {
            "type": "response.created",
            "response": {
                "id": "r",
                "created_at": 1,
                "model": "m",
                "status": "in_progress",
            },
        },
        {"type": "error", "code": "server_error", "message": "boom", "param": None},
    )
    with SferenceClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=body)),
        api_key="tok",
    ) as client:
        events = client.stream_response(model="m", input="hi")
        assert next(events).type == "response.created"
        with pytest.raises(ApiError) as excinfo:
            next(events)

    assert excinfo.value.code == "server_error"
    assert excinfo.value.message == "boom"


def test_stream_response_raises_refusal_before_stream() -> None:
    refusal = httpx.Response(
        402,
        json={
            "error": {
                "type": "billing",
                "code": "insufficient_balance",
                "message": "top up",
            }
        },
    )
    with (
        SferenceClient(
            transport=httpx.MockTransport(lambda _r: refusal), api_key="tok"
        ) as client,
        pytest.raises(ApiError) as excinfo,
    ):
        list(client.stream_response(model="m", input="hi"))

    assert excinfo.value.status_code == 402
    assert excinfo.value.code == "insufficient_balance"


async def test_stream_response_async_yields_events_until_completed() -> None:
    from sference_sdk import AsyncSferenceClient

    transport = httpx.MockTransport(
        lambda _r: httpx.Response(200, content=_frame(*_STREAM_EVENTS))
    )
    async with AsyncSferenceClient(transport=transport, api_key="tok") as client:
        events = [e async for e in client.stream_response(model="m", input="hi")]

    assert "".join(e.delta or "" for e in events) == "Hello"
    assert events[-1].type == "response.completed"


_RESPONSE_JSON = {"id": "r", "created_at": 1, "model": "m", "status": "completed"}


@pytest.mark.parametrize(
    ("client_timeout", "call", "read_timeout"),
    [
        # Blocking creates wait silently for the whole generation; the server allows up to 1800 s.
        (None, lambda c: c.create_response(model="m", input="hi"), 1860.0),
        (
            None,
            lambda c: c.create_response(model="m", input="hi", background=True),
            30.0,
        ),
        (None, lambda c: list(c.stream_response(model="m", input="hi")), 360.0),
        (
            None,
            lambda c: c.create_decision(
                model="m", state="s", questions={"q": {"type": "noul"}}
            ),
            45.0,
        ),
        (None, lambda c: c.get_response("r"), 30.0),
        # An explicit client timeout applies to every call, long ones included.
        (5.0, lambda c: c.create_response(model="m", input="hi"), 5.0),
        (5.0, lambda c: list(c.stream_response(model="m", input="hi")), 5.0),
    ],
)
def test_read_timeout_covers_the_servers_silent_wait(
    client_timeout: float | None,
    call: Callable[[SferenceClient], object],
    read_timeout: float,
) -> None:
    seen: dict[str, float] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.extensions["timeout"])
        if request.url.path == "/v1/decisions":
            return httpx.Response(
                200, json={"model": "m", "answers": {}, "usage": {"input_tokens": 1}}
            )
        if json.loads(request.content or b"{}").get("stream"):
            return httpx.Response(
                200,
                content=_frame(
                    {"type": "response.completed", "response": _RESPONSE_JSON}
                ),
            )
        return httpx.Response(200, json=_RESPONSE_JSON)

    with SferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok", timeout=client_timeout
    ) as client:
        call(client)

    assert seen["read"] == read_timeout
    assert seen["connect"] == (client_timeout or 30.0)
