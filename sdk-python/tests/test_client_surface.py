"""Every public client method, on both clients, against the API contract.

One case per method: the request it sends must be an operation in
``contract/openapi.json`` with a body the contract accepts, and the sync and
async clients must send the same request. A new public method fails
``test_every_public_method_has_a_case`` until it gets a case here.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest

from sference_sdk import AsyncSferenceClient, ChoiceQuestion, NoulQuestion, ScoreQuestion, SferenceClient

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONTRACT = json.loads((Path(__file__).resolve().parents[2] / "contract" / "openapi.json").read_text(encoding="utf-8"))

# Methods that never reach the API on their own.
LOCAL_METHODS = {"close", "aclose", "extra_headers", "set_api_key", "parse_inference_requests_jsonl"}


def _fixture(operation: str, status: int = 200) -> Any:
    return json.loads((FIXTURES / operation / f"{status}.json").read_text(encoding="utf-8"))


_BATCH = _fixture("V1BatchesCreateBatch", 201)
_STREAM = _fixture("V1StreamsCreateStream", 201)
_RESPONSE = {"id": "r", "object": "response", "created_at": 1, "model": "m", "status": "completed"}
_RESULTS = {"batch_id": "b", "status": "completed", "results": [{"custom_id": "r1", "status": "completed"}]}
_SSE = (
    'event: response.completed\ndata: {"type": "response.completed", "response": %s}\n\n' % json.dumps(_RESPONSE)
).encode()


@dataclass
class Case:
    call: Callable[[Any, Path], Any]
    method: str
    path: str
    reply: Any = None
    params: dict[str, str] = field(default_factory=dict)
    status: int = 200


CASES: dict[str, Case] = {
    "health": Case(lambda c, _: c.health(), "GET", "/v1/health", {"status": "ok"}),
    "get_me": Case(lambda c, _: c.get_me(), "GET", "/v1/auth/me", _fixture("V1AuthMeMe")),
    "list_models": Case(lambda c, _: c.list_models(), "GET", "/v1/models", {"object": "list", "data": []}),
    "get_billing_balance": Case(
        lambda c, _: c.get_billing_balance(), "GET", "/v1/billing/balance", {"credit_balance_micros": 1}
    ),
    "get_billing_summary": Case(
        lambda c, _: c.get_billing_summary(days=7),
        "GET",
        "/v1/billing/summary",
        {
            "summary": {
                "days": 7, "total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0,
                "total_requests": 0, "estimated_cost_micros": 0, "credit_balance_micros": 0,
            },
            "by_model": [],
        },
        {"days": "7"},
    ),
    "list_billing_transactions": Case(
        lambda c, _: c.list_billing_transactions(limit=10, starting_after="t"),
        "GET",
        "/v1/billing/transactions",
        {"items": []},
        {"limit": "10", "starting_after": "t"},
    ),
    "get_usage_summary": Case(
        lambda c, _: c.get_usage_summary(days=7),
        "GET",
        "/v1/usage/summary",
        {
            "summary": {"total_prompt_tokens": 0, "total_completion_tokens": 0, "total_tokens": 0, "total_requests": 0},
            "by_model": [],
        },
        {"days": "7"},
    ),
    # Unset filters are omitted so the server's defaults apply.
    "get_usage_daily": Case(lambda c, _: c.get_usage_daily(), "GET", "/v1/usage/daily", {"items": []}),
    "get_usage_by_key": Case(
        lambda c, _: c.get_usage_by_key(month="2026-11", tag="prod"),
        "GET",
        "/v1/usage/by-key",
        {"period_start": "2026-11-01", "period_end": "2026-11-30", "items": []},
        {"month": "2026-11", "tag": "prod"},
    ),
    "create_embeddings": Case(
        lambda c, _: c.create_embeddings(model="m", input=["a", "b"], dimensions=8),
        "POST",
        "/v1/embeddings",
        {"object": "list", "data": [{"index": 0, "embedding": [0.1]}], "model": "m"},
    ),
    "create_decision": Case(
        lambda c, _: c.create_decision(
            model="Cloudflare/clef",
            state={"ticket": "I was charged twice"},
            images=["data:image/png;base64,eA==", {"content_type": "image/jpeg", "base64": "eA=="}],
            questions={
                "route": ChoiceQuestion(criteria={"billing": "Payments", "support": "Everything else"}),
                "urgency": ScoreQuestion(criteria=["Low", "High"], instructions="How urgent?"),
                "refund": NoulQuestion(),
                "fraud": {"type": "noul", "criteria": {"true": "Unauthorised", "false": "Customer error"}},
            },
        ),
        "POST",
        "/v1/decisions",
        _fixture("createDecision"),
    ),
    "submit_batch": Case(
        lambda c, _: c.submit_batch(
            requests=[{"custom_id": "r1", "body": {"model": "m", "messages": [{"role": "user", "content": "hi"}]}}]
        ),
        "POST",
        "/v1/batches",
        _BATCH,
        status=201,
    ),
    "list_batches": Case(lambda c, _: c.list_batches(), "GET", "/v1/batches", _fixture("V1BatchesListBatches")),
    "get_batch": Case(lambda c, _: c.get_batch("b"), "GET", "/v1/batches/b", _BATCH),
    "wait_for_completion": Case(
        lambda c, _: c.wait_for_completion("b", poll_interval=0), "GET", "/v1/batches/b", {**_BATCH, "status": "completed"}
    ),
    "cancel_batch": Case(
        lambda c, _: c.cancel_batch("b"), "POST", "/v1/batches/b/cancel", _fixture("V1BatchesBatchIdCancelCancelBatch")
    ),
    "get_results": Case(lambda c, _: c.get_results("b"), "GET", "/v1/batches/b/results", _RESULTS),
    "get_results_indexed": Case(lambda c, _: c.get_results_indexed("b"), "GET", "/v1/batches/b/results", _RESULTS),
    "download_results_jsonl": Case(
        lambda c, tmp: c.download_results_jsonl("b", tmp / "out.jsonl"), "GET", "/v1/batches/b/results.jsonl", b"{}\n"
    ),
    "create_stream": Case(
        lambda c, _: c.create_stream("daily", window="24h"), "POST", "/v1/streams", _STREAM, status=201
    ),
    "list_streams": Case(lambda c, _: c.list_streams(), "GET", "/v1/streams", _fixture("V1StreamsListStreams")),
    "get_stream": Case(lambda c, _: c.get_stream("s"), "GET", "/v1/streams/s", _STREAM),
    "archive_stream": Case(lambda c, _: c.archive_stream("s"), "PATCH", "/v1/streams/s", _STREAM),
    "cancel_stream": Case(lambda c, _: c.cancel_stream("s"), "PATCH", "/v1/streams/s", _STREAM),
    "create_response": Case(
        lambda c, _: c.create_response(
            model="m",
            input=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "name": "lookup", "parameters": {"type": "object"}}],
            service_tier="flex",
            background=True,
            metadata={"completion_window": "24h"},
        ),
        "POST",
        "/v1/responses",
        _RESPONSE,
        status=201,
    ),
    "stream_response": Case(lambda c, _: c.stream_response(model="m", input="hi"), "POST", "/v1/responses", _SSE),
    "get_response": Case(lambda c, _: c.get_response("r"), "GET", "/v1/responses/r", _RESPONSE),
    "wait_for_response": Case(
        lambda c, _: c.wait_for_response("r", poll_interval=0), "GET", "/v1/responses/r", _RESPONSE
    ),
    "cancel_response": Case(
        lambda c, _: c.cancel_response("r"), "DELETE", "/v1/responses/r", {**_RESPONSE, "status": "cancelled"}
    ),
    "list_responses": Case(
        lambda c, _: c.list_responses(limit=5, stream_id="s"),
        "GET",
        "/v1/responses",
        {"object": "list", "data": [], "has_more": False},
        {"limit": "5", "stream_id": "s"},
    ),
    "list_responses_events": Case(
        lambda c, _: c.list_responses_events(stream_id="s", limit=10, starting_after="e", wait_ms=1000),
        "GET",
        "/v1/responses/events",
        _fixture("V1ResponsesEventsListResponseEvents"),
        {"stream_id": "s", "limit": "10", "starting_after": "e", "wait_ms": "1000"},
    ),
    "iter_responses_events": Case(
        lambda c, _: c.iter_responses_events(stream_id="s", checkpoint=False),
        "GET",
        "/v1/responses/events",
        _fixture("V1ResponsesEventsListResponseEvents"),
        {"stream_id": "s", "limit": "100", "wait_ms": "0"},
    ),
}


def _public(cls: type) -> set[str]:
    return {n for n in dir(cls) if not n.startswith("_")}


def test_every_public_method_has_a_case() -> None:
    assert _public(SferenceClient) - LOCAL_METHODS == set(CASES)
    assert _public(AsyncSferenceClient) - LOCAL_METHODS == set(CASES)


def _contract_operation(method: str, path: str) -> dict[str, Any]:
    for template, spec in CONTRACT["paths"].items():
        if re.fullmatch(re.sub(r"\{[^/]+\}", "[^/]+", template), path) and method.lower() in spec:
            return spec[method.lower()]
    raise AssertionError(f"{method} {path} is not an operation in contract/openapi.json")


def _deref(node: Any) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            return _deref(CONTRACT["components"]["schemas"][ref.rsplit("/", 1)[-1]])
        return {k: _deref(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_deref(v) for v in node]
    return node


def _recorder(case: Case, sent: list[httpx.Request]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if isinstance(case.reply, bytes):
            return httpx.Response(case.status, content=case.reply)
        return httpx.Response(case.status, json=case.reply)

    return httpx.MockTransport(handler)


def _wire(request: httpx.Request) -> tuple[str, str, dict[str, str], Any, str | None]:
    body = json.loads(request.content) if request.content else None
    return request.method, request.url.path, dict(request.url.params), body, request.headers.get("authorization")


@pytest.mark.parametrize("name", sorted(CASES))
async def test_method_sends_a_contract_request_from_both_clients(name: str, tmp_path: Path) -> None:
    case = CASES[name]
    sync_sent: list[httpx.Request] = []
    async_sent: list[httpx.Request] = []

    with SferenceClient(transport=_recorder(case, sync_sent), api_key="tok") as client:
        result = case.call(client, tmp_path)
        if inspect.isgenerator(result):
            list(result)

    async with AsyncSferenceClient(transport=_recorder(case, async_sent), api_key="tok") as aclient:
        result = case.call(aclient, tmp_path)
        if inspect.isawaitable(result):
            await result
        elif hasattr(result, "__aiter__"):
            [_ async for _ in result]

    method, path, params, body, auth = _wire(sync_sent[0])
    assert (method, path, params, auth) == (case.method, case.path, case.params, "Bearer tok")
    assert [_wire(r) for r in async_sent] == [_wire(r) for r in sync_sent]

    operation = _contract_operation(method, path)
    if body is not None:
        schema = operation["requestBody"]["content"]["application/json"]["schema"]
        jsonschema.validate(instance=body, schema=_deref(schema))
