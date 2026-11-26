from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from sference_sdk.async_client import AsyncSferenceClient
from sference_sdk.client import ApiError

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.mark.asyncio
async def test_async_submit_batch_raises_api_error_on_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/batches":
            return httpx.Response(status_code=401, json={"detail": "Unauthorized"})
        return httpx.Response(status_code=404, json={"detail": "not found"})

    async with AsyncSferenceClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ApiError, match="401"):
            await client.submit_batch(
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


@pytest.mark.asyncio
async def test_async_download_results_jsonl_via_mock_transport(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == "/v1/batches/batch_abc/results.jsonl"
        ):
            assert request.headers.get("authorization") == "Bearer tok"
            return httpx.Response(status_code=200, content=b'{"x": 1}\n{"x": 2}\n')
        return httpx.Response(status_code=404, json={"detail": "not found"})

    out_path = tmp_path / "out.jsonl"
    async with AsyncSferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        await client.download_results_jsonl("batch_abc", out_path)
    assert out_path.read_text(encoding="utf-8").splitlines() == ['{"x": 1}', '{"x": 2}']


def _minimal_batch_payload(*, batch_id: str, status: str) -> dict:
    return {
        "id": batch_id,
        "status": status,
        "window": "24h",
        "request_count": 1,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
        "completed_at": None,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
    }


@pytest.mark.asyncio
async def test_async_wait_for_completion_via_mock_transport() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/batches/batch_wait":
            calls.append("get")
            if len(calls) < 3:
                return httpx.Response(
                    status_code=200,
                    json=_minimal_batch_payload(
                        batch_id="batch_wait", status="pending"
                    ),
                )
            return httpx.Response(
                status_code=200,
                json=_minimal_batch_payload(batch_id="batch_wait", status="completed"),
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    async with AsyncSferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        batch = await client.wait_for_completion(
            "batch_wait", poll_interval=0.01, timeout=5.0
        )
    assert batch.status == "completed"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_async_wait_for_response_treats_incomplete_as_terminal() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/responses/resp_trunc":
            calls.append("get")
            return httpx.Response(
                status_code=200,
                json={
                    "id": "resp_trunc",
                    "object": "response",
                    "created_at": 1712345678,
                    "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            )
        return httpx.Response(status_code=404, json={"detail": "not found"})

    async with AsyncSferenceClient(
        transport=httpx.MockTransport(handler), api_key="tok"
    ) as client:
        resp = await client.wait_for_response(
            "resp_trunc", poll_interval=0.01, timeout=1.0
        )
    assert len(calls) == 1
    assert resp.status == "incomplete"
    assert resp.incomplete_details is not None
    assert resp.incomplete_details.reason == "max_output_tokens"
