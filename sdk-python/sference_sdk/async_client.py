from __future__ import annotations

import asyncio
import json
import os
import time
import warnings
from pathlib import Path
from collections.abc import Mapping
from typing import Any, AsyncIterator, BinaryIO

import httpx

from .client import ApiError, SferenceClient
from .checkpoint import clear_checkpoint, load_checkpoint, save_checkpoint
from .models import (
    COMPLETION_WINDOWS,
    Batch,
    BatchCreatePayload,
    BatchList,
    BatchResultRow,
    BatchResults,
    CreateEmbeddingPayload,
    EmbeddingInput,
    EmbeddingEncodingFormat,
    EmbeddingResponse,
    InferenceRequest,
    Response,
    ResponseList,
    Stream,
    StreamInferenceCompletionEvent,
    StreamEventList,
    StreamList,
    StreamWindow,
)


class AsyncSferenceClient:
    """Async HTTP client for the sference batch API (``httpx.AsyncClient``)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float | None = None,
    ) -> None:
        self.base_url = base_url or os.getenv("SFERENCE_BASE_URL", "https://api.sference.com")
        self._token = api_key or os.getenv("SFERENCE_API_KEY")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=30.0 if timeout is None else timeout,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncSferenceClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, Any] | None = None,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        response = await self._client.request(
            method, path, headers=self._headers(), json=json_body, params=params
        )
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = {"detail": response.text}
            raise ApiError(f"{response.status_code}: {payload}")
        return response.json()

    async def _request_response(
        self, method: str, path: str, json_body: dict[str, Any] | None = None
    ) -> httpx.Response:
        response = await self._client.request(method, path, headers=self._headers(), json=json_body)
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = {"detail": response.text}
            raise ApiError(f"{response.status_code}: {payload}")
        return response

    async def get_me(self) -> dict[str, Any]:
        return await self._request("GET", "/v1/auth/me")

    async def list_models(self) -> dict[str, Any]:
        """OpenAI-compatible model list (GET /v1/models)."""
        response = await self._request("GET", "/v1/models")
        if not isinstance(response, dict):
            raise ApiError("GET /v1/models returned unexpected payload")
        return response

    async def create_embeddings(
        self,
        *,
        model: str,
        input: EmbeddingInput,
        encoding_format: EmbeddingEncodingFormat = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> EmbeddingResponse:
        """Create embeddings (POST /v1/embeddings)."""
        payload = CreateEmbeddingPayload(
            model=model,
            input=input,
            encoding_format=encoding_format,
            dimensions=dimensions,
            user=user,
        )
        response = await self._request("POST", "/v1/embeddings", payload.model_dump(exclude_none=True))
        return EmbeddingResponse.model_validate(response)

    async def submit_batch(
        self,
        *,
        input_file: str | None = None,
        requests: list[dict[str, Any]] | None = None,
        model: str | None = None,
        window: str = "24h",
    ) -> Batch:
        if window not in COMPLETION_WINDOWS:
            allowed = ", ".join(f'"{w}"' for w in COMPLETION_WINDOWS)
            raise ValueError(f"window must be one of {allowed}.")
        resolved = requests or []
        if input_file:
            resolved = SferenceClient.parse_inference_requests_jsonl(Path(input_file), model=model)
        if not resolved:
            raise ValueError("At least one request is required (use input_file or non-empty requests)")
        if isinstance(resolved, list) and resolved and isinstance(resolved[0], dict):
            reqs: list[InferenceRequest] = []
            for r in resolved:
                if not isinstance(r, dict):
                    raise TypeError("requests must be a list of dicts or InferenceRequest objects")
                body = r.get("body")
                if isinstance(body, dict) and {"method", "url", "body"}.issubset(r.keys()):
                    reqs.append(InferenceRequest(custom_id=r.get("custom_id"), body=body))
                else:
                    reqs.append(InferenceRequest.model_validate(r))
            resolved = reqs
        payload = BatchCreatePayload(window=window, requests=resolved)  # type: ignore[arg-type]
        response = await self._request("POST", "/v1/batches", payload.model_dump())
        return Batch.model_validate(response)

    async def get_batch(self, batch_id: str) -> Batch:
        response = await self._request("GET", f"/v1/batches/{batch_id}")
        return Batch.model_validate(response)

    async def list_batches(self) -> BatchList:
        response = await self._request("GET", "/v1/batches")
        return BatchList.model_validate(response)

    async def get_results(self, batch_id: str) -> BatchResults:
        response = await self._request("GET", f"/v1/batches/{batch_id}/results")
        return BatchResults.model_validate(response)

    async def get_results_indexed(self, batch_id: str) -> dict[str, BatchResultRow]:
        """Fetch batch results keyed by ``custom_id``."""
        return (await self.get_results(batch_id)).index_by_custom_id()

    async def cancel_batch(self, batch_id: str) -> Batch:
        response = await self._request("POST", f"/v1/batches/{batch_id}/cancel")
        return Batch.model_validate(response)

    async def download_results_jsonl(self, batch_id: str, out: str | Path | BinaryIO) -> None:
        resp = await self._request_response("GET", f"/v1/batches/{batch_id}/results.jsonl")
        if isinstance(out, (str, Path)):
            Path(out).write_bytes(resp.content)
            return
        out.write(resp.content)

    async def wait_for_completion(self, batch_id: str, poll_interval: float = 1.0, timeout: float = 30.0) -> Batch:
        deadline = time.time() + timeout
        while True:
            batch = await self.get_batch(batch_id)
            if batch.status in ("completed", "failed", "cancelled"):
                return batch
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for batch {batch_id}")
            await asyncio.sleep(poll_interval)

    async def create_stream(self, name: str, window: StreamWindow = "24h") -> Stream:
        response = await self._request("POST", "/v1/streams", {"name": name, "window": window})
        return Stream.model_validate(response)

    async def list_streams(self) -> StreamList:
        response = await self._request("GET", "/v1/streams")
        return StreamList.model_validate(response)

    async def get_stream(self, stream_id: str) -> Stream:
        response = await self._request("GET", f"/v1/streams/{stream_id}")
        return Stream.model_validate(response)

    # Note: POST /v1/streams/{id}/items has been removed. Use create_response() with metadata.stream_id instead.

    async def create_response(
        self,
        *,
        model: str,
        input: str | list[dict[str, str]],  # bare string, or [{"role": "user", "content": "..."}]
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        include_reasoning: bool | None = None,
        enable_thinking: bool | None = None,
        background: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        """Create a standalone response or stream-associated response.

        For stream-associated, include stream_id in metadata:
            metadata={"stream_id": "uuid", "completion_window": "24h"}
        """
        payload: dict[str, Any] = {
            "model": model,
            "input": input,
        }
        if instructions is not None:
            payload["instructions"] = instructions
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if include_reasoning is not None:
            payload["include_reasoning"] = include_reasoning
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if background:
            payload["background"] = True
        if metadata is not None:
            payload["metadata"] = metadata

        response = await self._request("POST", "/v1/responses", payload)
        return Response.model_validate(response)

    async def wait_for_response(
        self,
        response_id: str,
        poll_interval: float = 2.0,
        timeout: float = 3600.0,
    ) -> Response:
        """Poll GET /v1/responses/{id} until status is terminal (async create + wait)."""
        deadline = time.time() + timeout
        while True:
            item = await self.get_response(response_id)
            if item.status in ("completed", "failed", "cancelled"):
                return item
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for response {response_id}")
            await asyncio.sleep(poll_interval)

    async def get_response(self, response_id: str) -> Response:
        """Get a response by ID."""
        response = await self._request("GET", f"/v1/responses/{response_id}")
        return Response.model_validate(response)

    async def cancel_response(self, response_id: str) -> Response:
        """Cancel a pending or running response."""
        response = await self._request("DELETE", f"/v1/responses/{response_id}")
        return Response.model_validate(response)

    async def list_responses(
        self,
        *,
        limit: int = 100,
        stream_id: str | None = None,
    ) -> ResponseList:
        """List responses. Optionally filter by stream_id."""
        params: dict[str, Any] = {"limit": limit}
        if stream_id is not None:
            params["stream_id"] = stream_id

        response = await self._request("GET", "/v1/responses", params=params)
        return ResponseList.model_validate(response)

    async def archive_stream(self, stream_id: str) -> Stream:
        response = await self._request("PATCH", f"/v1/streams/{stream_id}", {"status": "archived"})
        return Stream.model_validate(response)

    async def cancel_stream(self, stream_id: str) -> Stream:
        response = await self._request("PATCH", f"/v1/streams/{stream_id}", {"status": "cancelled"})
        return Stream.model_validate(response)

    async def list_responses_events(
        self,
        *,
        stream_id: str | None = None,
        limit: int = 20,
        starting_after: str | None = None,
        ending_before: str | None = None,
        wait_ms: int = 0,
    ) -> StreamEventList:
        """GET /v1/responses/events — completion events (omit stream_id for non-stream feed)."""
        params: dict[str, Any] = {"limit": limit, "wait_ms": wait_ms}
        if stream_id is not None:
            params["stream_id"] = stream_id
        if starting_after is not None:
            params["starting_after"] = starting_after
        if ending_before is not None:
            params["ending_before"] = ending_before
        response = await self._client.request(
            "GET",
            "/v1/responses/events",
            headers=self._headers(),
            params=params,
        )
        if response.status_code >= 400:
            try:
                payload = response.json()
            except Exception:
                payload = {"detail": response.text}
            raise ApiError(f"{response.status_code}: {payload}")
        return StreamEventList.model_validate(response.json())

    async def iter_responses_events(
        self,
        *,
        stream_id: str | None = None,
        consumer_name: str = "default",
        starting_after: str | None = None,
        checkpoint: bool = True,
        from_latest: bool = False,
    ) -> AsyncIterator[StreamInferenceCompletionEvent]:
        ck_stream = stream_id if stream_id is not None else "__responses_events__"
        if from_latest:
            clear_checkpoint(self.base_url, ck_stream, consumer_name)
        cur: str | None = starting_after
        if checkpoint and cur is None:
            cur = load_checkpoint(self.base_url, ck_stream, consumer_name)

        if cur is None:
            page = await self.list_responses_events(stream_id=stream_id, limit=100, wait_ms=0)
            for ev in reversed(page.data):
                yield ev
                if checkpoint:
                    save_checkpoint(self.base_url, ck_stream, consumer_name, ev.completion_id)
            return

        while True:
            page = await self.list_responses_events(stream_id=stream_id, limit=100, starting_after=cur, wait_ms=0)
            if not page.data:
                return
            for ev in page.data:
                yield ev
                if checkpoint:
                    save_checkpoint(self.base_url, ck_stream, consumer_name, ev.completion_id)
                cur = ev.completion_id
            if not page.has_more:
                return
