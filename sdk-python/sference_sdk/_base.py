"""Transport-independent pieces shared by the sync and async clients.

Both clients only do I/O; request building, header handling and error parsing
live here so the two cannot drift apart.
"""

from __future__ import annotations

import contextvars
import json
import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from .models import (
    COMPLETION_WINDOWS,
    BatchCreatePayload,
    CreateDecisionPayload,
    CreateEmbeddingPayload,
    DecisionImage,
    DecisionQuestion,
    EmbeddingEncodingFormat,
    EmbeddingInput,
    InferenceRequest,
    ServiceTier,
)

DEFAULT_BASE_URL = "https://api.sference.com"
DEFAULT_TIMEOUT = 30.0

TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "cancelled"})
TERMINAL_RESPONSE_STATUSES = frozenset({"completed", "failed", "cancelled", "incomplete"})

# Checkpoint key for the event feed when no stream_id filter is given.
EVENTS_CHECKPOINT_KEY = "__responses_events__"


class ApiError(Exception):
    """A non-2xx response from the sference API.

    ``str(err)`` is ``"<status>: <body>"``. The parsed fields come from the
    OpenAI/Anthropic-style ``{"error": {...}}`` envelope when the API sends one.
    """

    def __init__(self, message: str, *, status_code: int | None = None, body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        detail = body.get("error") if isinstance(body, dict) else None
        if not isinstance(detail, dict):
            detail = {}
        self.type: str | None = detail.get("type")
        self.code: str | None = detail.get("code")
        self.param: str | None = detail.get("param")
        fallback = body.get("detail") if isinstance(body, dict) else None
        self.message: str = str(detail.get("message") or fallback or message)

    @classmethod
    def from_response(cls, response: httpx.Response) -> ApiError:
        try:
            body: Any = response.json()
        except ValueError:
            body = {"detail": response.text}
        return cls(f"{response.status_code}: {body}", status_code=response.status_code, body=body)


def raise_for_status(response: httpx.Response) -> None:
    if response.status_code >= 400:
        raise ApiError.from_response(response)


_request_extra_headers: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "sference_request_extra_headers",
    default=None,
)


class BaseClient:
    """Credential, header and request-building logic common to both clients."""

    def __init__(self, base_url: str | None, api_key: str | None) -> None:
        self.base_url = base_url or os.getenv("SFERENCE_BASE_URL", DEFAULT_BASE_URL)
        self._token = api_key or os.getenv("SFERENCE_API_KEY")

    def set_api_key(self, api_key: str | None) -> None:
        """Swap the credential used for subsequent requests.

        For callers whose key changes while a client is alive — a rotated key in
        a long-running integration, for instance — so they need not rebuild the
        client and its connection pool.
        """
        self._token = api_key

    @contextmanager
    def extra_headers(self, headers: Mapping[str, str]) -> Iterator[None]:
        """Attach headers to every HTTP call in this block (e.g. one batch workflow).

        Prefer OpenTelemetry: if the host app has an active span, ``inject()`` adds
        W3C trace context automatically per request.
        """
        merged = {**(_request_extra_headers.get() or {}), **dict(headers)}
        token = _request_extra_headers.set(merged)
        try:
            yield
        finally:
            _request_extra_headers.reset(token)

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        try:
            from opentelemetry.propagate import inject

            carrier: dict[str, str] = {}
            inject(carrier)
            headers.update(carrier)
        except Exception:
            # OpenTelemetry is optional; tracing must never break a request.
            pass
        scoped = _request_extra_headers.get()
        if scoped:
            headers.update(scoped)
        if extra:
            headers.update(extra)
        return headers

    # Request builders -----------------------------------------------------

    @staticmethod
    def _embedding_payload(
        *,
        model: str,
        input: EmbeddingInput,
        encoding_format: EmbeddingEncodingFormat,
        dimensions: int | None,
        user: str | None,
    ) -> dict[str, Any]:
        payload = CreateEmbeddingPayload(
            model=model, input=input, encoding_format=encoding_format, dimensions=dimensions, user=user
        )
        return payload.model_dump(exclude_none=True)

    @staticmethod
    def _decision_payload(
        *,
        model: str,
        state: Any,
        questions: Mapping[str, DecisionQuestion | Mapping[str, Any]],
        images: Sequence[str | DecisionImage | Mapping[str, Any]] | None,
    ) -> dict[str, Any]:
        payload = CreateDecisionPayload.model_validate(
            {"model": model, "state": state, "questions": dict(questions), "images": list(images or [])}
        )
        return payload.model_dump(mode="json", exclude={"images"} if images is None else set())

    @classmethod
    def _batch_payload(
        cls,
        *,
        input_file: str | None,
        requests: Sequence[InferenceRequest | Mapping[str, Any]] | None,
        model: str | None,
        window: str,
    ) -> dict[str, Any]:
        if window not in COMPLETION_WINDOWS:
            allowed = ", ".join(f'"{w}"' for w in COMPLETION_WINDOWS)
            raise ValueError(f"window must be one of {allowed}.")
        if input_file:
            resolved = cls.parse_inference_requests_jsonl(Path(input_file), model=model)
        else:
            resolved = [_coerce_request(r) for r in requests or []]
        if not resolved:
            raise ValueError("At least one request is required (use input_file or non-empty requests)")
        return BatchCreatePayload(window=window, requests=resolved).model_dump()  # type: ignore[arg-type]

    @staticmethod
    def _response_request(
        *,
        model: str,
        input: str | Sequence[Mapping[str, Any]],
        instructions: str | None,
        max_output_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        include_reasoning: bool | None,
        enable_thinking: bool | None,
        reasoning: Mapping[str, Any] | None,
        tools: Sequence[Mapping[str, Any]] | None,
        tool_choice: str | Mapping[str, Any] | None,
        include: Sequence[str] | None,
        text: Mapping[str, Any] | None,
        service_tier: ServiceTier | None,
        prompt_cache_key: str | None,
        max_queue_wait_s: int | None,
        background: bool,
        metadata: Mapping[str, Any] | None,
        idempotency_key: str | None,
    ) -> tuple[dict[str, Any], dict[str, str] | None]:
        """Body and extra headers for POST /v1/responses; unset options are omitted."""
        if isinstance(input, Sequence) and not isinstance(input, str):
            if not input:
                raise ValueError("input must not be empty")
            input = [dict(item) for item in input]
        optional: dict[str, Any] = {
            "instructions": instructions,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "include_reasoning": include_reasoning,
            "enable_thinking": enable_thinking,
            "reasoning": dict(reasoning) if reasoning is not None else None,
            "tools": [dict(t) for t in tools] if tools is not None else None,
            "tool_choice": tool_choice if isinstance(tool_choice, (str, type(None))) else dict(tool_choice),
            "include": list(include) if include is not None else None,
            "text": dict(text) if text is not None else None,
            "service_tier": service_tier,
            "prompt_cache_key": prompt_cache_key,
            "max_queue_wait_s": max_queue_wait_s,
            "metadata": dict(metadata) if metadata is not None else None,
        }
        payload: dict[str, Any] = {"model": model, "input": input}
        payload.update({k: v for k, v in optional.items() if v is not None})
        if background:
            payload["background"] = True
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else None
        return payload, headers

    @staticmethod
    def _params(**params: Any) -> dict[str, Any]:
        """Query params without the unset (None) ones, so server defaults apply."""
        return {k: v for k, v in params.items() if v is not None}

    @staticmethod
    def parse_inference_requests_jsonl(path: Path, model: str | None = None) -> list[InferenceRequest]:
        """Parse a batch JSONL file.

        Each line is either an OpenAI batch envelope (``{"custom_id", "method",
        "url", "body"}``) or content-only (``{"content": ...}``, needs ``model``).
        """
        parsed: list[InferenceRequest] = []
        warn_model_ignored = False
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and {"method", "url", "body"}.issubset(obj.keys()):
                    if model:
                        warn_model_ignored = True
                    body = obj.get("body")
                    if not isinstance(body, dict) or not body.get("model"):
                        raise ValueError(f"Invalid JSONL line {idx}: body.model is required")
                    parsed.append(InferenceRequest(custom_id=obj.get("custom_id"), body=body))
                    continue
                if isinstance(obj, dict) and "content" in obj:
                    if not model:
                        raise ValueError("model is required for content-only JSONL format")
                    parsed.append(
                        InferenceRequest(
                            custom_id=obj.get("custom_id") or f"request-{idx}",
                            body={"model": model, "messages": [{"role": "user", "content": obj["content"]}]},
                        )
                    )
                    continue
                raise ValueError(f"Unsupported JSONL line format at line {idx}")
        if warn_model_ignored:
            warnings.warn("model argument is ignored for OpenAI-compatible JSONL lines", stacklevel=2)
        return parsed


def _coerce_request(r: InferenceRequest | Mapping[str, Any]) -> InferenceRequest:
    if isinstance(r, InferenceRequest):
        return r
    if not isinstance(r, Mapping):
        raise TypeError("requests must be a list of dicts or InferenceRequest objects")
    # OpenAI batch envelopes carry method/url; only custom_id and body are sent.
    if isinstance(r.get("body"), dict) and {"method", "url", "body"}.issubset(r.keys()):
        return InferenceRequest(custom_id=r.get("custom_id"), body=r["body"])
    return InferenceRequest.model_validate(dict(r))
