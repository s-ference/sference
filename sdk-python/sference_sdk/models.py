from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


BatchStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

# Completion window accepted by batches, streams, and background responses.
CompletionWindow = Literal["24h"]
COMPLETION_WINDOWS: tuple[str, ...] = ("24h",)


class InferenceRequest(BaseModel):
    """Unified inference request schema for batches and streams.

    custom_id is user-provided and can be used for correlation.
    """

    model_config = ConfigDict(extra="forbid")

    custom_id: str | None = Field(default=None, description="User-provided identifier for correlation.")
    body: dict[str, Any] = Field(description="OpenAI-style request body. Must include `model`.")

    @classmethod
    def chat(
        cls,
        *,
        custom_id: str,
        user_content: str,
        model: str,
        system_content: str | None = None,
        temperature: float | None = None,
    ) -> InferenceRequest:
        """Build a chat-completions batch row (``body.messages``)."""
        messages: list[dict[str, str]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
        body: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        return cls(custom_id=custom_id, body=body)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict[str, str]


class Batch(BaseModel):
    id: str
    status: BatchStatus
    window: CompletionWindow
    request_count: int
    created_at: str
    updated_at: str
    completed_at: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0


class BatchList(BaseModel):
    items: list[Batch]


class BatchResultRow(BaseModel):
    """One row from ``GET /v1/batches/{id}/results`` (sference ``result_json`` / ``error_json`` shape)."""

    model_config = ConfigDict(extra="allow")

    custom_id: str | None = None
    status: str | None = None
    result_json: dict[str, Any] | None = None
    error_json: dict[str, Any] | None = None

    @property
    def completion_text(self) -> str:
        """Assistant text for a completed row, or a bracketed status/error summary."""
        if self.status != "completed":
            if self.error_json:
                return f"[{self.status}] {self.error_json}"
            return f"[{self.status}]"
        result = self.result_json or {}
        choices = result.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else str(content or "")


class BatchResults(BaseModel):
    batch_id: str
    status: BatchStatus
    output_url: str | None = None
    completed_at: str | None = None
    results: list[dict[str, Any]] | None = None

    def rows(self) -> list[BatchResultRow]:
        return [BatchResultRow.model_validate(row) for row in self.results or []]

    def index_by_custom_id(self) -> dict[str, BatchResultRow]:
        indexed: dict[str, BatchResultRow] = {}
        for row in self.rows():
            if row.custom_id is not None:
                indexed[str(row.custom_id)] = row
        return indexed


class BatchCreatePayload(BaseModel):
    window: CompletionWindow = "24h"
    requests: list[InferenceRequest] = Field(min_length=1)


StreamStatus = Literal["open", "cancelled", "archived"]
StreamWindow = CompletionWindow


class Stream(BaseModel):
    """Stream detail (GET /v1/streams/{id}); list responses omit counter fields (defaults apply)."""

    id: str
    name: str
    window: StreamWindow
    status: StreamStatus
    created_at: str
    updated_at: str
    cancelled_at: str | None = None
    archived_at: str | None = None
    total_items: int = 0
    pending_items: int = 0
    running_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    cancelled_items: int = 0
    completion_ratio: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    last_completion_at: str | None = None
    latest_completion_id: str | None = None


class StreamInferenceCompletionEvent(BaseModel):
    completion_id: str
    response_id: str | None = None
    custom_id: str | None = None
    status: str
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    completed_at: str | None = None


class StreamEventList(BaseModel):
    object: str = "list"
    data: list[StreamInferenceCompletionEvent]
    has_more: bool


class StreamList(BaseModel):
    items: list[Stream]


# Response (OpenAI-compatible) models
ResponseStatus = Literal["in_progress", "completed", "failed", "cancelled"]


class ResponseInputMessage(BaseModel):
    """Message in the input array for a response."""

    role: Literal["user", "assistant", "system", "developer"]
    content: str


class ResponseCreatePayload(BaseModel):
    """Request body for POST /v1/responses."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(description='Model identifier, e.g. "zai-org/GLM-5"')
    input: str | list[ResponseInputMessage]
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    include_reasoning: bool = True
    enable_thinking: bool | None = None
    background: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input")
    @classmethod
    def _reject_empty_input(cls, v: str | list[ResponseInputMessage]) -> str | list[ResponseInputMessage]:
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("input must not be empty")
        return v


class ResponseReasoningSummaryPart(BaseModel):
    type: Literal["summary_text"] = "summary_text"
    text: str


class ResponseOutputText(BaseModel):
    """An ``output_text`` content part nested inside a message item."""

    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    """Assistant message output item wrapping ``output_text`` content parts.

    Mirrors the OpenAI Responses API, where final text lives in a ``message`` item
    with a nested ``content[]`` of ``output_text`` parts rather than a top-level
    ``output_text`` item.
    """

    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    status: Literal["completed", "in_progress", "incomplete"] = "completed"
    content: list[ResponseOutputText]


class ResponseOutputReasoning(BaseModel):
    """OpenAI Responses API reasoning output item (``summary``, not top-level ``text``)."""

    type: Literal["reasoning"] = "reasoning"
    id: str
    summary: list[ResponseReasoningSummaryPart] = Field(default_factory=list)


def reasoning_summary_plaintext(item: ResponseOutputReasoning) -> str:
    return "".join(part.text for part in item.summary)


class ResponseFunctionToolCall(BaseModel):
    type: Literal["function_call"] = "function_call"
    id: str
    call_id: str
    name: str
    arguments: str = "{}"
    status: Literal["completed", "in_progress", "incomplete"] = "completed"


ResponseOutputContent = Annotated[
    ResponseOutputMessage | ResponseOutputReasoning | ResponseFunctionToolCall,
    Field(discriminator="type"),
]


class ResponseInputTokensDetails(BaseModel):
    cached_tokens: int = 0


class ResponseOutputTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class ResponseUsage(BaseModel):
    """Token usage for a response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: ResponseInputTokensDetails = Field(default_factory=ResponseInputTokensDetails)
    output_tokens_details: ResponseOutputTokensDetails = Field(default_factory=ResponseOutputTokensDetails)


class ResponseError(BaseModel):
    """Error information for a failed response."""

    code: str
    message: str


class Response(BaseModel):
    """OpenAI-compatible response object."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: ResponseStatus
    output: list[ResponseOutputContent] | None = None
    error: ResponseError | None = None
    usage: ResponseUsage | None = None


class ResponseListItem(BaseModel):
    """Item in a response list."""

    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: ResponseStatus


class ResponseList(BaseModel):
    """Paginated list of responses."""

    object: Literal["list"] = "list"
    data: list[ResponseListItem]
    has_more: bool = False


# Embeddings (OpenAI-compatible)
EmbeddingInput = str | list[str] | list[int] | list[list[int]]
EmbeddingEncodingFormat = Literal["float", "base64"]


class EmbeddingData(BaseModel):
    index: int
    object: Literal["embedding"] = "embedding"
    embedding: list[float] | str


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage | None = None


class CreateEmbeddingPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: EmbeddingInput
    encoding_format: EmbeddingEncodingFormat = "float"
    dimensions: int | None = None
    user: str | None = None
