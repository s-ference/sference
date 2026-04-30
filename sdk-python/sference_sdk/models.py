from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


BatchStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class InferenceRequest(BaseModel):
    """Unified inference request schema for batches and streams.

    custom_id is user-provided and can be used for correlation.
    """

    model_config = ConfigDict(extra="forbid")

    custom_id: str | None = Field(default=None, description="User-provided identifier for correlation.")
    body: dict[str, Any] = Field(description="OpenAI-style request body. Must include `model`.")


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict[str, str]


class Batch(BaseModel):
    id: str
    status: BatchStatus
    window: Literal["24h"]
    request_count: int
    created_at: str
    updated_at: str
    completed_at: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0


class BatchList(BaseModel):
    items: list[Batch]


class BatchResults(BaseModel):
    batch_id: str
    status: BatchStatus
    output_url: str | None = None
    completed_at: str | None = None
    results: list[dict[str, Any]] | None = None


class BatchCreatePayload(BaseModel):
    window: Literal["24h"] = "24h"
    requests: list[InferenceRequest] = Field(min_length=1)


StreamStatus = Literal["open", "cancelled", "archived"]
StreamWindow = Literal["1h", "24h"]


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
    input: list[ResponseInputMessage] = Field(min_length=1)
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    include_reasoning: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


ReasoningFormat = Literal[
    "think_tag",
    "qwen_preamble",
    "provider_field",
    "anthropic_thinking",
    "openai_summary",
    "unknown",
]


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class ResponseOutputReasoning(BaseModel):
    type: Literal["reasoning"] = "reasoning"
    text: str
    format: ReasoningFormat


ResponseOutputContent = Annotated[
    ResponseOutputText | ResponseOutputReasoning,
    Field(discriminator="type"),
]


class ResponseUsage(BaseModel):
    """Token usage for a response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


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
