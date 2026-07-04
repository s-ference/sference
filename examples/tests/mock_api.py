"""Minimal in-process HTTP mock for sference batch + responses APIs."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


def _mock_completion_text(custom_id: str | None) -> str:
    label = custom_id or "row"
    return f"mock completion for {label}"


def _batch_result_row(custom_id: str | None) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "status": "completed",
        "result_json": {
            "choices": [
                {
                    "message": {
                        "content": _mock_completion_text(custom_id),
                    }
                }
            ]
        },
        "error_json": None,
    }


def _response_payload(response_id: str, *, custom_id: str | None = None) -> dict[str, Any]:
    text = _mock_completion_text(custom_id)
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1712345678,
        "model": "Qwen/Qwen3.6-35B-A3B",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_0",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "error": None,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


@dataclass
class MockSferenceState:
    batch_counter: int = 0
    response_counter: int = 0
    batches: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class MockSferenceHandler(BaseHTTPRequestHandler):
    state: MockSferenceState

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("expected JSON object")
        return parsed

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/v1/batches":
            body = self._read_json()
            requests = body.get("requests") or []
            with self.state.lock:
                self.state.batch_counter += 1
                batch_id = f"batch_mock_{self.state.batch_counter}"
                self.state.batches[batch_id] = requests
            self._send_json(
                201,
                {
                    "id": batch_id,
                    "status": "pending",
                    "window": body.get("window", "24h"),
                    "request_count": len(requests),
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "completed_at": None,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            return

        if path == "/v1/responses":
            body = self._read_json()
            metadata = body.get("metadata") or {}
            custom_id = metadata.get("custom_id")
            with self.state.lock:
                self.state.response_counter += 1
                response_id = f"resp_mock_{self.state.response_counter}"
                self.state.responses[response_id] = {"custom_id": custom_id}
            self._send_json(200, _response_payload(response_id, custom_id=custom_id))
            return

        self._send_json(404, {"detail": "not found"})

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        batch_match = re.fullmatch(r"/v1/batches/([^/]+)", path)
        if batch_match:
            batch_id = batch_match.group(1)
            requests = self.state.batches.get(batch_id, [])
            self._send_json(
                200,
                {
                    "id": batch_id,
                    "status": "completed",
                    "window": "24h",
                    "request_count": len(requests),
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:01+00:00",
                    "completed_at": "2026-01-01T00:00:01+00:00",
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            return

        results_match = re.fullmatch(r"/v1/batches/([^/]+)/results", path)
        if results_match:
            batch_id = results_match.group(1)
            requests = self.state.batches.get(batch_id, [])
            results = [
                _batch_result_row(req.get("custom_id") if isinstance(req, dict) else None)
                for req in requests
            ]
            self._send_json(
                200,
                {
                    "batch_id": batch_id,
                    "status": "completed",
                    "output_url": None,
                    "completed_at": "2026-01-01T00:00:01+00:00",
                    "results": results,
                },
            )
            return

        response_match = re.fullmatch(r"/v1/responses/([^/]+)", path)
        if response_match:
            response_id = response_match.group(1)
            meta = self.state.responses.get(response_id, {})
            self._send_json(
                200,
                _response_payload(response_id, custom_id=meta.get("custom_id")),
            )
            return

        self._send_json(404, {"detail": "not found"})


@dataclass
class MockSferenceServer:
    state: MockSferenceState = field(default_factory=MockSferenceState)
    _httpd: ThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("mock server not started")
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    def start(self) -> None:
        handler = type("BoundHandler", (MockSferenceHandler,), {"state": self.state})
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
