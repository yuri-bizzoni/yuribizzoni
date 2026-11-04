"""A stand-in for the Claude API: plays back scripted streaming replies and records the requests.

The real Anthropic SDK talks to it through a mocked HTTP transport, so the tests exercise the
same request building, streaming and parsing code that runs against the real API.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2


def text(words: str) -> list[dict[str, Any]]:
    return [
        {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": words}},
        {"type": "content_block_stop"},
    ]


def thinking() -> list[dict[str, Any]]:
    return [
        {"type": "content_block_start", "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
        {"type": "content_block_delta", "delta": {"type": "signature_delta", "signature": "c2lnbmF0dXJl"}},
        {"type": "content_block_stop"},
    ]


def tool_use(id: str, name: str, args: dict[str, Any], *, server: bool = False, raw: str | None = None) -> list[dict[str, Any]]:
    kind = "server_tool_use" if server else "tool_use"
    return [
        {"type": "content_block_start", "content_block": {"type": kind, "id": id, "name": name, "input": {}}},
        {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": raw if raw is not None else json.dumps(args)}},
        {"type": "content_block_stop"},
    ]


def search_results(tool_use_id: str) -> list[dict[str, Any]]:
    block = {
        "type": "web_search_tool_result",
        "tool_use_id": tool_use_id,
        "content": [{"type": "web_search_result", "url": "https://example.org/", "title": "Example", "encrypted_content": "e30=", "page_age": None}],
    }
    return [{"type": "content_block_start", "content_block": block}, {"type": "content_block_stop"}]


def fallback(from_model: str = "claude-opus-5-5", to_model: str = "claude-opus-4-8") -> list[dict[str, Any]]:
    block = {"type": "fallback", "from": {"model": from_model}, "to": {"model": to_model}}
    return [{"type": "content_block_start", "content_block": block}, {"type": "content_block_stop"}]


def reply(*blocks: list[dict[str, Any]], stop_reason: str = "end_turn", stop_details: dict[str, Any] | None = None) -> bytes:
    """One streamed message, as server-sent events."""
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
                "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        }
    ]
    for index, block in enumerate(blocks):
        events += [{**event, "index": index} for event in block]
    events += [
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details}, "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class ScriptedAPI:
    """Answers each request with the next scripted reply and keeps what was sent."""

    def __init__(self, *replies: bytes):
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx2.Headers] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(request.headers)
        if not self.replies:
            return httpx2.Response(500, json={"type": "error", "error": {"type": "api_error", "message": "script ran out"}})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=self.replies.pop(0))

    def client(self) -> anthropic.Anthropic:
        transport = httpx2.MockTransport(self)
        return anthropic.Anthropic(
            api_key="test-key",
            base_url="https://api.anthropic.test",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=transport),
        )
