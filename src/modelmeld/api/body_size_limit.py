# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 ModelMeld.

"""BodySizeLimitMiddleware — reject oversized request bodies with 413.

Pure-ASGI implementation (NOT BaseHTTPMiddleware) so SSE streaming and other
long-lived response bodies are not affected. Checks `Content-Length` on the
request side before any body reads happen.

Two enforcement paths:

1. `Content-Length` header present (typical) — reject up front with 413 if it
   exceeds the per-path or default limit. No body bytes are read.

2. `Content-Length` missing (chunked / streaming uploads) — count bytes as we
   stream them in via the `receive` callable; raise 413 the moment we cross
   the limit. Prevents unbounded body streams from exhausting memory.

Per-path overrides via `path_limits` (longest-prefix match wins). Default
limit applies to anything not covered by an override.
"""

from __future__ import annotations

import json
from typing import Any


class BodySizeLimitMiddleware:
    """Pure-ASGI body-size cap with per-path overrides."""

    def __init__(
        self,
        app: Any,
        *,
        default_max_bytes: int = 4 * 1024 * 1024,
        path_limits: dict[str, int] | None = None,
    ) -> None:
        self.app = app
        self.default_max_bytes = default_max_bytes
        # Sort by descending prefix length so longest-prefix wins on lookup.
        self._sorted_limits = sorted(
            (path_limits or {}).items(),
            key=lambda kv: -len(kv[0]),
        )

    def _limit_for_path(self, path: str) -> int:
        for prefix, limit in self._sorted_limits:
            if path.startswith(prefix):
                return limit
        return self.default_max_bytes

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        max_bytes = self._limit_for_path(path)

        # Check Content-Length header first — most well-behaved clients send it.
        content_length: int | None = None
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    content_length = int(value.decode("ascii"))
                except (ValueError, UnicodeDecodeError):
                    content_length = None
                break

        if content_length is not None and content_length > max_bytes:
            await _send_413(send, max_bytes)
            return

        # No Content-Length → wrap `receive` to count bytes as they stream.
        if content_length is None:
            wrapped_receive = _make_size_counting_receive(receive, max_bytes, send)
            await self.app(scope, wrapped_receive, send)
        else:
            await self.app(scope, receive, send)


def _make_size_counting_receive(receive: Any, max_bytes: int, send: Any) -> Any:
    """Wrap `receive` so streaming bodies are capped at max_bytes."""
    total = 0
    limit_hit = False

    async def wrapped_receive() -> Any:
        nonlocal total, limit_hit
        if limit_hit:
            # Once we've sent 413 we still need to drain the receive queue
            # without exposing more body to the inner app.
            return {"type": "http.disconnect"}
        message = await receive()
        if message.get("type") == "http.request":
            body = message.get("body") or b""
            total += len(body)
            if total > max_bytes:
                limit_hit = True
                await _send_413(send, max_bytes)
                return {"type": "http.disconnect"}
        return message

    return wrapped_receive


async def _send_413(send: Any, max_bytes: int) -> None:
    body = json.dumps({
        "error": "payload_too_large",
        "detail": f"Request body exceeds limit of {max_bytes} bytes.",
        "max_bytes": max_bytes,
    }).encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": 413,
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
    })
    await send({
        "type": "http.response.body",
        "body": body,
    })
