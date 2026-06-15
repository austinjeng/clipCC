from __future__ import annotations

import hmac
import json
from typing import Callable

from app.errors.handlers import UploadConcurrencyError
from app.resource_gates import ResourceGates

# Type alias for ASGI app callable
ASGIApp = Callable

# Route policy (spec §7): video-upload POST routes get the full gate stack;
# status/warm are auth-only; the /gemma page passes through like /.
GEMMA_UPLOAD_PATHS = frozenset({
    "/api/v1/gemma/label_scores",
    "/api/v1/gemma/qa",
    "/api/v1/gemma/events",
})
GEMMA_AUTH_ONLY_PATHS = frozenset({
    "/api/v1/gemma/status",
    "/api/v1/gemma/warm",
})


class _BodyTooLargeSignal(Exception):
    pass


class _SizeChecker:
    """Wraps an ASGI receive callable and raises _BodyTooLargeSignal if the body exceeds the limit."""

    def __init__(self, receive: Callable, max_bytes: int) -> None:
        self._receive = receive
        self._max_bytes = max_bytes
        self._seen = 0

    async def __call__(self) -> dict:
        message = await self._receive()
        if message.get("type") == "http.request":
            chunk = message.get("body", b"")
            self._seen += len(chunk)
            if self._seen > self._max_bytes:
                raise _BodyTooLargeSignal()
        return message


async def _send_json_response(send: Callable, status_code: int, body: dict) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload, "more_body": False})


class RequestGateMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        gates: ResourceGates,
        api_key: str | None,
        max_body_bytes: int,
        vlm_state: Callable | None = None,
    ) -> None:
        self._app = app
        self._gates = gates
        self._api_key = api_key
        self._max_body_bytes = max_body_bytes
        self._vlm_state = vlm_state

    def _check_auth(self, scope: dict) -> bool:
        """Return True if auth passes (key not configured or key matches)."""
        if self._api_key is None:
            return True
        if not self._api_key.strip():
            # A blank/whitespace key is a misconfiguration: never authenticate
            # with it (prevents an empty X-API-Key header from matching b"").
            return False
        expected = self._api_key.encode()
        headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        for name, value in headers:
            if name.lower() == b"x-api-key":
                # Compare raw bytes with a constant-time check: timing-safe, and
                # a non-UTF-8 header value simply fails to match (no 500 from a
                # decode error).
                return hmac.compare_digest(value, expected)
        return False

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path: str = scope.get("path", "")

        # /live is fully exempt from all gates
        if path == "/live":
            await self._app(scope, receive, send)
            return

        # /ready checks auth but is exempt from concurrency gates
        if path == "/ready":
            if not self._check_auth(scope):
                await _send_json_response(
                    send,
                    401,
                    {"detail": "Invalid or missing API key. Provide a valid key in the X-API-Key header."},
                )
                return
            await self._app(scope, receive, send)
            return

        # /api/v1/classify + Gemma upload routes: auth → [slot-state gate] → upload concurrency → body size
        is_gemma_upload = path in GEMMA_UPLOAD_PATHS

        if path == "/api/v1/classify" or is_gemma_upload:
            if not self._check_auth(scope):
                await _send_json_response(
                    send,
                    401,
                    {"detail": "Invalid or missing API key. Provide a valid key in the X-API-Key header."},
                )
                return

            if is_gemma_upload:
                state = self._vlm_state() if self._vlm_state is not None else "idle"
                if state != "loaded":
                    # Fail fast BEFORE draining the multipart body: a cold
                    # request must not stream 500MB while no model can serve it.
                    payload = json.dumps({
                        "detail": f"Gemma model is not loaded (state: {state}). "
                                  f"Trigger loading via POST /api/v1/gemma/warm and poll "
                                  f"GET /api/v1/gemma/status."
                    }).encode()
                    await send({
                        "type": "http.response.start",
                        "status": 503,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(payload)).encode()),
                            (b"retry-after", b"10"),
                        ],
                    })
                    await send({"type": "http.response.body", "body": payload, "more_body": False})
                    return

            try:
                async with self._gates.upload_admission():
                    size_checker = _SizeChecker(receive, self._max_body_bytes)
                    # Wrap send to track whether response has started so we can
                    # intercept before Starlette flushes headers on 413.
                    response_started = False
                    captured_signal: list[_BodyTooLargeSignal] = []

                    async def intercepting_send(message: dict) -> None:
                        nonlocal response_started
                        if captured_signal:
                            # Suppress anything the app tries to send after we
                            # detected an oversize body.
                            return
                        if message.get("type") == "http.response.start":
                            response_started = True
                        await send(message)

                    # Replace the size checker to capture signal before app writes
                    async def guarded_receive() -> dict:
                        try:
                            return await size_checker()
                        except _BodyTooLargeSignal as exc:
                            captured_signal.append(exc)
                            raise

                    try:
                        await self._app(scope, guarded_receive, intercepting_send)
                    except _BodyTooLargeSignal:
                        # Some app stacks let the signal propagate out; others
                        # (FastAPI multipart form parsing) catch it internally and
                        # convert it into their own response, which intercepting_send
                        # suppresses. Either way the 413 is emitted below.
                        pass

                    # Emit the 413 whether the signal propagated out or was
                    # swallowed inside the app — as long as no real response has
                    # already been flushed to the client.
                    if captured_signal and not response_started:
                        max_mb = self._max_body_bytes / (1024 * 1024)
                        await _send_json_response(
                            send,
                            413,
                            {
                                "detail": (
                                    f"Request body exceeds the maximum allowed size of {max_mb:.1f} MB."
                                )
                            },
                        )
            except UploadConcurrencyError:
                await _send_json_response(
                    send,
                    429,
                    {"detail": "Too many uploads in progress. Please retry in a moment."},
                )
            return

        # /api/v1/models*: auth required, no body size or concurrency gates
        if path.startswith("/api/v1/models"):
            if not self._check_auth(scope):
                await _send_json_response(
                    send,
                    401,
                    {"detail": "Invalid or missing API key. Provide a valid key in the X-API-Key header."},
                )
                return
            await self._app(scope, receive, send)
            return

        # Gemma status/warm: auth only — must NOT consume upload slots or body plumbing
        if path in GEMMA_AUTH_ONLY_PATHS:
            if not self._check_auth(scope):
                await _send_json_response(
                    send,
                    401,
                    {"detail": "Invalid or missing API key. Provide a valid key in the X-API-Key header."},
                )
                return
            await self._app(scope, receive, send)
            return

        # All other paths: pass through
        await self._app(scope, receive, send)
