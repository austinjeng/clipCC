from __future__ import annotations

import json
from typing import Callable

from app.errors.handlers import UploadConcurrencyError
from app.resource_gates import ResourceGates

# Type alias for ASGI app callable
ASGIApp = Callable


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
    ) -> None:
        self._app = app
        self._gates = gates
        self._api_key = api_key
        self._max_body_bytes = max_body_bytes

    def _check_auth(self, scope: dict) -> bool:
        """Return True if auth passes (key not configured or key matches)."""
        if self._api_key is None:
            return True
        headers: list[tuple[bytes, bytes]] = scope.get("headers", [])
        for name, value in headers:
            if name.lower() == b"x-api-key":
                return value.decode() == self._api_key
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

        # /api/v1/classify: auth → upload concurrency → body size
        if path == "/api/v1/classify":
            if not self._check_auth(scope):
                await _send_json_response(
                    send,
                    401,
                    {"detail": "Invalid or missing API key. Provide a valid key in the X-API-Key header."},
                )
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
                        if not response_started:
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

        # All other paths: pass through
        await self._app(scope, receive, send)
