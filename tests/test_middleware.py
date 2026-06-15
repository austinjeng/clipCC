import pytest
from contextlib import asynccontextmanager
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from app.middleware import RequestGateMiddleware
from app.resource_gates import ResourceGates

async def echo_app(request):
    body = await request.body()
    return JSONResponse({"size": len(body)})

async def health_app(request):
    return JSONResponse({"status": "ok"})

def make_test_app(api_key=None, max_body=1024, max_upload=2):
    gates = ResourceGates(max_upload_concurrency=max_upload, max_inference_concurrency=1)
    app = Starlette(routes=[
        Route("/api/v1/classify", echo_app, methods=["POST"]),
        Route("/live", health_app, methods=["GET"]),
        Route("/ready", health_app, methods=["GET"]),
    ])
    return RequestGateMiddleware(app, gates=gates, api_key=api_key, max_body_bytes=max_body)

@pytest.mark.anyio
async def test_live_bypasses_all_gates():
    app = make_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/live")
        assert r.status_code == 200

@pytest.mark.anyio
async def test_ready_bypasses_concurrency_but_checks_auth():
    app = make_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/ready")
        assert r.status_code == 401
        r = await c.get("/ready", headers={"X-API-Key": "secret"})
        assert r.status_code == 200

@pytest.mark.anyio
async def test_auth_rejects_missing_key():
    app = make_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"data")
        assert r.status_code == 401

@pytest.mark.anyio
async def test_auth_accepts_valid_key():
    app = make_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"data", headers={"X-API-Key": "secret"})
        assert r.status_code == 200

@pytest.mark.anyio
async def test_no_auth_when_key_not_configured():
    app = make_test_app(api_key=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"data")
        assert r.status_code == 200

def test_check_auth_valid_key_accepted():
    app = make_test_app(api_key="secret")
    assert app._check_auth({"headers": [(b"x-api-key", b"secret")]}) is True

def test_check_auth_wrong_key_rejected():
    app = make_test_app(api_key="secret")
    assert app._check_auth({"headers": [(b"x-api-key", b"wrong")]}) is False

def test_check_auth_non_utf8_key_rejected():
    # A non-UTF-8 header value must yield a clean reject, not a 500 from decode.
    app = make_test_app(api_key="secret")
    assert app._check_auth({"headers": [(b"x-api-key", b"\xff\xfe")]}) is False


def test_check_auth_blank_key_never_authenticates():
    # Defense-in-depth: a blank key reaching the middleware must reject every
    # request, including one sending an empty X-API-Key header (b"" == b"" trap).
    app = make_test_app(api_key="")
    assert app._check_auth({"headers": [(b"x-api-key", b"")]}) is False
    assert app._check_auth({"headers": []}) is False
    app_ws = make_test_app(api_key="   ")
    assert app_ws._check_auth({"headers": [(b"x-api-key", b"   ")]}) is False

@pytest.mark.anyio
async def test_body_size_rejection():
    app = make_test_app(max_body=100)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"x" * 200)
        assert r.status_code == 413


async def swallowing_app(request):
    # Reproduces FastAPI multipart form parsing: the body read happens inside a
    # try/except that catches ANY exception during parse and converts it into the
    # app's own response. This swallows _BodyTooLargeSignal so it never propagates
    # out of the ASGI app — the failure mode the 413 path must still handle.
    try:
        await request.body()
    except Exception:
        return JSONResponse({"detail": "parse failed"}, status_code=400)
    return JSONResponse({"ok": True})


@pytest.mark.anyio
async def test_body_size_rejection_when_app_swallows_signal():
    # Regression: when the inner app catches the oversize signal (as FastAPI's
    # multipart parser does) the middleware must still emit a real 413, not hang
    # the client with zero ASGI messages.
    gates = ResourceGates(max_upload_concurrency=2, max_inference_concurrency=1)
    inner = Starlette(routes=[Route("/api/v1/classify", swallowing_app, methods=["POST"])])
    app = RequestGateMiddleware(inner, gates=gates, api_key=None, max_body_bytes=100)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"x" * 200)
        assert r.status_code == 413
        assert "maximum allowed size" in r.json()["detail"]


@pytest.mark.anyio
async def test_under_limit_swallowing_app_unaffected():
    # The post-call 413 check must not fire for normal under-limit requests.
    gates = ResourceGates(max_upload_concurrency=2, max_inference_concurrency=1)
    inner = Starlette(routes=[Route("/api/v1/classify", swallowing_app, methods=["POST"])])
    app = RequestGateMiddleware(inner, gates=gates, api_key=None, max_body_bytes=100)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"x" * 10)
        assert r.status_code == 200
        assert r.json() == {"ok": True}


class NoUploadGates(ResourceGates):
    """Test double: any path that consumes upload admission fails the test.
    (CapacityLimiter requires >= 1 token, so 'capacity 0' can't be expressed
    directly — assert non-consumption instead.)"""

    @asynccontextmanager
    async def upload_admission(self):
        raise AssertionError("upload_admission must not be consumed by this path")
        yield  # pragma: no cover


def make_gemma_test_app(api_key=None, max_body=1024, max_upload=2,
                        vlm_state=lambda: "loaded", gates_cls=ResourceGates):
    gates = gates_cls(max_upload_concurrency=max_upload, max_inference_concurrency=1)
    app = Starlette(routes=[
        Route("/api/v1/classify", echo_app, methods=["POST"]),
        Route("/api/v1/gemma/label_scores", echo_app, methods=["POST"]),
        Route("/api/v1/gemma/qa", echo_app, methods=["POST"]),
        Route("/api/v1/gemma/status", health_app, methods=["GET"]),
        Route("/api/v1/gemma/warm", health_app, methods=["POST"]),
        Route("/gemma", health_app, methods=["GET"]),
        Route("/live", health_app, methods=["GET"]),
    ])
    return RequestGateMiddleware(app, gates=gates, api_key=api_key,
                                 max_body_bytes=max_body, vlm_state=vlm_state)


@pytest.mark.anyio
async def test_gemma_upload_route_requires_auth():
    app = make_gemma_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", content=b"data")
        assert r.status_code == 401


@pytest.mark.anyio
async def test_gemma_upload_route_enforces_body_size():
    app = make_gemma_test_app(api_key=None, max_body=10)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/qa", content=b"x" * 100)
        assert r.status_code == 413


@pytest.mark.anyio
async def test_gemma_cold_post_503_before_body():
    app = make_gemma_test_app(api_key=None, vlm_state=lambda: "idle")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", content=b"data")
        assert r.status_code == 503
        assert "Retry-After" in r.headers


@pytest.mark.anyio
async def test_gemma_loading_post_503():
    app = make_gemma_test_app(api_key=None, vlm_state=lambda: "loading")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/qa", content=b"data")
        assert r.status_code == 503


@pytest.mark.anyio
async def test_gemma_status_auth_only_no_upload_gate():
    # NoUploadGates raises if any route consumes upload admission —
    # status must NOT consume an upload slot.
    app = make_gemma_test_app(api_key="secret", gates_cls=NoUploadGates)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/api/v1/gemma/status", headers={"X-API-Key": "secret"})
        assert r.status_code == 200
        r = await c.get("/api/v1/gemma/status")
        assert r.status_code == 401


@pytest.mark.anyio
async def test_gemma_warm_auth_only():
    app = make_gemma_test_app(api_key="secret", gates_cls=NoUploadGates)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/warm", headers={"X-API-Key": "secret"})
        assert r.status_code == 200
        r = await c.post("/api/v1/gemma/warm")
        assert r.status_code == 401


@pytest.mark.anyio
async def test_gemma_page_passes_through():
    app = make_gemma_test_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/gemma")  # no auth, like /
        assert r.status_code == 200


@pytest.mark.anyio
async def test_classify_unaffected_by_vlm_state():
    app = make_gemma_test_app(api_key=None, vlm_state=lambda: "idle")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"data")
        assert r.status_code == 200


@pytest.mark.anyio
async def test_gemma_cold_unauthenticated_gets_401_not_503():
    # Auth ordering invariant: slot state must not be disclosed pre-auth
    app = make_gemma_test_app(api_key="secret", vlm_state=lambda: "idle")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/gemma/label_scores", content=b"data")
        assert r.status_code == 401
