import pytest
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

@pytest.mark.anyio
async def test_body_size_rejection():
    app = make_test_app(max_body=100)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/classify", content=b"x" * 200)
        assert r.status_code == 413
