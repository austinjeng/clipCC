import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from app.middleware import RequestGateMiddleware
from app.resource_gates import ResourceGates


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def echo_app(request):
    body = await request.body()
    return JSONResponse({"size": len(body)})


def make_app(api_key=None, max_body=1024, vlm="loaded"):
    gates = ResourceGates(max_upload_concurrency=2, max_inference_concurrency=1)
    app = Starlette(routes=[Route("/api/v1/hybrid", echo_app, methods=["POST"])])
    return RequestGateMiddleware(
        app, gates=gates, api_key=api_key, max_body_bytes=max_body, vlm_state=lambda: vlm
    )


@pytest.mark.anyio
async def test_hybrid_requires_auth():
    app = make_app(api_key="secret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/hybrid", content=b"data")
        assert r.status_code == 401


@pytest.mark.anyio
async def test_hybrid_cold_gemma_503_before_body():
    app = make_app(vlm="idle")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/hybrid", content=b"data")
        assert r.status_code == 503
        assert r.headers.get("retry-after") == "10"


@pytest.mark.anyio
async def test_hybrid_oversized_body_413():
    app = make_app(max_body=100, vlm="loaded")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/hybrid", content=b"x" * 200)
        assert r.status_code == 413


@pytest.mark.anyio
async def test_hybrid_loaded_passes_through():
    app = make_app(vlm="loaded")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.post("/api/v1/hybrid", content=b"hello")
        assert r.status_code == 200
        assert r.json()["size"] == 5
