from contextlib import asynccontextmanager
import anyio
from app.errors.handlers import InferenceConcurrencyError, UploadConcurrencyError

# anyio.WouldBlock and trio.WouldBlock are unrelated exception types; catch both.
try:
    import trio as _trio
    _WOULD_BLOCK = (anyio.WouldBlock, _trio.WouldBlock)
except ImportError:
    _WOULD_BLOCK = (anyio.WouldBlock,)


class ResourceGates:
    def __init__(self, max_upload_concurrency: int = 4, max_inference_concurrency: int = 2):
        self._upload_limiter = anyio.CapacityLimiter(max_upload_concurrency)
        self._inference_limiter = anyio.CapacityLimiter(max_inference_concurrency)

    @asynccontextmanager
    async def upload_admission(self):
        token = object()
        try:
            self._upload_limiter.acquire_on_behalf_of_nowait(token)
        except _WOULD_BLOCK:
            raise UploadConcurrencyError()
        try:
            yield
        finally:
            self._upload_limiter.release_on_behalf_of(token)

    @asynccontextmanager
    async def inference_admission(self):
        token = object()
        try:
            self._inference_limiter.acquire_on_behalf_of_nowait(token)
        except _WOULD_BLOCK:
            raise InferenceConcurrencyError()
        try:
            yield
        finally:
            self._inference_limiter.release_on_behalf_of(token)
