import pytest
import anyio
from app.resource_gates import ResourceGates
from app.errors.handlers import UploadConcurrencyError, InferenceConcurrencyError

@pytest.fixture
def gates():
    return ResourceGates(max_upload_concurrency=2, max_inference_concurrency=1)

@pytest.mark.anyio
async def test_upload_admission_allows_within_limit(gates):
    async with gates.upload_admission():
        pass

@pytest.mark.anyio
async def test_upload_admission_rejects_over_limit(gates):
    async with gates.upload_admission():
        async with gates.upload_admission():
            with pytest.raises(UploadConcurrencyError):
                async with gates.upload_admission():
                    pass

@pytest.mark.anyio
async def test_inference_admission_allows_within_limit(gates):
    async with gates.inference_admission():
        pass

@pytest.mark.anyio
async def test_inference_admission_rejects_over_limit(gates):
    async with gates.inference_admission():
        with pytest.raises(InferenceConcurrencyError):
            async with gates.inference_admission():
                pass

@pytest.mark.anyio
async def test_upload_slot_released_after_exception(gates):
    with pytest.raises(ValueError):
        async with gates.upload_admission():
            raise ValueError("boom")
    async with gates.upload_admission():
        pass
