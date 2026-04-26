import time
import pytest
from app.inference_runner import InferenceRunner

# Restrict anyio tests to asyncio backend — the implementation uses
# asyncio.Event and asyncio.get_running_loop() which are not compatible
# with trio.
@pytest.fixture
def anyio_backend():
    return "asyncio"

def slow_pipeline(cancel_event, runner_ref) -> str:
    for _ in range(50):
        if cancel_event.is_set():
            return "cancelled"
        time.sleep(0.05)
    return "done"

def fast_pipeline(cancel_event, runner_ref) -> str:
    return "done"

def failing_pipeline(cancel_event, runner_ref) -> str:
    raise ValueError("pipeline error")

@pytest.mark.anyio
async def test_successful_run():
    runner = InferenceRunner(timeout_seconds=10)
    result = await runner.run(fast_pipeline)
    assert result == "done"

@pytest.mark.anyio
async def test_timeout_cancels_and_returns_none():
    runner = InferenceRunner(timeout_seconds=0.3)
    result = await runner.run(slow_pipeline)
    assert result is None
    assert runner.timed_out

@pytest.mark.anyio
async def test_pipeline_error_propagates():
    runner = InferenceRunner(timeout_seconds=10)
    with pytest.raises(ValueError, match="pipeline error"):
        await runner.run(failing_pipeline)
