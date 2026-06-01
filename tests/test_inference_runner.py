import subprocess
import sys
import threading
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


@pytest.mark.anyio
async def test_timeout_waits_for_worker_to_finish():
    """On timeout, run() must not return until the worker thread has actually
    unwound — even if the worker outruns the deadline. Otherwise the caller's
    model lease is released while a leaked thread still holds the model
    (use-after-swap race). The old thread.join(timeout) returned early here."""
    finished = threading.Event()

    def slow_unwind_pipeline(cancel_event, runner_ref):
        try:
            # Ignores cancel for longer than the timeout (simulates an
            # uninterruptible torch forward pass already in flight).
            time.sleep(0.6)
            return "done"
        finally:
            finished.set()

    runner = InferenceRunner(timeout_seconds=0.2)
    result = await runner.run(slow_unwind_pipeline)
    assert result is None
    assert runner.timed_out
    assert finished.is_set()


@pytest.mark.anyio
async def test_timeout_kills_registered_process():
    """A subprocess registered with the runner is killed on timeout instead of
    being left to run to its own deadline."""
    proc_holder = {}

    def pipeline(cancel_event, runner_ref):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        proc_holder["proc"] = proc
        runner_ref.register_process(proc)
        try:
            proc.communicate(timeout=30)
        finally:
            runner_ref.unregister_process()
        return "done"

    runner = InferenceRunner(timeout_seconds=0.3)
    result = await runner.run(pipeline)
    assert result is None
    assert proc_holder["proc"].poll() is not None  # killed, not still running
