import asyncio
import subprocess
import threading
from typing import Any, Callable, Optional
import anyio


class InferenceRunner:
    def __init__(self, timeout_seconds: float = 300):
        self.timeout_seconds = timeout_seconds
        self.cancel_event = threading.Event()
        self.active_process: Optional[subprocess.Popen] = None
        self.timed_out = False
        self._lock = threading.Lock()

    def register_process(self, proc: Optional[subprocess.Popen]) -> None:
        with self._lock:
            self.active_process = proc

    def unregister_process(self) -> None:
        with self._lock:
            self.active_process = None

    def _kill_active_process(self) -> None:
        with self._lock:
            if self.active_process is not None:
                try:
                    self.active_process.kill()
                except OSError:
                    pass

    async def run(
        self, pipeline: Callable[[threading.Event, "InferenceRunner"], Any],
    ) -> Optional[Any]:
        result_holder: dict[str, Any] = {}
        error_holder: dict[str, BaseException] = {}
        done = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                result_holder["value"] = pipeline(self.cancel_event, self)
            except BaseException as e:
                error_holder["value"] = e
            finally:
                loop.call_soon_threadsafe(done.set)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        try:
            with anyio.fail_after(self.timeout_seconds):
                await done.wait()
        except TimeoutError:
            self.timed_out = True
            self.cancel_event.set()
            self._kill_active_process()
            thread.join(timeout=self.timeout_seconds)
            return None

        thread.join(timeout=5)

        if "value" in error_holder:
            raise error_holder["value"]

        return result_holder.get("value")
