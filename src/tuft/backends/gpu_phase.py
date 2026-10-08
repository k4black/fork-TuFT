"""Lazy switching of one GPU between a training phase and a sampling phase."""

import asyncio
from contextlib import asynccontextmanager, nullcontext
from typing import Any, AsyncIterator, Awaitable, Callable, Literal


Phase = Literal["train", "sample"]


class GpuPhase:
    """Work of the current phase runs concurrently. A switch waits for that work
    to drain, and while a switch is pending no new work of the current phase starts.
    """

    def __init__(
        self,
        to_train: Callable[[], Awaitable[None]],
        to_sample: Callable[[], Awaitable[None]],
        phase: Phase = "sample",
    ) -> None:
        self.phase: Phase = phase
        self._switch = {"train": to_train, "sample": to_sample}
        self._pending: Phase | None = None
        self._holders = 0
        self._cond = asyncio.Condition()

    @asynccontextmanager
    async def use(self, phase: Phase) -> AsyncIterator[None]:
        async with self._cond:
            while self._pending is not None or self.phase != phase:
                if self._pending is None:
                    self._pending = phase
                    try:
                        await self._cond.wait_for(lambda: self._holders == 0)
                        await self._switch[phase]()
                        self.phase = phase
                    finally:
                        self._pending = None
                        self._cond.notify_all()
                else:
                    await self._cond.wait()
            self._holders += 1
        try:
            yield
        finally:
            async with self._cond:
                self._holders -= 1
                self._cond.notify_all()


def use_phase(phases: dict[str, GpuPhase], model_name: str, phase: Phase):
    """``phases[model_name].use(phase)``, or a no-op for a model without one."""
    gpu = phases.get(model_name)
    return gpu.use(phase) if gpu is not None else nullcontext()


def sleep_phase(trainer: Any, engine: Any) -> GpuPhase:
    """Colocate "sleep": the vLLM engine and the HF trainer actors take turns on the GPU."""

    async def to_train() -> None:
        await engine.sleep.remote()
        await trainer.onload.remote()

    async def to_sample() -> None:
        await trainer.offload.remote()
        await engine.wake_up.remote()

    return GpuPhase(to_train, to_sample)
