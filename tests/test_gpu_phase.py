import asyncio

import pytest

from tuft.backends.gpu_phase import GpuPhase


async def test_switch_waits_for_holders_and_blocks_new_entrants() -> None:
    log: list[str] = []

    async def to_train() -> None:
        log.append("to_train")

    async def to_sample() -> None:
        log.append("to_sample")

    gpu = GpuPhase(to_train, to_sample)
    release = asyncio.Event()

    async def work(phase, name: str, hold: bool = False) -> None:
        async with gpu.use(phase):
            log.append(f"start {name}")
            if hold:
                await release.wait()
            log.append(f"end {name}")

    s1 = asyncio.create_task(work("sample", "s1", hold=True))
    s2 = asyncio.create_task(work("sample", "s2", hold=True))
    await asyncio.sleep(0.01)
    t1 = asyncio.create_task(work("train", "t1"))
    await asyncio.sleep(0.01)
    s3 = asyncio.create_task(work("sample", "s3"))
    await asyncio.sleep(0.01)
    assert log == ["start s1", "start s2"]

    release.set()
    await asyncio.gather(s1, s2, t1, s3)
    assert log == [
        "start s1",
        "start s2",
        "end s1",
        "end s2",
        "to_train",
        "start t1",
        "end t1",
        "to_sample",
        "start s3",
        "end s3",
    ]
    assert gpu.phase == "sample"


async def test_failed_switch_is_retried() -> None:
    calls: list[str] = []

    async def to_train() -> None:
        calls.append("to_train")
        if len(calls) == 1:
            raise RuntimeError("boom")

    async def to_sample() -> None:
        calls.append("to_sample")

    gpu = GpuPhase(to_train, to_sample)
    with pytest.raises(RuntimeError):
        async with gpu.use("train"):
            pass
    async with gpu.use("sample"):
        pass
    assert calls == ["to_train", "to_sample"]
