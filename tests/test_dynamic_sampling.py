"""Replica routing, drain and add of DPSamplingBackend on CPU."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator, cast

import httpx
import pytest
import ray
from fastapi.testclient import TestClient
from tinker import types

from tuft.backends.base_backend import BaseSamplingBackend
from tuft.backends.sampling_backend import DPSamplingBackend, VLLMSamplingBackend
from tuft.config import AppConfig, ModelConfig
from tuft.exceptions import ServiceUnavailableException
from tuft.oai.proxy import proxy_request
from tuft.sampling_controller import SamplingController, SamplingSessionRecord
from tuft.server import create_root_app


CONFIG = ModelConfig(model_name="m", model_path=Path("/dummy/m"), max_model_len=4096)


class Stub(BaseSamplingBackend):
    def __init__(self, fail: bool = False) -> None:
        super().__init__(CONFIG)
        self.fail = fail
        self.adapters: dict[str, str] = {}  # lora id -> replica state at load time
        self.stopped = False

    async def async_init(self) -> None:
        pass

    async def sample(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("engine error")

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        if self.fail:
            raise RuntimeError("load failed")
        self.adapters[lora_id] = self.state

    async def remove_adapter(self, lora_id: str) -> None:
        self.adapters.pop(lora_id, None)

    async def shutdown(self) -> None:
        self.stopped = True


async def _serving(*replicas: BaseSamplingBackend) -> DPSamplingBackend:
    dp = DPSamplingBackend(CONFIG, list(replicas))
    await dp.async_init()
    return dp


async def test_routes_to_fewest_in_flight() -> None:
    a, b = Stub(), Stub()
    dp = await _serving(a, b)
    assert dp.acquire() is a
    assert dp.acquire() is b
    dp.release(a)
    assert dp.acquire() is a


TURN1 = list(range(512))  # two full blocks


async def test_prefix_keeps_a_conversation_within_the_slack() -> None:
    a, b = Stub(), Stub()
    dp = await _serving(a, b)
    assert dp.acquire(TURN1) is a
    assert dp.acquire(TURN1 + [7] * 300) is a  # a has one more request than b


async def test_load_beats_prefix_above_the_slack() -> None:
    a, b = Stub(), Stub()
    dp = await _serving(a, b)
    dp.acquire(TURN1)
    a.requests += dp.PREFIX_SLACK  # a: 3 requests, b: 0
    assert dp.acquire(TURN1) is b


async def test_short_shared_prefix_spreads_by_load() -> None:
    a, b = Stub(), Stub()
    dp = await _serving(a, b)
    assert dp.acquire([1] * 100 + [2] * 200) is a
    assert dp.acquire([1] * 100 + [3] * 200) is b


async def test_prefix_lru_evicts_the_oldest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DPSamplingBackend, "PREFIX_LRU_CAP", 1)
    a = Stub()
    dp = await _serving(a)
    dp.acquire(TURN1)
    assert list(dp._served[a]) == [hash((hash((0, tuple(TURN1[:256]))), tuple(TURN1[256:])))]


async def test_failed_sample_releases_the_replica() -> None:
    a = Stub()
    dp = await _serving(a)
    with pytest.raises(RuntimeError):
        await dp.sample(types.ModelInput.from_ints([1]), 1, types.SamplingParams())
    assert a.requests == 0


async def test_stream_holds_the_counter_until_it_closes() -> None:
    sse = 'data: {"id": "x", "model": "b"}\n\ndata: [DONE]\n\n'
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text=sse))
    )
    closed: list[int] = []
    resp = await proxy_request(
        client,
        "http://vllm",
        "/v1/completions",
        {},
        True,
        user_model_name="m",
        response_id_prefix="p",
        on_close=lambda: closed.append(1),
    )
    stream = cast(AsyncGenerator[str, None], resp.body_iterator)  # type: ignore[union-attr]
    await anext(stream)
    assert closed == []
    await stream.aclose()  # the client went away mid-stream
    assert closed == [1]


def test_openai_router_releases_the_replica(tmp_path: Path) -> None:
    config = AppConfig(checkpoint_dir=tmp_path)
    config.supported_models = [CONFIG]
    config.authorized_users = {"key": "user"}
    with TestClient(create_root_app(config)) as client:
        replica = client.app.state.server_state.sampling._base_backends["m"]._replicas[0]  # type: ignore[attr-defined]
        for stream in (False, True):
            resp = client.post(
                "/oai/api/v1/completions",
                json={"model": "m", "prompt": "hi", "max_tokens": 2, "stream": stream},
                headers={"Authorization": "Bearer key"},
            )
            assert resp.status_code == 200
            assert replica.requests == 0


async def test_drain_waits_for_in_flight_then_stops() -> None:
    a, b = Stub(), Stub()
    dp = await _serving(a, b)
    dp.acquire()  # one request on a
    drain = asyncio.create_task(dp.drain(a, timeout_s=5))
    await asyncio.sleep(0.2)
    assert dp.acquire() is b  # a takes no new requests
    assert not a.stopped
    dp.release(a)
    await drain
    assert a.stopped
    assert dp._replicas == [b]


async def test_drain_kills_after_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def hang() -> None:
        await asyncio.Event().wait()

    replica = VLLMSamplingBackend.__new__(VLLMSamplingBackend)
    replica.engine = SimpleNamespace(shutdown=SimpleNamespace(remote=hang))  # type: ignore[assignment]
    replica._sweep_task = None
    replica.lora_adapters = {}
    killed: list[object] = []
    monkeypatch.setattr(ray, "kill", lambda actor, no_restart: killed.append(actor))
    replica.state = "serving"
    dp = DPSamplingBackend(CONFIG, [replica])
    dp.acquire()  # never released
    await dp.drain(replica, timeout_s=0.2)
    assert killed == [replica.engine]
    assert dp._replicas == []
    with pytest.raises(ServiceUnavailableException):
        dp.acquire()


async def test_add_replica_replays_adapters_before_serving(tmp_path: Path) -> None:
    config = AppConfig(checkpoint_dir=tmp_path)
    config.supported_models = [CONFIG]
    controller = SamplingController(config)
    old = Stub()
    dp = await _serving(old)
    controller._base_backends["m"] = dp
    controller.sampling_sessions["s1"] = SamplingSessionRecord(
        sampling_session_id="s1",
        session_id="t",
        model_id="s1",
        base_model="m",
        user_id="u",
        model_path=str(tmp_path),
        session_seq_id=0,
    )

    new = Stub()
    await controller.add_replica("m", new)
    assert new.adapters == {"s1": "pending"}
    assert new.state == "serving"

    broken = Stub(fail=True)
    with pytest.raises(RuntimeError):
        await controller.add_replica("m", broken)
    assert broken.stopped
    assert dp._replicas == [old, new]
    assert old.state == new.state == "serving"
    assert "s1" in controller.sampling_sessions
