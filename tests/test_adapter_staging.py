"""LoRA adapter staging: adapter bytes -> node-local dir -> path handed to vLLM."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from tinker import types

from tuft.backends.sampling_backend import VLLMSamplingBackend
from tuft.backends.vllm_engine import VLLMEngine
from tuft.checkpoints import read_adapter_files


# Carries both ':' and '/', as minted by oai.model_resolver.
LORA_ID = "run-1:run-1/checkpoint-0001"


def _engine(tmp_path: Path) -> VLLMEngine:
    """An engine with only its staging state; __init__ would import vllm."""
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._adapter_root = tmp_path / "staging"
    return engine


def _adapter_dir(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "checkpoint" / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text('{"r": 8}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"\x00weights\xff")
    (adapter_dir / "adapter.pt").write_bytes(b"fsdp-internal")
    (adapter_dir / "optimizer.pt").write_bytes(b"fsdp-internal")
    return adapter_dir


def _remote(fn):
    """Mimic a Ray actor method handle: ``handle.method.remote(...)``."""

    async def _call(*args, **kwargs):
        return fn(*args, **kwargs)

    return SimpleNamespace(remote=_call)


async def test_stage_adapter_round_trip(tmp_path):
    files = read_adapter_files(_adapter_dir(tmp_path))
    assert set(files) == {"adapter_config.json", "adapter_model.safetensors"}

    engine = _engine(tmp_path)
    staged = Path(await engine.stage_adapter(LORA_ID, files))

    # One flat directory despite the ':' and '/' in the adapter id.
    assert staged.parent == engine._adapter_root
    assert read_adapter_files(staged) == files

    await engine.unstage_adapter(LORA_ID)
    assert not staged.exists()


def test_read_adapter_files_requires_config(tmp_path):
    # vLLM reads a path without a config as a HF repo id and downloads it.
    adapter_dir = _adapter_dir(tmp_path)
    (adapter_dir / "adapter_config.json").unlink()
    with pytest.raises(ValueError, match="adapter_config.json"):
        read_adapter_files(adapter_dir)


STAGED = "/dev/shm/tuft-adapters-0123/staged"
OAI_URL = "http://vllm-node:8000"
# The OpenAI API registers "{training_run_id}:{checkpoint_id}"; sampling
# sessions register their own random id. Both live in one backend.
OAI_NAME = "run-1:checkpoint-0001"
SESSION_ID = "6f1c0b64-session"


def _backend(
    monkeypatch,
    calls: list,
    statuses: dict[str, int] | None = None,
    *,
    idle_ttl: float = 0.0,
) -> VLLMSamplingBackend:
    """Stub engine, OpenAI server and vllm import; calls land in ``calls``.

    ``statuses`` maps an endpoint to its status; 0 is a transport failure.
    """
    import httpx

    statuses = statuses or {}

    vllm = ModuleType("vllm")
    vllm_lora = ModuleType("vllm.lora")
    vllm_lora_request = ModuleType("vllm.lora.request")
    vllm_lora_request.LoRARequest = SimpleNamespace  # type: ignore[attr-defined]
    vllm_lora.request = vllm_lora_request  # type: ignore[attr-defined]
    vllm.lora = vllm_lora  # type: ignore[attr-defined]
    # Every parent package is imported before the leaf, so all three entries
    # must be present for `from vllm.lora.request import ...` to resolve.
    for name, module in [
        ("vllm", vllm),
        ("vllm.lora", vllm_lora),
        ("vllm.lora.request", vllm_lora_request),
    ]:
        monkeypatch.setitem(sys.modules, name, module)

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, post_url: str, json: dict, **_kwargs):
            calls.append(("post", post_url, json))
            status = statuses.get(post_url.rsplit("/", 1)[-1], 200)
            if status == 0:
                raise httpx.ConnectError("engine unreachable")
            return SimpleNamespace(status_code=status, text="nope")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *_a, **_kw: _FakeClient())

    def _stage(lora_id: str, files: dict[str, bytes]) -> str:
        calls.append(("stage", lora_id, files))
        return STAGED

    def _add_lora(request) -> bool:
        calls.append(("add_lora", request))
        return True

    backend = VLLMSamplingBackend.__new__(VLLMSamplingBackend)
    backend.engine = SimpleNamespace(  # type: ignore[assignment]
        stage_adapter=_remote(_stage),
        unstage_adapter=_remote(lambda lora_id: calls.append(("unstage", lora_id))),
        add_lora=_remote(_add_lora),
        remove_lora=_remote(lambda int_id: calls.append(("remove_lora", int_id))),
    )
    backend.lora_adapters = {}
    backend._oai_loaded = set()
    backend._adapter_paths = {}
    backend._last_used = {}
    backend._idle_ttl = idle_ttl
    backend._sweep_task = None
    backend._counter = 1
    backend._lock = asyncio.Lock()
    backend._openai_api_url = OAI_URL
    return backend


@pytest.mark.parametrize(
    "unload_status, confirmed",
    [
        pytest.param(404, True, id="404-no-such-adapter"),
        pytest.param(400, False, id="400-rejected-not-an-answer"),
        pytest.param(503, False, id="503-may-still-hold-it"),
        pytest.param(0, False, id="transport-failure"),
    ],
)
async def test_unstaging_waits_for_a_confirmed_unload(
    tmp_path, monkeypatch, unload_status, confirmed
):
    calls: list[tuple] = []
    backend = _backend(monkeypatch, calls, {"unload_lora_adapter": unload_status})
    await backend.ensure_oai_lora_loaded(OAI_NAME, _adapter_dir(tmp_path))

    await backend.remove_adapter(OAI_NAME)

    assert (("unstage", OAI_NAME) in calls) is confirmed
    # An unconfirmed unload keeps the name so a later removal retries it.
    assert (OAI_NAME in backend._oai_loaded) is not confirmed


async def test_sample_transparently_readds_a_swept_adapter(tmp_path, monkeypatch):
    adapter_dir = _adapter_dir(tmp_path)
    calls: list[tuple] = []
    backend = _backend(monkeypatch, calls, idle_ttl=60.0)
    backend.engine.generate = _remote(  # type: ignore[attr-defined]
        lambda **_kwargs: SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    finish_reason="stop",
                    token_ids=[7],
                    logprobs=[{7: SimpleNamespace(logprob=-0.1)}],
                )
            ]
        )
    )
    await backend.add_adapter(SESSION_ID, adapter_dir)
    # The engine got the bytes, and vLLM got the engine's own node-local path.
    assert ("stage", SESSION_ID, read_adapter_files(adapter_dir)) in calls
    assert backend.lora_adapters[SESSION_ID].lora_path == STAGED

    backend._last_used[SESSION_ID] -= 61
    await backend._sweep_idle_adapters()
    assert SESSION_ID not in backend.lora_adapters

    response = await backend.sample(
        prompt=types.ModelInput.from_ints([1, 2, 3]),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=1),
        lora_id=SESSION_ID,
    )

    assert response.sequences[0].tokens == [7]
    assert backend.lora_adapters[SESSION_ID].lora_path == STAGED
    assert backend._last_used[SESSION_ID] > time.monotonic() - 60


async def test_sweep_skips_an_adapter_refreshed_while_it_runs(tmp_path, monkeypatch):
    adapter_dir = _adapter_dir(tmp_path)
    calls: list[tuple] = []
    backend = _backend(monkeypatch, calls, idle_ttl=60.0)
    await backend.add_adapter(SESSION_ID, adapter_dir)
    await backend.ensure_oai_lora_loaded(OAI_NAME, adapter_dir)
    backend._last_used[SESSION_ID] -= 61
    backend._last_used[OAI_NAME] -= 61  # both look idle when the sweep starts

    # Holding the lock the way a request does: the sweep snapshots, then blocks.
    async with backend._lock:
        sweep = asyncio.create_task(backend._sweep_idle_adapters())
        await asyncio.sleep(0)
        backend._last_used[OAI_NAME] = time.monotonic()  # a request lands meanwhile
    await sweep

    assert ("unstage", SESSION_ID) in calls  # still idle
    assert SESSION_ID not in backend.lora_adapters
    assert ("unstage", OAI_NAME) not in calls  # refreshed after the snapshot
    assert backend._oai_loaded == {OAI_NAME}


async def test_swept_oai_name_reloads_on_the_next_request(tmp_path, monkeypatch):
    adapter_dir = _adapter_dir(tmp_path)
    calls: list[tuple] = []
    backend = _backend(monkeypatch, calls, idle_ttl=60.0)
    await backend.ensure_oai_lora_loaded(OAI_NAME, adapter_dir)

    backend._last_used[OAI_NAME] -= 61
    await backend._sweep_idle_adapters()
    assert backend._oai_loaded == set()

    await backend.ensure_oai_lora_loaded(OAI_NAME, adapter_dir)

    # Unload before unstage: vLLM re-reads the staged path while the name lives.
    load = ("post", f"{OAI_URL}/v1/load_lora_adapter", {"lora_name": OAI_NAME, "lora_path": STAGED})
    assert [c for c in calls if c[0] != "stage"] == [
        load,
        ("post", f"{OAI_URL}/v1/unload_lora_adapter", {"lora_name": OAI_NAME}),
        ("unstage", OAI_NAME),
        load,
    ]
    assert backend._oai_loaded == {OAI_NAME}
