"""LoRA adapter staging: adapter bytes -> node-local dir -> path handed to vLLM."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

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


async def test_stage_adapter_fails_loudly_without_config(tmp_path):
    # vLLM reads a lora_path it cannot find as a Hugging Face repo id and
    # downloads it, so staging must never return a path it did not write to.
    with pytest.raises(RuntimeError, match="adapter_config.json"):
        await _engine(tmp_path).stage_adapter(LORA_ID, {"adapter_model.safetensors": b"x"})


async def test_add_adapter_registers_the_staged_path(tmp_path, monkeypatch):
    adapter_dir = _adapter_dir(tmp_path)
    recorded: dict = {}

    class _LoRARequest:
        def __init__(self, lora_int_id: int, lora_name: str, lora_path: str) -> None:
            self.lora_path = lora_path

    vllm = ModuleType("vllm")
    vllm_lora = ModuleType("vllm.lora")
    vllm_lora_request = ModuleType("vllm.lora.request")
    vllm_lora_request.LoRARequest = _LoRARequest  # type: ignore[attr-defined]
    vllm_lora.request = vllm_lora_request  # type: ignore[attr-defined]
    vllm.lora = vllm_lora  # type: ignore[attr-defined]
    for name, module in [
        ("vllm", vllm),
        ("vllm.lora", vllm_lora),
        ("vllm.lora.request", vllm_lora_request),
    ]:
        monkeypatch.setitem(sys.modules, name, module)

    def _stage(lora_id: str, files: dict[str, bytes]) -> str:
        recorded["files"] = files
        return "/dev/shm/tuft-adapters-0123/staged"

    def _add_lora(request) -> bool:
        recorded["request"] = request
        return True

    backend = VLLMSamplingBackend.__new__(VLLMSamplingBackend)
    backend.engine = SimpleNamespace(  # type: ignore[assignment]
        stage_adapter=_remote(_stage), add_lora=_remote(_add_lora)
    )
    backend.lora_adapters = {}
    backend._counter = 1
    backend._lock = asyncio.Lock()

    await backend.add_adapter(LORA_ID, adapter_dir)

    # The engine got the bytes, and vLLM got the engine's own node-local path.
    assert recorded["files"] == read_adapter_files(adapter_dir)
    assert recorded["request"].lora_path == "/dev/shm/tuft-adapters-0123/staged"


STAGED = "/dev/shm/tuft-adapters-0123/staged"
OAI_URL = "http://vllm-node:8000"


def _oai_backend(monkeypatch, calls: list, statuses: dict[str, int]) -> VLLMSamplingBackend:
    """A backend whose engine and vLLM OpenAI server are both recorders.

    ``statuses`` maps an endpoint name to the status it answers with; 0 stands
    for a transport failure.
    """
    import httpx

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
            return SimpleNamespace(
                status_code=status, json=lambda: {"message": "nope"}, text="nope"
            )

    monkeypatch.setattr(httpx, "AsyncClient", lambda *_a, **_kw: _FakeClient())

    backend = VLLMSamplingBackend.__new__(VLLMSamplingBackend)
    backend.engine = SimpleNamespace(  # type: ignore[assignment]
        stage_adapter=_remote(lambda lora_id, files: STAGED),
        unstage_adapter=_remote(lambda lora_id: calls.append(("unstage", lora_id))),
    )
    backend.lora_adapters = {}
    backend._oai_loaded = set()
    backend._lock = asyncio.Lock()
    backend._openai_api_url = OAI_URL
    return backend


async def test_oai_loaded_adapter_is_unloaded_before_unstaging(tmp_path, monkeypatch):
    calls: list[tuple] = []
    backend = _oai_backend(monkeypatch, calls, {})

    await backend.ensure_oai_lora_loaded(LORA_ID, _adapter_dir(tmp_path))
    assert backend._oai_loaded == {LORA_ID}

    await backend.remove_adapter(LORA_ID)

    # The serving layer drops the name BEFORE the staged files go: vLLM re-reads
    # that path whenever its worker-side LRU evicts the adapter.
    assert calls == [
        ("post", f"{OAI_URL}/v1/load_lora_adapter", {"lora_name": LORA_ID, "lora_path": STAGED}),
        ("post", f"{OAI_URL}/v1/unload_lora_adapter", {"lora_name": LORA_ID}),
        ("unstage", LORA_ID),
    ]
    assert backend._oai_loaded == set()


@pytest.mark.parametrize(
    "unload_status, confirmed",
    [
        (404, True),  # the server answered: it holds no such name
        (400, True),
        (503, False),  # it may still hold the name
        (0, False),  # transport failure: unknown
    ],
)
async def test_unstaging_waits_for_a_confirmed_unload(
    tmp_path, monkeypatch, unload_status, confirmed
):
    calls: list[tuple] = []
    backend = _oai_backend(monkeypatch, calls, {"unload_lora_adapter": unload_status})
    await backend.ensure_oai_lora_loaded(LORA_ID, _adapter_dir(tmp_path))

    await backend.remove_adapter(LORA_ID)

    assert (("unstage", LORA_ID) in calls) is confirmed
    # An unconfirmed unload keeps the name so a later removal retries it.
    assert (LORA_ID in backend._oai_loaded) is not confirmed


async def test_failed_oai_load_keeps_the_staged_copy(tmp_path, monkeypatch):
    # The load may have registered the name despite the error response, and the
    # staged path is deterministic, so a retry rewrites this same directory.
    calls: list[tuple] = []
    backend = _oai_backend(monkeypatch, calls, {"load_lora_adapter": 500})

    with pytest.raises(RuntimeError, match="Failed to load LoRA adapter"):
        await backend.ensure_oai_lora_loaded(LORA_ID, _adapter_dir(tmp_path))

    assert ("unstage", LORA_ID) not in calls
