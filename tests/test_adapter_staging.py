"""LoRA adapter staging: adapter bytes -> node-local dir -> path handed to vLLM."""

from __future__ import annotations

import asyncio
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from tinker import types

from tuft.backends.sampling_backend import VLLMSamplingBackend, _build_sample_response
from tuft.backends.vllm_engine import VLLMEngine, _plain_output
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
    max_staged: int = 8,
) -> VLLMSamplingBackend:
    """Stub engine and OpenAI server; calls land in ``calls``.

    ``statuses`` maps an endpoint to its status; 0 is a transport failure.
    """
    import httpx

    statuses = statuses or {}

    # The server side must run without vLLM (the tuft-train image has none).
    monkeypatch.setitem(sys.modules, "vllm", None)

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
    backend._in_flight = Counter()
    backend._max_staged = max_staged
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


async def test_lru_cap_unloads_the_oldest_idle_adapter(tmp_path, monkeypatch):
    adapter_dir = _adapter_dir(tmp_path)
    calls: list[tuple] = []
    backend = _backend(monkeypatch, calls, max_staged=2)
    for session in ("old", "busy", "new"):
        await backend.add_adapter(session, adapter_dir)
        backend._last_used[session] -= 100 if session == "old" else 50
    # "old" was evicted when "new" arrived; a request keeps "busy" pinned.
    assert set(backend.lora_adapters) == {"busy", "new"}
    assert ("unstage", "old") in calls

    backend._in_flight["busy"] += 1
    await backend.add_adapter("newest", adapter_dir)
    assert set(backend.lora_adapters) == {"busy", "newest"}
    # The evicted session keeps its source path, so its next request re-adds it.
    assert backend._adapter_paths["old"] == adapter_dir


def test_engine_output_unpickles_without_vllm(monkeypatch):
    # Stand-ins for vLLM's output classes, importable only while "fake_vllm" is.
    fake = ModuleType("fake_vllm")
    for name in ("Logprob", "CompletionOutput", "RequestOutput"):
        cls = type(name, (SimpleNamespace,), {"__module__": "fake_vllm"})
        setattr(fake, name, cls)
    monkeypatch.setitem(sys.modules, "fake_vllm", fake)
    raw = fake.RequestOutput(
        prompt_logprobs=[None, {5: fake.Logprob(logprob=-0.5, rank=1)}],
        outputs=[
            fake.CompletionOutput(
                token_ids=(7,),
                finish_reason="stop",
                logprobs=[{7: fake.Logprob(logprob=-0.1, rank=1)}],
            )
        ],
    )
    raw_bytes, plain_bytes = pickle.dumps(raw), pickle.dumps(_plain_output(raw))

    monkeypatch.delitem(sys.modules, "fake_vllm")  # the server side: no vLLM
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(raw_bytes)
    response = _build_sample_response(pickle.loads(plain_bytes), include_prompt_logprobs=True)
    assert response.sequences[0].tokens == [7]
    assert response.prompt_logprobs == [None, -0.5]


@pytest.mark.skipif(
    "topk_sample_logprobs" not in types.SampleRequest.model_fields,
    reason="topk_sample_logprobs only exists from tinker 0.29",
)
def test_topk_sample_logprobs_are_ordered_by_rank():
    lp = lambda logprob, rank: SimpleNamespace(logprob=logprob, rank=rank)  # noqa: E731
    # vLLM lists the sampled token first, here with rank 3 (outside the top 2).
    position = {7: lp(-3.0, 3), 4: lp(-2.0, 2), 9: lp(-1.0, 1)}
    output = SimpleNamespace(
        prompt_logprobs=None,
        outputs=[SimpleNamespace(token_ids=[7], finish_reason="length", logprobs=[position])],
    )
    seq = _build_sample_response(output, topk_sample_logprobs=2).sequences[0]
    assert seq.logprobs == [-3.0]
    assert seq.topk_logprobs == [[(9, -1.0), (4, -2.0)]]
