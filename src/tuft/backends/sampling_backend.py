"""Sampling backend implemented using vLLM"""

import asyncio
import json
import os
import socket
import threading
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Optional

from opentelemetry.trace import StatusCode
from tinker import types

from ..checkpoints import read_adapter_files
from ..compat import sampled_sequence
from ..config import ModelConfig
from ..telemetry.tracing import get_tracer
from .base_backend import BaseSamplingBackend
from .vllm_engine import VLLMEngine, VLLMEngineConfig


_get_tracer = lambda: get_tracer("tuft.sampling_backend")  # noqa: E731


logger = getLogger(__name__)


def _build_sample_response(
    req_output: Any,
    include_prompt_logprobs: bool = False,
    topk_prompt_logprobs: int = 0,
) -> types.SampleResponse:
    """Build a tinker 0.18.2 SampleResponse from a raw vLLM RequestOutput.

    The engine actor (``vllm_engine.VLLMEngine.generate``) returns vLLM's
    ``RequestOutput`` untouched; this function is the single adapter between
    vLLM's output format and tinker's frozen dataclasses (constructed via
    their list-based ``_tokens_list=`` / ``_logprobs_list=`` fields in
    tinker 0.18.2).
    """
    sequences: list[types.SampledSequence] = []
    topk_prompt_logprobs_list: list[list[tuple[int, float]] | None] = [None]
    prompt_logprobs: list[float | None] = [None]

    # collect prompt logprobs
    if include_prompt_logprobs:
        for logprob_dict in req_output.prompt_logprobs[1:]:
            prompt_logprobs.append(next(iter(logprob_dict.values())).logprob)
            if topk_prompt_logprobs > 0:
                logprob_items = sorted(logprob_dict.items(), key=lambda x: x[1].rank)
                topk = logprob_items[:topk_prompt_logprobs]
                topk_prompt_logprobs_list.append(
                    [(token_id, logprob.logprob) for token_id, logprob in topk]
                )

    # collect response sequences
    for seq_output in req_output.outputs:
        seq = sampled_sequence(
            stop_reason="length" if seq_output.finish_reason == "length" else "stop",
            _tokens_list=seq_output.token_ids,
            _logprobs_list=[
                next(iter(logprob_dict.values())).logprob for logprob_dict in seq_output.logprobs
            ],
        )
        sequences.append(seq)

    return types.SampleResponse(
        sequences=sequences,
        _prompt_logprobs_list=prompt_logprobs if include_prompt_logprobs else None,
        _topk_prompt_logprobs_list=(
            topk_prompt_logprobs_list
            if include_prompt_logprobs and topk_prompt_logprobs > 0
            else None
        ),
    )


def _response_message(resp: Any) -> str:
    """Best-effort message out of a vLLM OpenAI error body."""
    try:
        body = resp.json()
    except ValueError:
        return resp.text
    return body.get("message", "") if isinstance(body, dict) else str(body)


class VLLMSamplingBackend(BaseSamplingBackend):
    """A sampling backend using vLLM.

    User side `sample`, `sample_async`, `compute_logprobs` and
    `compute_logprobs_async` are all supported by the sample method.
    """

    def __init__(
        self,
        config: ModelConfig,
        worker_venv_path: Optional[str] = None,
        *,
        instance_index: int = 0,
    ) -> None:
        from vllm.lora.request import LoRARequest

        super().__init__(config)
        self._worker_venv_path = worker_venv_path
        self._instance_index = instance_index
        self.engine = self._create_engine(config)
        self.lora_adapters: dict[str, LoRARequest] = {}
        # Adapter names this instance registered with its own vLLM OpenAI server.
        # Held here, not in the oai router, so removal can unload them before the
        # staged files they point at are deleted. Fast path only -- a lost
        # response can leave it disagreeing with the server, so correctness rests
        # on _confirm_oai_unloaded, which always asks.
        self._oai_loaded: set[str] = set()
        # Source adapter directory per id, kept after an idle unload so the next
        # request can re-stage it. A Path costs nothing next to the staged copy
        # it lets us drop.
        self._adapter_paths: dict[str, Path] = {}
        # Request-start time per adapter id; the idle sweep reads it.
        self._last_used: dict[str, float] = {}
        self._idle_ttl = config.adapter_idle_ttl_minutes * 60
        self._sweep_task: Optional[asyncio.Task] = None
        self._counter = 1
        self._lock = asyncio.Lock()
        self._openai_api_url: Optional[str] = None

    def _create_engine(self, config: ModelConfig):
        if config.colocate:
            return self._create_colocated_engine(config)
        else:
            return self._create_standalone_engine(config)

    def _build_engine_config(
        self,
        config: ModelConfig,
        *,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        bundle_indices: str = "",
    ) -> VLLMEngineConfig:
        return VLLMEngineConfig(
            model_path=str(config.model_path),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=(
                config.sampling_max_model_len
                if config.sampling_max_model_len is not None
                else config.max_model_len
            ),
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=config.sampling_enforce_eager,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            logprobs=config.logprobs,
            min_response_tokens=config.min_response_tokens,
            repetition_penalty=1.0,
            enable_lora=True,
            max_lora_rank=config.max_lora_rank,
            max_loras=config.max_loras,
            quantization=config.quantization,
            enable_runtime_lora_updating=True,
            enable_openai_api=True,
            enable_auto_tool_choice=config.enable_auto_tool_choice,
            tool_call_parser=config.tool_call_parser,
            reasoning_parser=config.reasoning_parser,
            bundle_indices=bundle_indices,
        )

    def _create_colocated_engine(self, config: ModelConfig):
        import ray

        if not self._worker_venv_path or not self._worker_venv_path.strip():
            _runtime_env = {}
        else:
            _path = os.environ.get("PATH", "")
            _venv_python = str(Path(self._worker_venv_path) / "bin" / "python")
            _runtime_env = {
                "py_executable": _venv_python,
                "env_vars": {
                    "VIRTUAL_ENV": self._worker_venv_path,
                    "PATH": f"{self._worker_venv_path}/bin:{_path}",
                },
            }
        return (
            ray.remote(VLLMEngine)
            .options(
                name="sampling_model_" + self.base_model,
                num_gpus=config.sampling_memory_fraction,
                resources=config.actor_resources("sampling", config.sampling_memory_fraction),
                runtime_env=_runtime_env,
            )
            .remote(
                config=self._build_engine_config(
                    config,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=config.sampling_memory_fraction,
                )
            )
        )

    def _create_standalone_engine(self, config: ModelConfig):
        import ray

        # Assign tensor_parallel_size GPUs to the actor itself
        # so that Ray populates CUDA_VISIBLE_DEVICES correctly.  vLLM then
        # creates its own placement group inside the EngineCore process where
        # the GPUs are visible.
        num_gpus = config.tensor_parallel_size
        bundle_indices = ",".join(str(i) for i in range(config.tensor_parallel_size))

        if not self._worker_venv_path or not self._worker_venv_path.strip():
            _runtime_env = {}
        else:
            _path = os.environ.get("PATH", "")
            _venv_python = str(Path(self._worker_venv_path) / "bin" / "python")
            _runtime_env = {
                "py_executable": _venv_python,
                "env_vars": {
                    "VIRTUAL_ENV": self._worker_venv_path,
                    "PATH": f"{self._worker_venv_path}/bin:{_path}",
                },
            }

        # Use instance_index to differentiate Ray actor names for DP replicas
        actor_name = f"sampling_model_{self.base_model}"
        if self._instance_index > 0:
            actor_name = f"{actor_name}_dp{self._instance_index}"

        # In standalone/DP mode, each vLLM instance has a dedicated GPU.
        # Use 0.9 (vLLM default) for max KV cache, not sampling_memory_fraction
        # which is designed for colocate mode (shared GPU with training).
        standalone_gpu_memory_utilization = 0.9

        return (
            ray.remote(VLLMEngine)
            .options(
                name=actor_name,
                num_gpus=num_gpus,
                resources=config.actor_resources("sampling", num_gpus),
                runtime_env=_runtime_env,
            )
            .remote(
                config=self._build_engine_config(
                    config,
                    tensor_parallel_size=config.tensor_parallel_size,
                    gpu_memory_utilization=standalone_gpu_memory_utilization,
                    bundle_indices=bundle_indices,
                )
            )
        )

    async def async_init(self) -> None:
        """Initialize the backend for sampling."""
        # Ray @ray.remote decorator adds .remote() method dynamically
        await self.engine.prepare.remote()  # type: ignore[attr-defined]
        self._openai_api_url = await self.engine.get_api_server_url.remote()  # type: ignore[attr-defined]
        logger.info(
            f"SamplingBackend for model {self.base_model} initialized. "
            f"OpenAI API URL: {self._openai_api_url}"
        )
        # Wait for the OpenAI API server to be ready to accept connections
        if self._openai_api_url:
            await self._wait_for_openai_server(self._openai_api_url)
        if self._idle_ttl > 0 and self._sweep_task is None:
            self._sweep_task = asyncio.create_task(self._sweep_loop())

    async def _wait_for_openai_server(self, url: str, timeout: float = 120.0) -> None:
        """Poll the vLLM OpenAI server until it accepts connections."""
        import asyncio

        import httpx as _httpx

        deadline = asyncio.get_event_loop().time() + timeout
        check_url = f"{url}/v1/models"
        attempt = 0
        async with _httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                attempt += 1
                try:
                    resp = await client.get(check_url, timeout=3.0)
                    if resp.status_code == 200:
                        logger.info(f"vLLM OpenAI server at {url} is ready (attempt {attempt})")
                        return
                except (_httpx.ConnectError, _httpx.TimeoutException, OSError):
                    pass
                if attempt % 10 == 0:
                    logger.info(f"Waiting for vLLM OpenAI server at {url}... attempt {attempt}")
                await asyncio.sleep(2.0)
        logger.warning(f"vLLM OpenAI server at {url} not ready after {timeout}s")

    def get_openai_api_url(self) -> Optional[str]:
        """Return the vLLM OpenAI API base URL."""
        return self._openai_api_url

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Sampling using vLLM engine."""
        with _get_tracer().start_as_current_span("sampling_backend.sample") as span:
            span.set_attribute("tuft.num_samples", num_samples)
            span.set_attribute("tuft.has_lora", lora_id is not None)
            try:
                lora_request = None
                # One lock hold for the whole resolution: a sweep waiting on the
                # lock must not be able to unload the adapter between the check,
                # the re-add and the read.
                async with self._lock:
                    if lora_id is not None:
                        if lora_id not in self.lora_adapters:
                            adapter_path = self._adapter_paths.get(lora_id)
                            if adapter_path is None:
                                raise ValueError(f"LoRA adapter {lora_id} not found in backend.")
                            # The idle sweep unloaded it; the caller pays the
                            # re-staging latency instead of seeing an error.
                            logger.info("Re-registering idle-unloaded LoRA adapter %s", lora_id)
                            await self._add_adapter_locked(lora_id, adapter_path)
                        # ponytail: stamped at request start, so a generation
                        # still running when the TTL expires could have its
                        # adapter unloaded under it. At the 30-minute default
                        # that cannot happen; track in-flight requests if the
                        # TTL is ever configured near a generation's length.
                        self._last_used[lora_id] = time.monotonic()
                        lora_request = self.lora_adapters[lora_id]

                prompt_token_ids = prompt.to_ints()
                params = {
                    "max_tokens": (
                        sampling_params.max_tokens if sampling_params.max_tokens is not None else 16
                    ),
                    "seed": sampling_params.seed,
                    "top_k": sampling_params.top_k,
                    "top_p": sampling_params.top_p,
                    "temperature": sampling_params.temperature,
                    "n": num_samples,
                    "prompt_logprobs": (topk_prompt_logprobs if include_prompt_logprobs else None),
                    "logprobs": 0,
                }
                # Avoid prefix cache reads when computing prompt logprobs
                # (cached prompt chunks would otherwise skip logit computation
                # and corrupt the returned logprobs). Native vLLM SamplingParams
                # field since 0.12; newer vLLM auto-sets it when prompt_logprobs
                # is requested -- kept explicit here for clarity.
                if include_prompt_logprobs:
                    params["skip_reading_prefix_cache"] = True
                if sampling_params.stop is not None:
                    # tinker.SamplingParams.stop accepts a heterogeneous list of
                    # str (substring stops) and int (token-id stops). vLLM
                    # however expects them in two separate fields: `stop`
                    # (list[str]) and `stop_token_ids` (list[int]). Forwarding
                    # the raw list to vLLM's `stop` triggers `TypeError: object
                    # of type 'int' has no len()` inside
                    # SamplingParams.__post_init__ (vllm/sampling_params.py:358).
                    # Split by isinstance to route each kind correctly.
                    str_stops: list[str] = [s for s in sampling_params.stop if isinstance(s, str)]
                    int_stops: list[int] = [
                        s
                        for s in sampling_params.stop
                        if isinstance(s, int) and not isinstance(s, bool)
                    ]
                    if str_stops:
                        params["stop"] = str_stops
                    if int_stops:
                        params["stop_token_ids"] = int_stops

                # Ray @ray.remote decorator adds .remote() method dynamically
                req_output = await self.engine.generate.remote(  # type: ignore[attr-defined]
                    prompt={"prompt_token_ids": prompt_token_ids},
                    lora_request=lora_request,
                    **params,
                )
                return _build_sample_response(
                    req_output=req_output,
                    include_prompt_logprobs=include_prompt_logprobs,
                    topk_prompt_logprobs=topk_prompt_logprobs,
                )
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    async def stage_adapter(self, lora_id: str, adapter_path: Path) -> str:
        """Copy the adapter onto the engine actor's node; return the local path.

        The bytes travel as a plain Ray argument, which Ray promotes into its
        object store, so the server node and the vLLM node share no filesystem.
        """
        files = read_adapter_files(adapter_path)
        return await self.engine.stage_adapter.remote(lora_id, files)  # type: ignore[attr-defined]

    async def ensure_oai_lora_loaded(self, lora_name: str, adapter_path: Path) -> None:
        """Stage the adapter and register it with this instance's OpenAI server.

        vLLM's serving layer keeps its own name to adapter mapping, separate from
        the engine registry ``add_adapter`` fills, and reads the adapter from the
        path given here for as long as the name stays registered.
        """
        import httpx

        # ponytail: an adapter served over OAI stays loaded and staged until
        # remove_adapter runs for its name or the engine shuts down. Tinker
        # sessions evict under their own random session id, so in practice only
        # shutdown frees these; give the OAI names an eviction hook of their own
        # if shm pressure shows up.
        if self._openai_api_url is None:
            return
        async with self._lock:
            # A cache hit is a use: stamp before the early return, or the sweep
            # would unload an adapter that OpenAI-API traffic keeps busy.
            self._last_used[lora_name] = time.monotonic()
            if lora_name in self._oai_loaded:
                return
            lora_path = await self.stage_adapter(lora_name, adapter_path)
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._openai_api_url}/v1/load_lora_adapter",
                    json={"lora_name": lora_name, "lora_path": lora_path},
                    headers={"Authorization": "Bearer EMPTY"},
                    timeout=60.0,
                )
            # A 400 "already loaded" is a success: the adapter name is immutable,
            # so whatever is registered under it is the same weights.
            if resp.status_code != 200 and "already" not in _response_message(resp).lower():
                # Deliberately leave the staged files: a timed-out load may have
                # registered the name anyway, and the staged path is
                # deterministic, so a retry rewrites exactly this directory.
                raise RuntimeError(
                    f"Failed to load LoRA adapter '{lora_name}': {resp.status_code} {resp.text}"
                )
            self._oai_loaded.add(lora_name)
            logger.info(
                "Loaded LoRA '%s' via vLLM OpenAI API at %s", lora_name, self._openai_api_url
            )

    async def _confirm_oai_unloaded(self, lora_name: str) -> bool:
        """Return True once the OpenAI serving layer is known not to hold the name.

        Always asks the server, whatever ``_oai_loaded`` says: a load whose
        response was lost may still have registered the name. Any answer is a
        confirmation -- 2xx unloaded it, 4xx means the server has no such name --
        while a transport error, a timeout or a 5xx leaves it unknown, and
        unknown must be treated as still registered.
        """
        import httpx

        if self._openai_api_url is None:
            return True
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._openai_api_url}/v1/unload_lora_adapter",
                    json={"lora_name": lora_name},
                    headers={"Authorization": "Bearer EMPTY"},
                    timeout=30.0,
                )
        except Exception:
            logger.warning(
                "Could not reach %s to unload LoRA '%s'; leaving it staged.",
                self._openai_api_url,
                lora_name,
            )
            return False
        if resp.status_code >= 500:
            logger.warning(
                "vLLM at %s failed to unload LoRA '%s' (%s); leaving it staged.",
                self._openai_api_url,
                lora_name,
                resp.status_code,
            )
            return False
        self._oai_loaded.discard(lora_name)
        return True

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        with _get_tracer().start_as_current_span("sampling_backend.add_adapter") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            try:
                async with self._lock:
                    await self._add_adapter_locked(lora_id, adapter_path)
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR)
                raise

    async def _add_adapter_locked(self, lora_id: str, adapter_path: Path) -> None:
        """Stage and register the adapter. The caller must hold ``_lock``.

        ``sample`` re-adds a swept adapter from inside its own lock hold, and
        ``asyncio.Lock`` is not reentrant.
        """
        from vllm.lora.request import LoRARequest

        if not adapter_path.exists():
            raise ValueError(f"LoRA adapter path {adapter_path} does not exist.")
        staged_path = await self.stage_adapter(lora_id, adapter_path)
        self._counter += 1
        request = LoRARequest(
            lora_int_id=self._counter + 1,
            lora_name=lora_id,
            lora_path=staged_path,
        )
        # Register with vLLM first; only record the adapter in the local
        # registry after success. Otherwise a failure (missing path, or the
        # engine raising) would leave a stale entry that sample() would later
        # pass to vLLM even though it was never registered.
        try:
            added = await self.engine.add_lora.remote(request)  # type: ignore[attr-defined]
            if added is False:
                raise RuntimeError(
                    f"vLLM did not register LoRA adapter {lora_id} from {adapter_path}."
                )
        except Exception:
            # Registration failed, so remove_adapter will never run for this id;
            # drop the staged copy instead of leaking tmpfs -- unless the OpenAI
            # serving layer registered the same name against the same directory
            # and may still read it.
            if lora_id not in self._oai_loaded:
                await self.engine.unstage_adapter.remote(lora_id)  # type: ignore[attr-defined]
            raise
        self.lora_adapters[lora_id] = request
        self._adapter_paths[lora_id] = adapter_path
        self._last_used[lora_id] = time.monotonic()

    async def _sweep_loop(self) -> None:
        """Unload adapters that have served nothing for ``adapter_idle_ttl``."""
        while True:
            await asyncio.sleep(max(1.0, self._idle_ttl / 10))
            try:
                await self._sweep_idle_adapters()
            except Exception:
                logger.exception("Idle adapter sweep failed")

    async def _sweep_idle_adapters(self) -> None:
        cutoff = time.monotonic() - self._idle_ttl
        for lora_id in list(self._last_used):
            async with self._lock:
                # Re-read the stamp under the lock. Every stamp is taken under it
                # too, so an adapter used since this sweep began cannot be
                # unloaded here; a missing entry counts as fresh.
                if self._last_used.get(lora_id, cutoff) >= cutoff:
                    continue
                logger.info("Unloading LoRA adapter %s, idle for %.0fs", lora_id, self._idle_ttl)
                await self._remove_adapter_locked(lora_id)

    async def remove_adapter(self, lora_id: str) -> None:
        with _get_tracer().start_as_current_span("sampling_backend.remove_adapter") as span:
            span.set_attribute("tuft.lora_id", lora_id)
            async with self._lock:
                await self._remove_adapter_locked(lora_id)

    async def _remove_adapter_locked(self, lora_id: str) -> None:
        """Unregister, unload and unstage the adapter. Caller holds ``_lock``."""
        if lora_id in self.lora_adapters:
            # vLLM removes adapters by their integer id, not name.
            lora_request = self.lora_adapters.pop(lora_id)
            await self.engine.remove_lora.remote(lora_request.lora_int_id)  # type: ignore[attr-defined]
        # Unstage only against a confirmed unload: while the name is registered
        # anywhere, vLLM re-reads the staged path after a worker-side LRU
        # eviction. An unconfirmed unload leaks one staged directory until the
        # next eviction or engine shutdown, which is the safe direction.
        if await self._confirm_oai_unloaded(lora_id):
            await self.engine.unstage_adapter.remote(lora_id)  # type: ignore[attr-defined]
            # Keeping it on an unconfirmed unload leaves the next sweep to retry
            # the whole removal.
            self._last_used.pop(lora_id, None)

    async def shutdown(self) -> None:
        """Shut down the vLLM engine Ray actor and release GPU resources."""
        import ray

        if self._sweep_task is not None:
            self._sweep_task.cancel()
            self._sweep_task = None
        try:
            await self.engine.shutdown.remote()  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            ray.kill(self.engine, no_restart=True)  # type: ignore[arg-type]
        except Exception:
            pass
        self.lora_adapters.clear()


class DPSamplingBackend(BaseSamplingBackend):
    """Data-Parallel sampling backend: N independent vLLM instances with round-robin LB.

    Each instance runs on its own GPU(s) (tensor_parallel_size per instance).
    Requests are distributed across instances using round-robin for uniform load.
    All instances share the same LoRA adapters.
    """

    def __init__(
        self,
        config: ModelConfig,
        worker_venv_path: Optional[str] = None,
    ) -> None:
        super().__init__(config)
        self._dp_size = config.data_parallel_size
        self._instances: list[VLLMSamplingBackend] = []
        for i in range(self._dp_size):
            instance = VLLMSamplingBackend(
                config, worker_venv_path=worker_venv_path, instance_index=i
            )
            self._instances.append(instance)
        # Atomic round-robin counter
        self._rr_counter = 0
        self._rr_lock = asyncio.Lock()
        logger.info(
            "DPSamplingBackend: created %d instances for model %s",
            self._dp_size,
            config.model_name,
        )

    def _next_instance(self) -> VLLMSamplingBackend:
        """Round-robin selection (no lock needed for simple modular arithmetic)."""
        idx = self._rr_counter % self._dp_size
        self._rr_counter += 1
        return self._instances[idx]

    async def async_init(self) -> None:
        """Initialize all DP instances in parallel."""
        await asyncio.gather(*[inst.async_init() for inst in self._instances])
        logger.info(
            "DPSamplingBackend: all %d instances initialized for model %s",
            self._dp_size,
            self.base_model,
        )

    def get_openai_api_url(self) -> Optional[str]:
        """Return a single URL (first instance) for backward compat.

        For DP-aware routing, use get_openai_api_urls() instead.
        """
        return self._instances[0].get_openai_api_url() if self._instances else None

    def get_openai_api_urls(self) -> list[str]:
        """Return all vLLM instance URLs for DP-aware load balancing."""
        urls = []
        for inst in self._instances:
            url = inst.get_openai_api_url()
            if url:
                urls.append(url)
        return urls

    def get_next_openai_api_url(self) -> Optional[str]:
        """Return the next URL via round-robin for load-balanced proxying."""
        inst = self._next_instance()
        return inst.get_openai_api_url()

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Route sampling to a DP instance via round-robin."""
        instance = self._next_instance()
        return await instance.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=sampling_params,
            include_prompt_logprobs=include_prompt_logprobs,
            topk_prompt_logprobs=topk_prompt_logprobs,
            lora_id=lora_id,
        )

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        """Add LoRA adapter to ALL DP instances; roll back on partial failure.

        Without rollback, a failure on one replica would leave the DP instances with
        inconsistent adapter sets (some registered, some not), causing round-robin
        sampling to intermittently fail with an unknown lora_id.
        """
        results = await asyncio.gather(
            *[inst.add_adapter(lora_id, adapter_path) for inst in self._instances],
            return_exceptions=True,
        )
        failures = [i for i, res in enumerate(results) if isinstance(res, BaseException)]
        if failures:
            # Roll back the instances that succeeded so the adapter set stays consistent.
            for i, res in enumerate(results):
                if not isinstance(res, BaseException):
                    try:
                        await self._instances[i].remove_adapter(lora_id)
                    except Exception:  # pylint: disable=broad-except
                        logger.warning("Failed to roll back adapter %s on instance %d", lora_id, i)
            first_error = results[failures[0]]
            if isinstance(first_error, BaseException):
                raise first_error

    async def ensure_oai_lora_loaded(self, lora_name: str, adapter_path: Path) -> None:
        """Load the adapter on every DP instance; each stages its own copy."""
        await asyncio.gather(
            *[inst.ensure_oai_lora_loaded(lora_name, adapter_path) for inst in self._instances]
        )

    async def remove_adapter(self, lora_id: str) -> None:
        """Remove LoRA adapter from ALL DP instances (unloading and unstaging it)."""
        await asyncio.gather(*[inst.remove_adapter(lora_id) for inst in self._instances])

    async def shutdown(self) -> None:
        """Shut down all DP vLLM instances."""
        for inst in self._instances:
            try:
                await inst.shutdown()
            except Exception:
                pass


def _build_mock_openai_app(model_name: str):
    """Build a minimal FastAPI app that mimics vLLM's OpenAI-compatible endpoints."""
    import itertools

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    mock_app = FastAPI()
    _id_counter = itertools.count(1)

    def _make_completion_response(body: dict) -> dict:
        model = body.get("model", model_name)
        raw_prompt = body.get("prompt", "")
        max_tokens = body.get("max_tokens", 16)
        n = body.get("n", 1)
        logprobs_n = body.get("logprobs")
        stop = body.get("stop")

        # Handle batch prompts (list of strings)
        prompts = raw_prompt if isinstance(raw_prompt, list) else [raw_prompt]
        choices = []
        idx = 0
        for prompt in prompts:
            prompt_str = str(prompt)[:20]
            for _j in range(n):
                text = f" dummy completion output for '{prompt_str}'"
                finish = "length"
                # Respect stop sequences
                if stop:
                    for s in stop if isinstance(stop, list) else [stop]:
                        pos = text.find(s)
                        if pos != -1:
                            text = text[:pos]
                            finish = "stop"
                            break
                logprobs_data = None
                if logprobs_n is not None:
                    logprobs_data = {
                        "tokens": [" dummy"],
                        "token_logprobs": [-0.5],
                        "top_logprobs": [{"dummy": -0.5}] if logprobs_n > 0 else None,
                        "text_offset": [0],
                    }
                choices.append(
                    {
                        "index": idx,
                        "text": text,
                        "logprobs": logprobs_data,
                        "finish_reason": finish,
                    }
                )
                idx += 1
        prompt_tokens = sum(max(1, len(str(p).split())) for p in prompts)
        return {
            "id": f"cmpl-dummy-{next(_id_counter)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max_tokens * len(choices),
                "total_tokens": prompt_tokens + max_tokens * len(choices),
            },
        }

    def _make_chat_response(body: dict) -> dict:
        model = body.get("model", model_name)
        n = body.get("n", 1)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or 16
        choices = []
        for i in range(n):
            choices.append(
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": "This is a dummy response from the mock OpenAI server.",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            )
        return {
            "id": f"chatcmpl-dummy-{next(_id_counter)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": max_tokens,
                "total_tokens": 10 + max_tokens,
            },
        }

    async def _stream_chat_chunks(body: dict):
        model = body.get("model", model_name)
        chunk_id = f"chatcmpl-dummy-{next(_id_counter)}"
        # First chunk with role
        first = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
            ],
        }
        yield f"data: {json.dumps(first)}\n\n"
        # Content chunks
        words = ["This", " is", " a", " dummy", " streamed", " response."]
        for word in words:
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        # Final chunk
        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": len(words),
                "total_tokens": 10 + len(words),
            },
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    async def _stream_completion_chunks(body: dict):
        model = body.get("model", model_name)
        chunk_id = f"cmpl-dummy-{next(_id_counter)}"
        words = [" dummy", " completion", " output"]
        for word in words:
            chunk = {
                "id": chunk_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "text": word, "logprobs": None, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        final = {
            "id": chunk_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": "", "logprobs": None, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": len(words),
                "total_tokens": 5 + len(words),
            },
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    @mock_app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        if body.get("stream"):
            return StreamingResponse(
                _stream_completion_chunks(body), media_type="text/event-stream"
            )
        return JSONResponse(_make_completion_response(body))

    @mock_app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        if body.get("stream"):
            return StreamingResponse(_stream_chat_chunks(body), media_type="text/event-stream")
        return JSONResponse(_make_chat_response(body))

    @mock_app.get("/v1/models")
    async def list_models():
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "dummy",
                    }
                ],
            }
        )

    return mock_app


class DummySamplingBackend(BaseSamplingBackend):
    """A dummy sampling backend that returns fixed responses for unittest."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.lora_adapters: dict[str, Path] = {}
        self._counter = 1
        self._lock = asyncio.Lock()
        self._openai_api_url: Optional[str] = None
        self._mock_server: object | None = None
        self._mock_thread: threading.Thread | None = None

    async def async_init(self) -> None:
        """Start an embedded mock OpenAI server for testing."""
        self._start_mock_openai_server()

    def _start_mock_openai_server(self) -> None:
        """Start a lightweight mock OpenAI-compatible HTTP server in a background thread."""
        import uvicorn

        mock_app = _build_mock_openai_app(str(self.config.model_path))

        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        server = uvicorn.Server(
            uvicorn.Config(mock_app, host="127.0.0.1", port=port, log_level="error")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait for the server to become ready
        import httpx

        for _ in range(50):
            try:
                resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=0.5)
                if resp.status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)

        self._openai_api_url = f"http://127.0.0.1:{port}"
        self._mock_server = server
        self._mock_thread = thread
        logger.info(f"DummySamplingBackend mock OpenAI server started at {self._openai_api_url}")

    def get_openai_api_url(self) -> Optional[str]:
        """Return the mock OpenAI API base URL."""
        return self._openai_api_url

    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
    ) -> types.SampleResponse:
        """Return a fixed dummy response."""
        prompt_tokens = prompt.to_ints()
        max_tokens = sampling_params.max_tokens or 16
        sequences: list[types.SampledSequence] = []
        for _ in range(num_samples):
            generated = self._generate_tokens(prompt_tokens, max_tokens)
            seq = sampled_sequence(
                stop_reason="length",
                _tokens_list=generated,
                _logprobs_list=[-0.3 for _ in generated],
            )
            sequences.append(seq)
        prompt_logprobs = None
        topk_prompt = None
        if include_prompt_logprobs:
            prompt_logprobs = [-0.1 if tok is not None else None for tok in prompt_tokens]
        if topk_prompt_logprobs > 0:
            topk_prompt = [
                [
                    (token, round(-0.05 - idx * 0.01, 4))
                    for idx, token in enumerate(prompt_tokens[:topk_prompt_logprobs])
                ]
                if token is not None
                else None
                for token in prompt_tokens
            ]
        return types.SampleResponse(
            sequences=sequences,
            _prompt_logprobs_list=prompt_logprobs,
            _topk_prompt_logprobs_list=topk_prompt,
        )

    def _generate_tokens(self, prompt_tokens: list[int], max_tokens: int) -> list[int]:
        start = prompt_tokens[-1] if prompt_tokens else (abs(self.config.seed) % 32000) + 1
        return [(start + i) % 32000 for i in range(1, max_tokens + 1)]

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        self.lora_adapters[lora_id] = adapter_path

    async def remove_adapter(self, lora_id: str) -> None:
        if lora_id in self.lora_adapters:
            del self.lora_adapters[lora_id]

    async def shutdown(self) -> None:
        """Stop the mock OpenAI server if running."""
        if self._mock_server is not None:
            try:
                self._mock_server.should_exit = True  # type: ignore[attr-defined]
            except Exception:
                pass
        self._mock_server = None
        self._mock_thread = None
