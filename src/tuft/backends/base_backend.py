import asyncio
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from tinker import types

from ..checkpoints import CheckpointRecord
from ..config import TRAINING_CAPABILITY, ModelConfig


def gpu_request(num_gpus: float) -> float:
    """``num_gpus`` for a Ray actor: 0 when ``TUFT_NO_GPU=1`` (CPU-only runs)."""
    return 0 if os.environ.get("TUFT_NO_GPU") == "1" else num_gpus


def check_gpu_layout(models: list[ModelConfig]) -> None:
    """Warn when the models request more GPUs than the Ray cluster has now.

    Ray otherwise queues the actors silently until a startup timeout. Only a
    warning: autoscaling nodes may join later. Counts cluster totals, not
    per-node packing.
    """
    if os.environ.get("TUFT_NO_GPU") == "1" or os.environ.get("TUFT_CPU_TEST") == "1":
        return  # actors request no GPUs
    import ray

    need: dict[str, float] = {}
    roles: list[str] = []
    for m in models:
        train = m.fsdp_num_gpus if m.training_backend == "fsdp" else 1
        infer = m.tensor_parallel_size * m.data_parallel_size
        if m.colocate:
            train, infer = 1 - m.sampling_memory_fraction, m.sampling_memory_fraction
        for role in m.capabilities:
            n = train if role == TRAINING_CAPABILITY else infer
            if n:
                roles.append(f"{m.model_name} {role}: {n:g}")
                for res, k in {"GPU": n, **m.actor_resources(role, n)}.items():
                    need[res] = round(need.get(res, 0) + k, 6)
    if not ray.is_initialized():
        ray.init()
    have = ray.cluster_resources()
    if short := {r: n for r, n in need.items() if n > have.get(r, 0)}:
        logging.getLogger(__name__).warning(
            "GPU layout exceeds the Ray cluster: "
            + ", ".join(
                f"{r} requested {n:g}, available {have.get(r, 0):g}" for r, n in short.items()
            )
            + f" ({'; '.join(roles)})."
        )


class BaseBackend(ABC):
    """Base class for all backends."""

    def __init__(self, config: ModelConfig) -> None:
        self.base_model = config.model_name
        self.config = config

    @abstractmethod
    async def async_init(self) -> None:
        """Asynchronous initialization if needed."""

    async def ping(self) -> None:  # noqa: B027
        """Raise if the backend's actors are dead. Used by /api/v1/readyz."""

    async def shutdown(self) -> None:  # noqa: B027
        """Clean up resources (Ray actors, GPU memory, NCCL communicators, etc.).

        Called when the server is shutting down. Subclasses that own Ray actors
        or other resources should override this to ensure proper cleanup.
        """
        pass


class BaseSamplingBackend(BaseBackend):
    """Abstract sampling backend."""

    @abstractmethod
    async def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        lora_id: Optional[str] = None,
        topk_sample_logprobs: int = 0,
    ) -> types.SampleResponse:
        """Abstract method for sampling."""

    @abstractmethod
    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        """Add LoRA adapter to the backend."""

    @abstractmethod
    async def remove_adapter(self, lora_id: str) -> None:
        """Remove LoRA adapter from the backend."""

    async def ensure_oai_lora_loaded(self, lora_name: str, adapter_path: Path) -> None:  # noqa: B027
        """Load the adapter into the OpenAI serving layer; no-op without one."""

    def get_openai_api_url(self) -> Optional[str]:
        """Return the vLLM OpenAI API base URL, or None if not available."""
        return None

    @classmethod
    def create_backend(
        cls, config: ModelConfig, worker_venv_path: Optional[str] = None
    ) -> "BaseSamplingBackend":
        """Factory method to create a sampling backend instance.

        TUFT_CPU_TEST=1: use DummySamplingBackend (no vLLM, for CPU-only unit tests).
        data_parallel_size > 1: use DPSamplingBackend (multiple vLLM instances with LB).
        Otherwise: VLLMSamplingBackend (creates Ray/vLLM actor in __init__, may block startup).
        """
        if os.getenv("TUFT_CPU_TEST", "0") == "1":
            from ..backends.sampling_backend import DummySamplingBackend

            return DummySamplingBackend(config)
        if config.data_parallel_size > 1:
            from ..backends.sampling_backend import DPSamplingBackend

            return DPSamplingBackend(config, worker_venv_path=worker_venv_path)
        from ..backends.sampling_backend import VLLMSamplingBackend

        return VLLMSamplingBackend(config, worker_venv_path=worker_venv_path)


class BaseTrainingBackend(BaseBackend):
    """Abstract training backend."""

    config: ModelConfig
    _vocab_size: int | None = None  # set by _load_vocab_size() from async_init

    async def _load_vocab_size(self) -> None:
        """Read the vocabulary size from the checkpoint config, off the event loop."""
        from transformers import AutoConfig

        config = await asyncio.to_thread(
            AutoConfig.from_pretrained, self.config.model_path, trust_remote_code=True
        )
        self._vocab_size = config.get_text_config().vocab_size

    @abstractmethod
    async def forward(
        self,
        data: list[types.Datum],
        lora_id: str,
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
        backward: bool = False,
    ) -> types.ForwardBackwardOutput:
        """Abstract method for forward pass."""

    @abstractmethod
    async def create_adapter(self, lora_id: str, lora_config: types.LoraConfig) -> None:
        """Abstract method for creating LoRA adapter."""

    @abstractmethod
    async def remove_adapter(self, lora_id: str) -> None:
        """Abstract method for removing LoRA adapter."""

    @abstractmethod
    async def optim_step(
        self,
        adam_params: types.AdamParams,
        lora_id: str,
    ) -> types.OptimStepResponse:
        """Abstract method for optimization step."""

    @abstractmethod
    async def save_state(
        self, lora_id: str, checkpoint_record: "CheckpointRecord", optimizer: bool
    ) -> None:
        """Abstract method for saving model state."""

    @abstractmethod
    async def load_state(
        self, lora_id: str, checkpoint_record: "CheckpointRecord", optimizer: bool
    ) -> None:
        """Abstract method for loading model state."""

    @classmethod
    def create_backend(
        cls,
        config: ModelConfig,
        fsdp_index: Optional[int] = None,
        worker_venv_path: Optional[str] = None,
    ) -> "BaseTrainingBackend":
        """Factory method to create a training backend instance.

        fsdp_index: For FSDP backends, master port is config.fsdp_master_port + fsdp_index.
        The base port is configurable via ModelConfig.fsdp_master_port (default 29500); multiple
        FSDP models use base, base+1, ... Pass the index of this model among FSDP models in
        supported_models order. Omit for non-FSDP or when only one FSDP model (uses config port).
        """
        if os.getenv("TUFT_CPU_TEST", "0") == "1":
            from ..backends.training_backend import DummyTrainingBackend

            return DummyTrainingBackend(config)
        training_backend = getattr(config, "training_backend", "hf")
        if training_backend == "fsdp":
            from ..backends.fsdp_training_backend import FSDPTrainingBackend

            return FSDPTrainingBackend(
                config, fsdp_index=fsdp_index, worker_venv_path=worker_venv_path
            )
        from ..backends.training_backend import HFTrainingBackend

        return HFTrainingBackend(config)
