"""Colocate "hf": sample on the HF training actor, so one weight copy serves both roles."""

from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
from tinker import types

from .base_backend import BaseSamplingBackend
from .sampling_backend import _build_sample_response


class HFSamplingBackend(BaseSamplingBackend):
    """Forwards sampling to ``HFTrainingModel.generate`` on ``training_model_<model_name>``."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.actor: Any = None
        self._paths: dict[str, Path] = {}  # sampler lora id -> adapter dir, loaded lazily

    async def async_init(self) -> None:
        import ray

        self.actor = ray.get_actor("training_model_" + self.base_model)
        await self.actor.async_init.remote()

    async def add_adapter(self, lora_id: str, adapter_path: Path) -> None:
        if not adapter_path.exists():
            raise ValueError(f"LoRA adapter path {adapter_path} does not exist.")
        self._paths[lora_id] = adapter_path

    async def remove_adapter(self, lora_id: str) -> None:
        if self._paths.pop(lora_id, None) is not None:
            await self.actor.remove_sampling_adapter.remote(lora_id)

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
        path = None
        if lora_id is not None:
            path = self._paths.get(lora_id)
            if path is None:
                raise ValueError(f"LoRA adapter {lora_id} not found in backend.")
        stop = sampling_params.stop or []
        out = await self.actor.generate.remote(
            prompt=prompt.to_ints(),
            num_samples=num_samples,
            max_tokens=sampling_params.max_tokens if sampling_params.max_tokens is not None else 16,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            seed=sampling_params.seed,
            stop=[stop] if isinstance(stop, str) else list(stop),
            prompt_logprobs=topk_prompt_logprobs if include_prompt_logprobs else None,
            logprobs=topk_sample_logprobs,
            lora_id=lora_id,
            adapter_path=str(path) if path else None,
        )
        return _build_sample_response(
            out, include_prompt_logprobs, topk_prompt_logprobs, topk_sample_logprobs
        )


# ---- actor side: called by HFTrainingModel under its lock ----


@lru_cache
def _tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def remove_sampling_adapter(m: Any, lora_id: str) -> None:
    if lora_id in m.sampling_adapters:
        del m.sampling_adapters[lora_id]
        m.model.delete_adapter(lora_id)


def _use_sampling_adapter(m: Any, lora_id: str, adapter_path: str) -> None:
    """Load inference-only on first use; keep at most ``max_loras`` loaded."""
    if lora_id in m.sampling_adapters:
        m.sampling_adapters.move_to_end(lora_id)
    else:
        m.model.load_adapter(adapter_path, adapter_name=lora_id, is_trainable=False)
        m.sampling_adapters[lora_id] = None
        while len(m.sampling_adapters) > m.config.max_loras:
            remove_sampling_adapter(m, next(iter(m.sampling_adapters)))
    m.model.set_adapter(lora_id, inference_mode=True)


def _position(row: torch.Tensor, token: int, k: int) -> dict[int, SimpleNamespace]:
    """One position in vLLM's logprobs shape: the chosen token first, then the top k."""
    lp = row[token]
    out = {token: SimpleNamespace(logprob=lp.item(), rank=int((row > lp).sum()) + 1)}
    if k:
        values, ids = row.topk(k)
        for rank, (tid, value) in enumerate(zip(ids.tolist(), values.tolist(), strict=True), 1):
            out.setdefault(tid, SimpleNamespace(logprob=value, rank=rank))
    return out


def _pick(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float, gen: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next tokens and the logprobs they were sampled from (vLLM ``processed_logprobs``)."""
    if temperature < 1e-5:
        logprobs = logits.log_softmax(-1)
        return logprobs.argmax(-1), logprobs
    logits = logits / temperature
    if 0 < top_k < logits.shape[-1]:
        kth = logits.topk(top_k).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, ids = logits.sort(-1, descending=True)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p
        logits = logits.scatter(-1, ids, sorted_logits.masked_fill(drop, float("-inf")))
    logprobs = logits.log_softmax(-1)
    return torch.multinomial(logprobs.exp(), 1, generator=gen).squeeze(-1), logprobs


@torch.no_grad()
def generate(
    m: Any,
    *,
    prompt: list[int],
    num_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int | None,
    stop: list[str | int],
    prompt_logprobs: int | None,
    logprobs: int,
    lora_id: str | None,
    adapter_path: str | None,
) -> SimpleNamespace:
    """Sample ``num_samples`` continuations; returns what ``_build_sample_response`` reads."""
    model = m.model
    was_training = model.training  # read first: peft load_adapter switches to eval
    stop_strs = [s for s in stop if isinstance(s, str)]
    stop_ids = {s for s in stop if isinstance(s, int)}
    for eos in (model.generation_config.eos_token_id, model.config.eos_token_id):
        stop_ids.update(eos if isinstance(eos, list) else [] if eos is None else [eos])
    device = next(model.parameters()).device
    gen = torch.Generator(device).manual_seed(seed) if seed is not None else None
    try:
        if lora_id is not None:
            assert adapter_path is not None
            _use_sampling_adapter(m, lora_id, adapter_path)
            adapters: Any = nullcontext()
        else:
            adapters = model.disable_adapter()
        model.eval()  # also turns off gradient checkpointing, so the KV cache works
        with adapters:
            # ponytail: full-prompt logits in fp32 for prompt logprobs; chunk if long prompts OOM.
            out = model(
                input_ids=torch.tensor([prompt], device=device),
                use_cache=True,
                logits_to_keep=0 if prompt_logprobs is not None else 1,
            )
            prompt_positions: list = [None]
            if prompt_logprobs is not None:
                scaled = out.logits[0, :-1].float()
                if temperature >= 1e-5:  # matches the TuFT vLLM worker patch
                    scaled = scaled / temperature
                rows = scaled.log_softmax(-1)
                for row, token in zip(rows, prompt[1:], strict=True):
                    prompt_positions.append(_position(row, token, prompt_logprobs))
            cache = out.past_key_values
            cache.batch_repeat_interleave(num_samples)
            logits = out.logits[:, -1].float().repeat(num_samples, 1)
            seqs = [
                SimpleNamespace(token_ids=[], logprobs=[], finish_reason=None)
                for _ in range(num_samples)
            ]
            for _ in range(max_tokens):
                tokens, rows = _pick(logits, temperature, top_k, top_p, gen)
                for seq, token, row in zip(seqs, tokens.tolist(), rows, strict=True):
                    if seq.finish_reason is not None:
                        continue
                    seq.token_ids.append(token)
                    seq.logprobs.append(_position(row, token, logprobs))
                    text = (
                        _tokenizer(str(m.config.model_path)).decode(seq.token_ids)
                        if stop_strs
                        else ""
                    )
                    if token in stop_ids or any(s in text for s in stop_strs):
                        seq.finish_reason = "stop"
                    elif len(seq.token_ids) == max_tokens:
                        seq.finish_reason = "length"
                if all(seq.finish_reason is not None for seq in seqs):
                    break
                out = model(input_ids=tokens[:, None], past_key_values=cache, use_cache=True)
                logits = out.logits[:, -1].float()
    finally:
        model.train(was_training)
    return SimpleNamespace(prompt_logprobs=prompt_positions, outputs=seqs)
