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
        # The training actor reads this path itself: same node or a shared checkpoint_dir.
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

PROMPT_CHUNK = 1024  # prompt positions per lm_head call for prompt logprobs


@lru_cache
def _tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def remove_sampling_adapter(m: Any, lora_id: str) -> None:
    if lora_id in m.sampling_adapters:
        m.model.delete_adapter(lora_id)
        del m.sampling_adapters[lora_id]


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


def _ranked(rows: torch.Tensor, tokens: torch.Tensor, k: int) -> list[torch.Tensor]:
    """Per row, on device: the token's logprob, its rank, and the top-k values and ids."""
    chosen = rows.gather(-1, tokens[:, None])
    rank = (rows > chosen).sum(-1) + 1
    top = rows.topk(k)
    return [tokens, chosen.squeeze(-1), rank, top.values, top.indices]


def _positions(ranked: list[torch.Tensor]) -> list[dict[int, SimpleNamespace]]:
    """vLLM's logprobs shape, the chosen token first; one host copy per tensor."""
    tokens, chosen, rank, values, ids = (t.tolist() for t in ranked)
    out = []
    for token, lp, r, top_values, top_ids in zip(tokens, chosen, rank, values, ids, strict=True):
        position = {token: SimpleNamespace(logprob=lp, rank=r)}
        for top_rank, (tid, value) in enumerate(zip(top_ids, top_values, strict=True), 1):
            position.setdefault(tid, SimpleNamespace(logprob=value, rank=top_rank))
        out.append(position)
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
        if isinstance(eos, list):
            stop_ids.update(eos)
        elif eos is not None:
            stop_ids.add(eos)
    # A stop string spans at most 4 tokens per character (byte-level BPE).
    window = 4 * max(map(len, stop_strs), default=0)
    tokenizer = _tokenizer(str(m.config.model_path)) if stop_strs else None
    device = next(model.parameters()).device
    gen = torch.Generator(device).manual_seed(seed) if seed is not None else None
    base = model.get_base_model()
    head = base.get_output_embeddings()
    try:
        if lora_id is not None:
            assert adapter_path is not None
            _use_sampling_adapter(m, lora_id, adapter_path)
            adapters: Any = nullcontext()
        else:
            adapters = model.disable_adapter()
        model.eval()  # also turns off gradient checkpointing, so the KV cache works
        with adapters:
            # The decoder, not the LM: full-prompt logits would not fit for long prompts.
            # Nested LMs (qwen3_5) return the text model; text-only positions follow the cache.
            out = base.get_decoder()(
                input_ids=torch.tensor([prompt], device=device), use_cache=True
            )
            hidden = out.last_hidden_state[0]
            prompt_positions: list = [None]
            if prompt_logprobs is not None:
                targets = torch.tensor(prompt[1:], device=device)
                for i in range(0, len(prompt) - 1, PROMPT_CHUNK):
                    j = min(i + PROMPT_CHUNK, len(prompt) - 1)
                    rows = head(hidden[i:j]).float()
                    if temperature >= 1e-5:  # matches the TuFT vLLM worker patch
                        rows = rows / temperature
                    ranked = _ranked(rows.log_softmax(-1), targets[i:j], prompt_logprobs)
                    prompt_positions += _positions(ranked)
            cache = out.past_key_values
            cache.batch_repeat_interleave(num_samples)
            logits = head(hidden[-1:]).float().repeat(num_samples, 1)
            steps: list[list[torch.Tensor]] = []
            lengths: list[int | None] = [None] * num_samples
            reasons = ["length"] * num_samples
            token_ids: list[list[int]] = [[] for _ in range(num_samples)]
            for step in range(max_tokens):
                tokens, rows = _pick(logits, temperature, top_k, top_p, gen)
                steps.append(_ranked(rows, tokens, logprobs))
                for i, token in enumerate(tokens.tolist()):  # one sync per step
                    if lengths[i] is not None:
                        continue
                    token_ids[i].append(token)
                    tail = tokenizer.decode(token_ids[i][-window:]) if tokenizer else ""
                    if token in stop_ids or any(s in tail for s in stop_strs):
                        lengths[i], reasons[i] = step + 1, "stop"
                if all(n is not None for n in lengths) or step + 1 == max_tokens:
                    break
                out = model(input_ids=tokens[:, None], past_key_values=cache, use_cache=True)
                logits = out.logits[:, -1].float()
            # [steps, n, ...] -> one host copy per tensor, then per sequence.
            stacked = [torch.stack(parts, 1).cpu() for parts in zip(*steps, strict=True)]
            per_seq = [
                _positions([t[i] for t in stacked]) if steps else [] for i in range(num_samples)
            ]
    finally:
        model.train(was_training)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free the KV cache before the next training step
    seqs = [
        SimpleNamespace(
            token_ids=ids,
            logprobs=positions[: len(ids)],
            finish_reason=reason,
        )
        for ids, positions, reason in zip(token_ids, per_seq, reasons, strict=True)
    ]
    return SimpleNamespace(prompt_logprobs=prompt_positions, outputs=seqs)
