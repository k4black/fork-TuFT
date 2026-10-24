"""Custom vLLM GPU worker for TuFT.

Adapted from Trinity-RFT (``trinity/common/models/vllm_worker.py`` and
``trinity/common/models/vllm_patch/worker_patch.py``, Apache-2.0,
https://github.com/agentscope-ai/Trinity-RFT), with the weight-sync pieces
(MoE weight-loader patch and checkpoint weight-transfer engine) removed --
TuFT delivers trained weights to the sampler as on-disk LoRA adapters and
never syncs full model weights into vLLM.

WHY THIS PATCH EXISTS:
  Upstream vLLM does not apply temperature scaling when computing *prompt*
  logprobs (sampled-token logprobs are temperature-scaled natively via the
  ``logprobs_mode="processed_logprobs"`` engine argument). TuFT's
  Tinker-compatible sampling API exposes prompt logprobs, which are consumed
  by importance-sampling-style losses and must therefore be consistent with
  the actual sampling distribution when temperature != 1.0.
  ``patch_vllm_prompt_logprobs`` replaces
  ``GPUModelRunner._get_prompt_logprobs_dict`` with a copy of the upstream
  implementation plus the missing ``logits /= temperature`` step (the blocks
  marked ``PATCH START``/``PATCH END`` below are the only deltas).
  The same patch scores ``target_prompt_logprobs`` ids and draws ``prompt_alt_tokens_k``
  alternatives (``_score_prompt_rows``).

This module is imported inside vLLM worker processes (via the
``worker_cls`` engine argument set in ``vllm_engine.py``), so it may import
vllm at module level; nothing in TuFT's server process imports it.
"""

from types import MethodType
from typing import Any

import torch
import vllm
from packaging.version import InvalidVersion, parse as parse_version
from vllm.platforms import current_platform
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as VLLMGPUWorker


def get_vllm_version():
    """Parsed vLLM version (self-compiled builds fall back to the lowest supported)."""
    try:
        return parse_version(vllm.__version__)
    except InvalidVersion:
        return parse_version("0.19.1")


def _score_prompt_rows(request: Any, logprobs: torch.Tensor, start: int) -> None:
    """Fill rows ``start:start+len(logprobs)`` of ``request.tuft_scores`` from ``extra_args``.

    Row i covers prompt position i + 1. Its columns, in order:
    - ``tuft_target_ids`` (``[len(prompt) - 1, K]``, ``-1`` = none): the logprob of each id,
      0.0 for ``-1``, -inf for an id past the vocab.
    - ``tuft_alt_k`` draws from the tempered distribution: k ids (as floats, exact below
      2**24), then their k logprobs.
    The scores reach ``RequestOutput.prompt_token_id_logprobs`` (vLLM's V2-only channel,
    which the V1 runner never fills) through the ``sample_tokens`` wrapper.
    """
    extra = request.sampling_params.extra_args or {}
    target, alt_k = extra.get("tuft_target_ids"), extra.get("tuft_alt_k", 0)
    if target is None and not alt_k:
        return
    rows = slice(start, start + logprobs.shape[0])
    parts = []
    if target is not None:
        ids = torch.tensor(target[rows], device=logprobs.device)
        vocab = logprobs.shape[-1]
        picked = logprobs.gather(1, ids.clamp(0, vocab - 1))
        parts.append(picked.masked_fill(ids >= vocab, float("-inf")).masked_fill(ids < 0, 0.0))
    if alt_k:
        params = request.sampling_params
        if params.temperature < 1e-5:  # greedy: every draw is the argmax
            draws = logprobs.argmax(-1, keepdim=True).expand(-1, alt_k)
        else:
            # shortcut: draws ignore top_k/top_p; mask the logits first if a caller needs them.
            gen = None
            if params.seed is not None:
                gen = torch.Generator(logprobs.device).manual_seed(params.seed + start)
            draws = torch.multinomial(logprobs.exp(), alt_k, replacement=True, generator=gen)
        parts += [draws.float(), logprobs.gather(1, draws)]
    scores = torch.cat(parts, 1).float().cpu()
    if getattr(request, "tuft_scores", None) is None:
        request.tuft_scores = scores.new_zeros(len(request.prompt_token_ids) - 1, scores.shape[1])
    request.tuft_scores[rows] = scores


def patch_vllm_prompt_logprobs(model_runner: GPUModelRunner):  # noqa: C901
    """Patch vLLM model runner to apply temperature scaling to prompt logprobs."""
    version = get_vllm_version()
    cpu = current_platform.is_cpu()

    def _get_prompt_logprobs_dict_v21(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        """Patched version of _get_prompt_logprobs_dict for vLLM v0.21.0."""
        # PATCH START: prefer the snapshot taken before the drafter ran (see below).
        snapshot = vars(self).pop("_tuft_prompt_hidden_states", None)
        if snapshot is not None:
            hidden_states = snapshot
        # PATCH END
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = request.in_progress_prompt_logprobs_cpu
            if logprobs_tensors is None:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                request.in_progress_prompt_logprobs_cpu = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            # PATCH START: the LoRA logits mapping has one row per request, which
            # the CPU kernel cannot match to prompt rows. Map only this request;
            # its single row broadcasts over all prompt rows.
            punica = getattr(self.model.logits_processor, "punica_wrapper", None) if cpu else None
            if punica is None:
                logits = self.model.compute_logits(prompt_hidden_states)
            else:
                saved = punica._sampler_indices[0].item(), punica.indices_len[1]
                punica._sampler_indices[0] = punica._sampler_indices[req_idx].item()
                punica.indices_len[1] = 1
                try:
                    logits = self.model.compute_logits(prompt_hidden_states)
                finally:
                    punica._sampler_indices[0], punica.indices_len[1] = saved
            # PATCH END

            # PATCH START
            temp = request.sampling_params.temperature
            if temp >= 1e-5:
                logits.div_(temp)
            # PATCH END

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)

            _score_prompt_rows(request, logprobs, start_idx)  # PATCH: TuFT extra_args
            token_ids, logprobs, ranks, *_ = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            self.requests[req_id].in_progress_prompt_logprobs_cpu = None
            # PATCH START: the sample_tokens wrapper below ships the finished scores.
            scores = vars(self.requests[req_id]).pop("tuft_scores", None)
            if scores is not None:
                vars(self).setdefault("_tuft_scores", {})[req_id] = scores
            # PATCH END

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    if version < parse_version("0.21.0"):
        raise ValueError(
            f"Unsupported vllm version for patching: {vllm.__version__}; expected >=0.21.0."
        )
    model_runner._get_prompt_logprobs_dict = MethodType(_get_prompt_logprobs_dict_v21, model_runner)

    # vLLM 0.28 to 0.32 run a padded EAGLE/MTP drafter before prompt-logprob bookkeeping and its
    # CUDA graphs share the target's memory pool, so the target hidden states are
    # overwritten first (vllm-project/vllm#53488). Snapshot them before the drafter runs.
    propose = model_runner.propose_draft_token_ids

    def _propose_draft_token_ids(
        self,
        scheduler_output: Any,
        sampled_token_ids: Any,
        sampling_metadata: Any,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        spec = self.speculative_config
        if (
            self.num_prompt_logprobs
            and spec is not None
            and not spec.disable_padded_drafter_batch
            and (spec.use_eagle() or spec.uses_draft_model() or spec.uses_extract_hidden_states())
        ):
            num_tokens = scheduler_output.total_num_scheduled_tokens
            self._tuft_prompt_hidden_states = hidden_states[:num_tokens].clone()
        return propose(
            scheduler_output, sampled_token_ids, sampling_metadata, hidden_states, *args, **kwargs
        )

    model_runner.propose_draft_token_ids = MethodType(_propose_draft_token_ids, model_runner)

    # Ship the scores of requests that finished prefill this step (see _score_prompt_rows).
    sample_tokens = model_runner.sample_tokens

    def _sample_tokens(self, *args: Any, **kwargs: Any) -> Any:
        output = sample_tokens(*args, **kwargs)
        scores = vars(self).pop("_tuft_scores", None)
        if scores:
            # AsyncGPUModelRunnerOutput returns this object from get_output().
            getattr(output, "_model_runner_output", output).prompt_token_id_logprobs_dict = scores
        return output

    model_runner.sample_tokens = MethodType(_sample_tokens, model_runner)


class TuFTGPUWorker(VLLMGPUWorker):
    """vLLM GPU worker with TuFT patches applied.

    ``apply_patches`` is invoked on every worker right after engine creation
    via ``collective_rpc`` (see ``VLLMEngine.prepare`` in ``vllm_engine.py``).
    """

    def apply_patches(self, *args: Any, **kwargs: Any) -> None:
        """Apply necessary patches to vLLM inside the worker process."""
        patch_vllm_prompt_logprobs(self.model_runner)


if current_platform.is_cpu():
    from vllm.v1.worker.cpu_worker import CPUWorker

    class TuFTCPUWorker(CPUWorker):
        """vLLM CPU worker with the same patches, for GPU-less CI."""

        apply_patches = TuFTGPUWorker.apply_patches
