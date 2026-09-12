from __future__ import annotations

import torch
from tinker import types

_RLHF_LOSS_FNS = frozenset({"ppo", "grpo", "cispo", "importance_sampling", "dro"})


def validate_training_batch_inputs(
    data: list[types.Datum],
    loss_fn_name: str,
    max_model_len: int,
) -> None:
    """Validate all batch inputs upfront before any microstep or backward pass.

    Ensures:
    1. No row sequence is empty or exceeds max_model_len.
    2. Explicit target_tokens, weights, logprobs, advantages match sequence length.
    3. For RLHF losses, behavior logprobs and advantages are strictly present and finite.
    """
    if not data:
        return

    is_rlhf = loss_fn_name.lower() in _RLHF_LOSS_FNS

    for idx, datum in enumerate(data):
        tokens = datum.model_input.to_ints()
        seq_len = len(tokens)
        if seq_len == 0:
            raise ValueError(f"Row {idx}: empty token sequence is not supported")
        if seq_len > max_model_len:
            raise ValueError(
                f"Row {idx}: sequence length {seq_len} exceeds max_model_len {max_model_len}"
            )

        inputs = datum.loss_fn_inputs or {}

        # Validate target_tokens
        if "target_tokens" in inputs and inputs["target_tokens"] is not None:
            t_len = len(inputs["target_tokens"].to_torch())
            if t_len != seq_len:
                raise ValueError(
                    f"Row {idx}: target_tokens length {t_len} does not match model_input length {seq_len}"
                )

        # Validate weights
        if "weights" in inputs and inputs["weights"] is not None:
            w_tensor = inputs["weights"].to_torch()
            if len(w_tensor) != seq_len:
                raise ValueError(
                    f"Row {idx}: weights length {len(w_tensor)} does not match model_input length {seq_len}"
                )
            if not torch.isfinite(w_tensor).all():
                raise ValueError(f"Row {idx}: weights tensor contains NaN or Inf")

        if is_rlhf:
            # RLHF strictly requires logprobs
            if "logprobs" not in inputs or inputs["logprobs"] is None:
                raise ValueError(
                    f"Row {idx}: missing required 'logprobs' input for RLHF loss '{loss_fn_name}'"
                )
            lp_tensor = inputs["logprobs"].to_torch()
            if len(lp_tensor) != seq_len:
                raise ValueError(
                    f"Row {idx}: logprobs length {len(lp_tensor)} does not match model_input length {seq_len}"
                )
            if not torch.isfinite(lp_tensor).all():
                raise ValueError(f"Row {idx}: logprobs contains NaN or Inf")

            # RLHF strictly requires advantages
            if "advantages" not in inputs or inputs["advantages"] is None:
                raise ValueError(
                    f"Row {idx}: missing required 'advantages' input for RLHF loss '{loss_fn_name}'"
                )
            adv_tensor = inputs["advantages"].to_torch()
            if len(adv_tensor) != seq_len:
                raise ValueError(
                    f"Row {idx}: advantages length {len(adv_tensor)} does not match model_input length {seq_len}"
                )
            if not torch.isfinite(adv_tensor).all():
                raise ValueError(f"Row {idx}: advantages contains NaN or Inf")
