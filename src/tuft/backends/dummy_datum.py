from __future__ import annotations

import torch
from tinker import types


def create_zero_weight_dummy_datum(reference_datum: types.Datum) -> types.Datum:
    """Create a dummy datum whose loss contribution is zero.

    Copies sequence structure from reference_datum but sets weights to all zeros
    so it participates in collective backward passes without affecting model gradients.
    """
    ints = reference_datum.model_input.to_ints()
    seq_len = len(ints)
    zero_weights = torch.zeros(seq_len, dtype=torch.float32)

    loss_inputs: dict[str, types.TensorData] = {
        "weights": types.TensorData.from_torch(zero_weights),
    }

    ref_inputs = reference_datum.loss_fn_inputs or {}
    if "target_tokens" in ref_inputs and ref_inputs["target_tokens"] is not None:
        loss_inputs["target_tokens"] = ref_inputs["target_tokens"]

    if "logprobs" in ref_inputs and ref_inputs["logprobs"] is not None:
        loss_inputs["logprobs"] = types.TensorData.from_torch(
            torch.zeros(seq_len, dtype=torch.float32)
        )

    if "advantages" in ref_inputs and ref_inputs["advantages"] is not None:
        loss_inputs["advantages"] = types.TensorData.from_torch(
            torch.zeros(seq_len, dtype=torch.float32)
        )

    return types.Datum(
        model_input=reference_datum.model_input,
        loss_fn_inputs=loss_inputs,
    )
