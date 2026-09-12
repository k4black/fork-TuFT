from unittest.mock import MagicMock
import pytest
import torch
from tinker import types
from tuft.backends.dummy_datum import create_zero_weight_dummy_datum


def test_create_zero_weight_dummy_datum():
    ref = types.Datum(
        model_input=types.ModelInput.from_ints([10, 20, 30]),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_torch(torch.tensor([10, 20, 30])),
            "weights": types.TensorData.from_torch(torch.tensor([1.0, 1.0, 1.0])),
            "logprobs": types.TensorData.from_torch(torch.tensor([-0.1, -0.2, -0.3])),
            "advantages": types.TensorData.from_torch(torch.tensor([0.5, 0.5, 0.5])),
        },
    )

    dummy = create_zero_weight_dummy_datum(ref)
    assert dummy.model_input.to_ints() == [10, 20, 30]
    weights = dummy.loss_fn_inputs["weights"].to_torch()
    assert (weights == 0.0).all()
    assert len(weights) == 3
