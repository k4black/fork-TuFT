from __future__ import annotations

import math
from typing import Any

import pytest
from tinker import types

from tuft.backends.validation import validate_training_batch_inputs


class MockTensor:
    def __init__(self, data: list[float] | list[int]):
        self.data = list(data)

    def __len__(self) -> int:
        return len(self.data)

    def to_torch(self) -> Any:
        return self

    def all(self) -> bool:
        return all(not (math.isnan(x) or math.isinf(x)) for x in self.data)


def make_datum(tokens: list[int], inputs: dict[str, Any] | None = None) -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens),
        loss_fn_inputs=inputs,  # type: ignore[arg-type]
    )


def test_validate_valid_datum():
    d = make_datum(
        [1, 2, 3, 4],
        {"target_tokens": MockTensor([1, 2, 3, 4]), "weights": MockTensor([1.0, 1.0, 1.0, 0.0])},
    )
    validate_training_batch_inputs([d], "cross_entropy", 1024)


def test_validate_exceeds_max_len():
    d = make_datum(list(range(10)))
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        validate_training_batch_inputs([d], "cross_entropy", 8)


def test_validate_target_tokens_mismatch():
    d = make_datum([1, 2, 3, 4], {"target_tokens": MockTensor([1, 2, 3])})
    with pytest.raises(ValueError, match="target_tokens length"):
        validate_training_batch_inputs([d], "cross_entropy", 1024)


def test_validate_weights_mismatch():
    d = make_datum([1, 2, 3, 4], {"weights": MockTensor([1.0, 1.0])})
    with pytest.raises(ValueError, match="weights length"):
        validate_training_batch_inputs([d], "cross_entropy", 1024)


def test_validate_weights_nan():
    d = make_datum([1, 2, 3, 4], {"weights": MockTensor([1.0, float("nan"), 1.0, 0.0])})
    with pytest.raises(ValueError, match="weights tensor contains NaN"):
        validate_training_batch_inputs([d], "cross_entropy", 1024)


def test_validate_rlhf_missing_logprobs():
    d = make_datum([1, 2, 3, 4], {"advantages": MockTensor([1.0, 1.0, 1.0, 1.0])})
    with pytest.raises(ValueError, match="missing required 'logprobs'"):
        validate_training_batch_inputs([d], "ppo", 1024)


def test_validate_rlhf_missing_advantages():
    d = make_datum([1, 2, 3, 4], {"logprobs": MockTensor([-0.5, -0.5, -0.5, -0.5])})
    with pytest.raises(ValueError, match="missing required 'advantages'"):
        validate_training_batch_inputs([d], "ppo", 1024)


def test_validate_rlhf_valid():
    d = make_datum(
        [1, 2, 3, 4],
        {"logprobs": MockTensor([-0.5] * 4), "advantages": MockTensor([1.0] * 4)},
    )
    validate_training_batch_inputs([d], "ppo", 1024)
