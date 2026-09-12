from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock
from tuft.backends.validation import validate_training_batch_inputs


class MockTensor:
    def __init__(self, data):
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def to_torch(self):
        return self

    def all(self):
        return all(not (math.isnan(x) or math.isinf(x)) for x in self.data)


class MockModelInput:
    def __init__(self, ints):
        self.ints = ints

    def to_ints(self):
        return self.ints


class MockDatum:
    def __init__(self, tokens, inputs=None):
        self.model_input = MockModelInput(tokens)
        self.loss_fn_inputs = inputs or {}


def test_validate_valid_datum():
    d = MockDatum(
        [1, 2, 3, 4],
        {"target_tokens": MockTensor([1, 2, 3, 4]), "weights": MockTensor([1.0, 1.0, 1.0, 0.0])},
    )
    validate_training_batch_inputs([d], "cross_entropy", 1024)


def test_validate_exceeds_max_len():
    d = MockDatum(list(range(10)))
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        validate_training_batch_inputs([d], "cross_entropy", 8)


def test_validate_target_tokens_mismatch():
    d = MockDatum([1, 2, 3, 4], {"target_tokens": MockTensor([1, 2, 3])})
    with pytest.raises(ValueError, match="target_tokens length"):
        validate_training_batch_inputs([d], "cross_entropy", 1024)


def test_validate_weights_mismatch():
    d = MockDatum([1, 2, 3, 4], {"weights": MockTensor([1.0, 1.0])})
    with pytest.raises(ValueError, match="weights length"):
        validate_training_batch_inputs([d], "cross_entropy", 1024)


def test_validate_weights_nan():
    d = MockDatum([1, 2, 3, 4], {"weights": MockTensor([1.0, float("nan"), 1.0, 0.0])})
    with pytest.raises(ValueError, match="weights tensor contains NaN"):
        validate_training_batch_inputs([d], "cross_entropy", 1024)


def test_validate_rlhf_missing_logprobs():
    d = MockDatum([1, 2, 3, 4], {"advantages": MockTensor([1.0, 1.0, 1.0, 1.0])})
    with pytest.raises(ValueError, match="missing required 'logprobs'"):
        validate_training_batch_inputs([d], "ppo", 1024)


def test_validate_rlhf_missing_advantages():
    d = MockDatum([1, 2, 3, 4], {"logprobs": MockTensor([-0.5, -0.5, -0.5, -0.5])})
    with pytest.raises(ValueError, match="missing required 'advantages'"):
        validate_training_batch_inputs([d], "ppo", 1024)


def test_validate_rlhf_valid():
    d = MockDatum(
        [1, 2, 3, 4],
        {"logprobs": MockTensor([-0.5] * 4), "advantages": MockTensor([1.0] * 4)},
    )
    validate_training_batch_inputs([d], "ppo", 1024)
