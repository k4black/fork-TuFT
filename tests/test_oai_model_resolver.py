"""Checkpoint identity regressions for the OpenAI-compatible router."""

import shutil

import pytest

from tuft.checkpoints import CheckpointRecord
from tuft.config import AppConfig, ModelConfig
from tuft.exceptions import CheckpointAccessDeniedException, CheckpointNotFoundException
from tuft.oai.model_resolver import resolve_model


def test_same_run_checkpoints_resolve_to_distinct_adapters(tmp_path):
    config = AppConfig(
        checkpoint_dir=tmp_path,
        supported_models=[
            ModelConfig(model_name="test-model", model_path=tmp_path / "base", max_model_len=128)
        ],
    )
    checkpoints = []
    for name in ("000001", "000002"):
        checkpoint = CheckpointRecord.from_training_run(
            training_run_id="same-run",
            checkpoint_name=name,
            owner_name="tester",
            checkpoint_type="sampler",
            checkpoint_root_dir=tmp_path,
        )
        checkpoint.save_metadata(base_model="test-model", session_id="session", lora_rank=4)
        checkpoint.adapter_path.mkdir()
        checkpoints.append(checkpoint)

    first, second = [
        resolve_model(checkpoint.tinker_path, config, "tester") for checkpoint in checkpoints
    ]
    assert first.lora_id == first.backend_model_name == "same-run:000001"
    assert second.lora_id == second.backend_model_name == "same-run:000002"
    assert first.lora_adapter_path == checkpoints[0].adapter_path
    assert second.lora_adapter_path == checkpoints[1].adapter_path
    assert resolve_model(checkpoints[0].tinker_path, config, "tester") == first


def test_checkpoint_resolves_only_for_owner_or_when_public(tmp_path):
    config = AppConfig(
        checkpoint_dir=tmp_path,
        supported_models=[
            ModelConfig(model_name="test-model", model_path=tmp_path / "base", max_model_len=128)
        ],
    )
    checkpoint = CheckpointRecord.from_training_run(
        training_run_id="run",
        checkpoint_name="000001",
        owner_name="tester",
        checkpoint_type="sampler",
        checkpoint_root_dir=tmp_path,
    )
    checkpoint.save_metadata(base_model="test-model", session_id="session", lora_rank=4)
    checkpoint.adapter_path.mkdir()

    with pytest.raises(CheckpointAccessDeniedException):
        resolve_model(checkpoint.tinker_path, config, "intruder")
    checkpoint.set_visibility(True)
    assert resolve_model(checkpoint.tinker_path, config, "intruder").lora_id == "run:000001"

    shutil.rmtree(checkpoint.adapter_path)
    with pytest.raises(CheckpointNotFoundException):
        resolve_model(checkpoint.tinker_path, config, "tester")
