from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tuft.checkpoints import CheckpointRecord
from tuft.config import AppConfig, ModelConfig
from tuft.oai.model_resolver import resolve_model


def test_resolve_model_immutable_lora_id(tmp_path: Path):
    checkpoint_dir = tmp_path / "user1" / "run1" / "checkpoints" / "0001"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    record = CheckpointRecord.from_training_run(
        training_run_id="run1",
        checkpoint_name="0001",
        owner_name="user1",
        checkpoint_type="sampler",
        checkpoint_root_dir=tmp_path,
    )
    record.adapter_path.mkdir(parents=True, exist_ok=True)
    record.save_metadata(
        session_id="s1",
        base_model="Qwen/Qwen3-4B",
        lora_rank=16,
        lora_alpha=32,
    )

    app_config = AppConfig(
        checkpoint_dir=tmp_path,
        supported_models=[
            ModelConfig(
                model_name="Qwen/Qwen3-4B",
                model_path=Path("Qwen/Qwen3-4B"),
                max_model_len=1024,
            )
        ],
    )

    resolved = resolve_model(record.tinker_path, app_config)
    assert resolved.lora_id == "run1:0001"
    assert resolved.backend_model_name == "run1:0001"


@pytest.mark.asyncio
async def test_sampling_controller_evict_removes_adapter():
    from tuft.sampling_controller import SamplingController, SamplingSessionRecord

    config = AppConfig(
        supported_models=[
            ModelConfig(
                model_name="Qwen/Qwen3-4B",
                model_path=Path("Qwen/Qwen3-4B"),
                max_model_len=1024,
            )
        ]
    )
    controller = SamplingController(config)
    mock_backend = AsyncMock()
    controller._base_backends["Qwen/Qwen3-4B"] = mock_backend

    record = SamplingSessionRecord(
        sampling_session_id="session1",
        session_id="s1",
        user_id="user1",
        base_model="Qwen/Qwen3-4B",
        model_id="m1",
        model_path="/tmp/adapter",
        session_seq_id=0,
    )
    controller.sampling_sessions["session1"] = record

    await controller.evict_model("m1", "user1")

    assert "session1" not in controller.sampling_sessions
    mock_backend.remove_adapter.assert_awaited_once_with("session1")
