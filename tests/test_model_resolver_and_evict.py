from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from tuft.checkpoints import CheckpointMetadata, CheckpointRecord
from tuft.config import AppConfig, ModelConfig
from tuft.oai.model_resolver import resolve_model


def test_resolve_model_immutable_lora_id(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_path = ckpt_dir / "user1" / "run1" / "checkpoints" / "0001"
    adapter_path = ckpt_path / "adapter"
    adapter_path.mkdir(parents=True)

    metadata = CheckpointMetadata(
        base_model="Qwen/Qwen3-4B",
        step=1,
        loss=0.5,
    )
    (ckpt_path / "metadata.json").write_text(metadata.model_dump_json(), encoding="utf-8")

    app_config = AppConfig(
        checkpoint_dir=ckpt_dir,
        supported_models=[
            ModelConfig(
                model_name="Qwen/Qwen3-4B",
                model_path=Path("Qwen/Qwen3-4B"),
                max_model_len=1024,
            )
        ],
    )

    resolved = resolve_model("tinker://user1/run1/checkpoints/0001", app_config)
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
    )
    controller.sampling_sessions["session1"] = record

    await controller.evict_model("m1", "user1")

    assert "session1" not in controller.sampling_sessions
    mock_backend.remove_adapter.assert_awaited_once_with("session1")
