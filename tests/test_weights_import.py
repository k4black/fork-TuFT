from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

from tuft import weights_import
from tuft.exceptions import InvalidRequestException


@pytest.mark.parametrize(
    ("bucket", "key", "allowed"),
    [
        ("b", "adapters/run1/", True),
        ("b", "adapters", True),
        ("other", "adapters/run1/", False),
        ("b", "adapters-evil/run1/", False),
        ("b", "adapters/../secret/", False),
        ("open", "anything/", True),
    ],
)
def test_s3_allowed(bucket: str, key: str, allowed: bool) -> None:
    prefixes = ["s3://b/adapters", "s3://open/"]
    assert weights_import.s3_allowed(bucket, key, prefixes) is allowed


def _write_adapter(directory: Path, *, bin_only: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text("{}")
    weights = "adapter_model.bin" if bin_only else "adapter_model.safetensors"
    (directory / weights).write_bytes(b"w")


def test_fetch_hf_uses_only_the_request_token(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_snapshot_download(repo, *, revision, allow_patterns, token, local_dir):
        calls.append((repo, revision, allow_patterns, token))
        _write_adapter(Path(local_dir) / "sub")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    weights_import.fetch("hf://org/repo@v1/sub", None, tmp_path / "a", [])
    weights_import.fetch("hf://org/repo/sub", "tok", tmp_path / "b", [])

    assert calls == [
        ("org/repo", "v1", ["sub/adapter_config.json", "sub/adapter_model.safetensors"], False),
        ("org/repo", None, ["sub/adapter_config.json", "sub/adapter_model.safetensors"], "tok"),
    ]
    assert sorted(p.name for p in (tmp_path / "a").iterdir()) == sorted(
        weights_import.ADAPTER_FILES
    )


def test_fetch_rejects_bin_only_adapter(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda repo, **kw: _write_adapter(Path(kw["local_dir"]), bin_only=True),
    )
    with pytest.raises(InvalidRequestException, match="only safetensors"):
        weights_import.fetch("hf://org/repo", None, tmp_path / "a", [])


def test_fetch_s3_under_prefix_only(tmp_path, monkeypatch) -> None:
    downloaded = []

    class FakeS3:
        def download_file(self, bucket, key, filename):
            downloaded.append((bucket, key))
            Path(filename).write_bytes(b"w")

    monkeypatch.setattr("boto3.client", lambda service: FakeS3())

    weights_import.fetch("s3://b/adapters/run1", None, tmp_path / "a", ["s3://b/adapters/"])
    assert downloaded == [
        ("b", "adapters/run1/adapter_config.json"),
        ("b", "adapters/run1/adapter_model.safetensors"),
    ]
    with pytest.raises(InvalidRequestException, match="import_s3_prefixes"):
        weights_import.fetch("s3://b/private/run1", None, tmp_path / "b", ["s3://b/adapters/"])


def test_check_shapes(tmp_path) -> None:
    model_dir = tmp_path / "model"
    Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
    ).save_pretrained(model_dir)
    q_proj = "base_model.model.model.layers.0.self_attn.q_proj"
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    def check(tensors: dict[str, torch.Tensor], targets: tuple[str, ...] = ("q_proj",)) -> None:
        (adapter / "adapter_config.json").write_text(json.dumps({"target_modules": targets}))
        save_file(tensors, adapter / "adapter_model.safetensors")
        weights_import.check_shapes(adapter, model_dir, rank=4)

    check(
        {
            f"{q_proj}.lora_A.weight": torch.zeros(4, 16),
            f"{q_proj}.lora_B.weight": torch.zeros(16, 4),
        }
    )
    with pytest.raises(InvalidRequestException, match="shape"):
        check({f"{q_proj}.lora_B.weight": torch.zeros(32, 4)})
    with pytest.raises(InvalidRequestException, match="no linear module"):
        check({"base_model.model.model.layers.0.nope.lora_A.weight": torch.zeros(4, 16)})
    with pytest.raises(InvalidRequestException, match="lora_A and lora_B"):
        check({f"{q_proj}.lora_A.weight": torch.zeros(4, 16)})
    with pytest.raises(InvalidRequestException, match="lora_A and lora_B"):
        check({})
    with pytest.raises(InvalidRequestException, match="every module"):
        check(
            {
                f"{q_proj}.lora_A.weight": torch.zeros(4, 16),
                f"{q_proj}.lora_B.weight": torch.zeros(16, 4),
            },
            targets=("q_proj", "k_proj"),
        )
    (adapter / "adapter_model.safetensors").write_bytes(b"not safetensors")
    with pytest.raises(InvalidRequestException, match="Unreadable"):
        weights_import.check_shapes(adapter, model_dir, rank=4)


def test_fetch_hf_rejects_traversal(tmp_path) -> None:
    with pytest.raises(InvalidRequestException, match=r"\.\."):
        weights_import.fetch("hf://org/repo/../../etc", None, tmp_path / "a", [])
