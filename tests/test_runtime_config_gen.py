"""Tests for tuft.runtime._config_gen module."""

from __future__ import annotations

import json

from tuft.runtime._config_gen import (
    _infer_max_model_len,
    _read_max_position_embeddings,
    generate_api_key,
    generate_config_dict,
    generate_config_file,
)


class TestReadMaxPositionEmbeddings:
    def test_reads_from_config_json(self, tmp_path):
        config = {"max_position_embeddings": 8192, "hidden_size": 4096}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _read_max_position_embeddings(tmp_path) == 8192

    def test_returns_none_when_no_config(self, tmp_path):
        assert _read_max_position_embeddings(tmp_path) is None

    def test_returns_none_on_invalid_json(self, tmp_path):
        (tmp_path / "config.json").write_text("not json")
        assert _read_max_position_embeddings(tmp_path) is None

    def test_reads_n_positions(self, tmp_path):
        config = {"n_positions": 2048}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _read_max_position_embeddings(tmp_path) == 2048


class TestInferMaxModelLen:
    def test_caps_at_32768(self, tmp_path):
        config = {"max_position_embeddings": 131072}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _infer_max_model_len(tmp_path) == 32768

    def test_uses_value_when_below_cap(self, tmp_path):
        config = {"max_position_embeddings": 4096}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert _infer_max_model_len(tmp_path) == 4096

    def test_default_when_no_config(self, tmp_path):
        assert _infer_max_model_len(tmp_path) == 4096


class TestGenerateApiKey:
    def test_format(self):
        key = generate_api_key()
        assert key.startswith("tml-")
        assert len(key) > 20

    def test_uniqueness(self):
        assert generate_api_key() != generate_api_key()


class TestGenerateConfigDict:
    def test_basic_structure(self, tmp_path):
        model_path = tmp_path / "Qwen2.5-0.5B-Instruct"
        model_path.mkdir()
        config_json = {"max_position_embeddings": 16384}
        (model_path / "config.json").write_text(json.dumps(config_json))

        result = generate_config_dict(model_path, api_key="test-key")
        assert "supported_models" in result
        assert len(result["supported_models"]) == 1
        assert result["supported_models"][0]["model_name"] == "Qwen2.5-0.5B-Instruct"
        assert result["supported_models"][0]["max_model_len"] == 16384
        assert result["authorized_users"] == {"test-key": "local"}  # pragma: allowlist secret


class TestGenerateConfigFile:
    def test_creates_yaml_file(self, tmp_path):
        model_path = tmp_path / "test-model"
        model_path.mkdir()

        config_path, api_key = generate_config_file(model_path, api_key="my-key")
        assert config_path.exists()
        assert config_path.suffix == ".yaml"
        assert api_key == "my-key"  # pragma: allowlist secret

        content = config_path.read_text()
        assert "test-model" in content
