"""Ray custom-resource placement of training vs sampling actors."""

from pathlib import Path

import pytest

from tuft.config import ModelConfig


class _RecordingRemote:
    """Stand-in for ``ray.remote(cls)`` that records the ``.options()`` kwargs."""

    def __init__(self) -> None:
        self.options_kwargs: dict = {}

    def __call__(self, _cls):
        return self

    def options(self, **kwargs):
        self.options_kwargs = kwargs
        return self

    def remote(self, *_args, **_kwargs):
        return object()


def _config(**overrides) -> ModelConfig:
    return ModelConfig(
        model_name="test",
        model_path=Path("/tmp/qwen-model"),
        max_model_len=1024,
        **overrides,
    )


@pytest.mark.parametrize(
    "train_gpu_resource, expected",
    [(None, {}), ("train_gpu", {"train_gpu": 1})],
)
def test_training_actor_requests_configured_resource(monkeypatch, train_gpu_resource, expected):
    import ray

    from tuft.backends.hf_training_model import HFTrainingModel

    recorder = _RecordingRemote()
    monkeypatch.setattr(ray, "remote", recorder)

    HFTrainingModel.get_actor(_config(train_gpu_resource=train_gpu_resource))

    assert recorder.options_kwargs["resources"] == expected


def test_sampling_actor_requests_configured_resource_per_gpu(monkeypatch):
    import ray

    from tuft.backends.sampling_backend import VLLMSamplingBackend

    recorder = _RecordingRemote()
    monkeypatch.setattr(ray, "remote", recorder)
    backend = VLLMSamplingBackend.__new__(VLLMSamplingBackend)
    backend.base_model = "test"
    backend._worker_venv_path = None
    backend._instance_index = 0

    backend._create_standalone_engine(
        _config(infer_gpu_resource="infer_gpu", tensor_parallel_size=2)
    )

    assert recorder.options_kwargs["resources"] == {"infer_gpu": 2}
