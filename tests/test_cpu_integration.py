"""Real-server SDK wiring test on CPU: HF training backend, vLLM CPU wheel, tiny Qwen3."""

from __future__ import annotations

import math
import os
from pathlib import Path

import httpx
import pytest
import tinker.types as types
from tinker.lib.public_interfaces.service_client import ServiceClient
from transformers import AutoTokenizer

from tuft.config import ModelConfig

from .helpers import ServerFixtureConfig, _create_server_endpoint, _create_training_data


pytestmark = pytest.mark.cpu_integration

TIMEOUT = 600


@pytest.fixture(scope="module", params=[False, True, "sleep"], ids=lambda m: f"colocate={m}")
def cpu_server_endpoint(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    model_path = Path(os.environ["TUFT_TINY_MODEL"])
    config = ServerFixtureConfig(
        model_configs=[
            ModelConfig(
                model_name="Qwen/Qwen3-0.6B",
                model_path=model_path,
                max_model_len=1024,
                max_lora_rank=8,
                max_loras=1,
                sampling_enforce_eager=True,
                colocate=request.param,
            )
        ]
    )
    yield from _create_server_endpoint(tmp_path_factory, config)


def _rl_data(tok, advantage: float) -> list[types.Datum]:
    data = _create_training_data(tok)
    for datum in data:
        n = datum.model_input.length
        datum.loss_fn_inputs["logprobs"] = types.TensorData(
            data=[0.0] * n, dtype="float32", shape=[n]
        )
        datum.loss_fn_inputs["advantages"] = types.TensorData(
            data=[advantage] * n, dtype="float32", shape=[n]
        )
    return data


def test_sdk_flow(cpu_server_endpoint: str) -> None:
    service = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        project_id="tuft",  # tinker 0.34+ copy_weights needs a project; the server ignores it
        base_url=cpu_server_endpoint,
        timeout=TIMEOUT,
    )
    tok = AutoTokenizer.from_pretrained(os.environ["TUFT_TINY_MODEL"])
    try:
        trainer = service.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=8)
        fb = trainer.forward_backward(_create_training_data(tok), "cross_entropy").result(
            timeout=TIMEOUT
        )
        assert all(math.isfinite(v) for v in fb.metrics.values())
        trainer.optim_step(types.AdamParams(learning_rate=1e-3)).result(timeout=TIMEOUT)

        # Importance sampling through the real HF path: negating the advantages
        # negates the loss exactly, so both the ratio and the advantages reach it.
        pos = trainer.forward_backward(_rl_data(tok, 1.0), "importance_sampling").result(
            timeout=TIMEOUT
        )
        neg = trainer.forward_backward(_rl_data(tok, -1.0), "importance_sampling").result(
            timeout=TIMEOUT
        )
        assert math.isfinite(pos.metrics["loss:sum"]) and pos.metrics["loss:sum"] != 0
        assert pos.metrics["loss:sum"] == pytest.approx(-neg.metrics["loss:sum"], rel=1e-3)
        trainer.optim_step(types.AdamParams(learning_rate=1e-3)).result(timeout=TIMEOUT)

        sampler = trainer.save_weights_and_get_sampling_client()
        prompt = types.ModelInput.from_ints(tok.encode("Hello there, how are"))

        def sample(temperature: float) -> types.SampleResponse:
            return sampler.sample(
                prompt,
                1,
                types.SamplingParams(max_tokens=4, temperature=temperature, seed=0),
                include_prompt_logprobs=True,
            ).result(timeout=TIMEOUT)

        res = sample(1.0)
        seq = res.sequences[0]
        assert len(seq.tokens) == 4
        assert seq.logprobs is not None and all(lp <= 0 for lp in seq.logprobs)
        assert res.prompt_logprobs is not None
        assert res.prompt_logprobs[0] is None
        assert len(res.prompt_logprobs) == prompt.length
        # The TuFTCPUWorker patch scales prompt logprobs by temperature.
        assert sample(0.5).prompt_logprobs != res.prompt_logprobs

        # A second adapter with max_loras=1 makes vLLM evict the first and re-add it.
        other = service.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=8)
        other.forward_backward(_create_training_data(tok), "cross_entropy").result(timeout=TIMEOUT)
        other.optim_step(types.AdamParams(learning_rate=0.1)).result(timeout=TIMEOUT)
        other_res = (
            other.save_weights_and_get_sampling_client()
            .sample(
                prompt,
                1,
                types.SamplingParams(max_tokens=4, temperature=1.0, seed=0),
                include_prompt_logprobs=True,
            )
            .result(timeout=TIMEOUT)
        )
        again = sample(1.0).prompt_logprobs
        assert other_res.prompt_logprobs is not None and again is not None
        assert other_res.prompt_logprobs[1:] != res.prompt_logprobs[1:]
        assert again[1:] == pytest.approx(res.prompt_logprobs[1:], abs=1e-4)

        topk = sampler.sample(
            prompt, 1, types.SamplingParams(max_tokens=4), topk_sample_logprobs=3
        ).result(timeout=TIMEOUT)
        rows = topk.sequences[0].topk_logprobs
        assert rows is not None and len(rows) == 4
        for row in rows:
            assert row is not None and len(row) == 3
            assert [lp for _, lp in row] == sorted((lp for _, lp in row), reverse=True)
        assert trainer.get_info().model_data.arch == "qwen3"

        def greedy(stop: list[str] | list[int] | None = None) -> types.SampledSequence:
            params = types.SamplingParams(max_tokens=8, temperature=0.0, stop=stop)
            return sampler.sample(prompt, 1, params).result(timeout=TIMEOUT).sequences[0]

        full = greedy().tokens
        by_id = greedy(stop=[full[2]])
        assert by_id.stop_reason == "stop"
        assert by_id.tokens == full[: full.index(full[2]) + 1]
        by_str = greedy(stop=[tok.decode(full[2:3])])
        assert by_str.stop_reason == "stop" and len(by_str.tokens) <= 3

        state_path = trainer.save_state("cpu-state").result(timeout=TIMEOUT).path
        restored = service.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=8)
        restored.load_state(state_path).result(timeout=TIMEOUT)
        sampler2 = restored.save_weights_and_get_sampling_client()
        out = sampler2.sample(prompt, 1, types.SamplingParams(max_tokens=4)).result(timeout=TIMEOUT)
        assert len(out.sequences[0].tokens) == 4

        sampler_path = trainer.save_weights_for_sampler("cpu-sampler").result(timeout=TIMEOUT).path
        copied = service.copy_weights(sampler_path).result(timeout=TIMEOUT)
        assert copied.startswith("tinker://") and copied != sampler_path
        out = (
            service.create_sampling_client(model_path=copied)
            .sample(prompt, 1, types.SamplingParams(max_tokens=4))
            .result(timeout=TIMEOUT)
        )
        assert len(out.sequences[0].tokens) == 4
    finally:
        service.holder.close()


def test_trained_checkpoint_serves_openai_chat(cpu_server_endpoint: str) -> None:
    service = ServiceClient(
        api_key="tml-test-key",  # pragma: allowlist secret
        base_url=cpu_server_endpoint,
        timeout=TIMEOUT,
    )
    tok = AutoTokenizer.from_pretrained(os.environ["TUFT_TINY_MODEL"])
    try:
        trainer = service.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=8)
        trainer.forward_backward(_create_training_data(tok), "cross_entropy").result(
            timeout=TIMEOUT
        )
        trainer.optim_step(types.AdamParams(learning_rate=1e-3)).result(timeout=TIMEOUT)
        path = trainer.save_weights_for_sampler("cpu-oai").result(timeout=TIMEOUT).path
    finally:
        service.holder.close()

    def chat(api_key: str) -> httpx.Response:
        return httpx.post(
            f"{cpu_server_endpoint}/oai/api/v1/chat/completions",
            json={
                "model": path,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4,
                "temperature": 0,
                "ignore_eos": True,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT,
        )

    resp = chat("tml-test-key")  # pragma: allowlist secret
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == path
    assert len(body["choices"]) == 1
    assert body["usage"]["completion_tokens"] == 4
    # Checkpoints are private by default.
    assert chat("tml-test-key-2").status_code == 403  # pragma: allowlist secret
