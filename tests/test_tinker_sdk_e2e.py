"""End-to-end qualification of TuFT against the stock tinker SDK (D2).

These tests drive a CPU-only TuFT server (``DummyTrainingBackend`` +
``DummySamplingBackend``, no GPU) with the real ``tinker`` client and assert both
wire formats interoperate:

- JSON for the pydantic endpoints (capabilities, create_model, optim_step,
  save_weights, create_sampling_session, ...);
- protobuf for ``/forward_backward`` requests and for ``/retrieve_future``
  responses of the two dataclass payloads (``ForwardBackwardOutput`` /
  ``SampleResponse``), which the SDK only decodes as protobuf.

The CI ``tinker-compat`` matrix runs this file against 0.25.0 and 0.28.1, so the
version-adaptive compat shims (sample sequence identity, ``loss_fn_config_v2``,
the renamed top-k field) are exercised on both wire generations from one server.

Complements ``tests/test_tinker_compat.py``, which unit-tests the wire codec at
the ASGI layer; here everything goes over a real socket via the SDK transport.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from pathlib import Path
from typing import Generator

import h2  # noqa: F401  # tinker opens HTTP/2 clients; h2 arrives via httpx[http2]
import httpx
import pytest
import ray
import uvicorn
from tinker import types
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker.proto.request_conv import forward_backward_request_to_proto
from tinker.proto.response_conv import deserialize_forward_backward_output

from tuft.config import AppConfig, ModelConfig
from tuft.server import create_root_app

from .helpers import CPU_TEST_TIMEOUT, _find_free_port, clear_ray_state


API_KEY = "tml-test-key-1"  # pragma: allowlist secret
BASE_MODEL = "Qwen/Qwen3-0.6B"
_PROTOBUF = "application/x-protobuf"

_SAMPLED_SEQUENCE_HAS_SEQUENCE_ID = any(
    field.name == "sequence_id" for field in dataclasses.fields(types.SampledSequence)
)


@pytest.fixture(scope="module")
def sdk_server(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Start a CPU-only TuFT server and yield its base URL."""
    saved_api_key = os.environ.pop("TINKER_API_KEY", None)
    ray.init(ignore_reinit_error=True)

    checkpoint_dir = tmp_path_factory.mktemp("checkpoints_sdk_e2e")
    config = AppConfig(checkpoint_dir=Path(checkpoint_dir))
    config.supported_models = [
        ModelConfig(
            model_name=BASE_MODEL,
            model_path=Path("/dummy/qwen-model"),
            max_model_len=4096,
            tensor_parallel_size=1,
            sampling_memory_fraction=0.5,
        )
    ]
    config.authorized_users = {API_KEY: "user-alpha"}

    app = create_root_app(config)
    port = _find_free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    with httpx.Client() as probe:
        for _ in range(120):
            try:
                if probe.get(f"{base_url}/api/v1/healthz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(2)
        else:
            server.should_exit = True
            thread.join(timeout=5)
            raise RuntimeError("Server failed to start")

    yield base_url

    server.should_exit = True
    thread.join(timeout=30)
    clear_ray_state()
    if saved_api_key is not None:
        os.environ["TINKER_API_KEY"] = saved_api_key


def _service_client(base_url: str) -> ServiceClient:
    return ServiceClient(api_key=API_KEY, base_url=base_url, timeout=CPU_TEST_TIMEOUT)


def _datum(input_tokens: list[int], target_tokens: list[int]) -> types.Datum:
    weights = [1.0] * len(target_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=target_tokens, dtype="int64", shape=[len(target_tokens)]
            ),
            "weights": types.TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


@pytest.mark.integration
def test_full_training_and_sampling_sequence(sdk_server: str) -> None:
    """create adapter -> forward_backward -> optim_step -> save -> sampler -> sample.

    The SDK returns real ``ForwardBackwardOutput`` / ``SampleResponse`` dataclasses
    only when the protobuf response wire works: on JSON it raises for these types.
    So a green run is itself proof that both wire formats interoperate end to end.
    """
    service_client = _service_client(sdk_server)
    try:
        capabilities = service_client.get_server_capabilities()
        assert capabilities.supported_models
        base_model = capabilities.supported_models[0].model_name or BASE_MODEL

        training_client = service_client.create_lora_training_client(
            base_model=base_model, rank=8, train_unembed=False
        )

        datum = _datum([11, 12, 13, 14], [12, 13, 14, 15])
        fb_output = training_client.forward_backward([datum], "cross_entropy").result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert isinstance(fb_output, types.ForwardBackwardOutput)
        assert fb_output.metrics["loss:sum"] >= 0
        assert fb_output.loss_fn_outputs[0]["logprobs"].data

        optim_result = training_client.optim_step(types.AdamParams(learning_rate=1e-3)).result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert optim_result is not None

        save_result = training_client.save_state("ckpt-e2e").result(timeout=CPU_TEST_TIMEOUT)
        sampler_result = training_client.save_weights_for_sampler("sampler-e2e").result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert save_result.path.startswith("tinker://")
        assert sampler_result.path.startswith("tinker://")

        sampling_client = service_client.create_sampling_client(model_path=sampler_result.path)
        sample_result = sampling_client.sample(
            prompt=types.ModelInput.from_ints([99, 5, 12]),
            num_samples=2,
            sampling_params=types.SamplingParams(max_tokens=5, temperature=0.5),
        ).result(timeout=CPU_TEST_TIMEOUT)

        assert isinstance(sample_result, types.SampleResponse)
        assert len(sample_result.sequences) == 2
        assert all(seq.tokens for seq in sample_result.sequences)
    finally:
        service_client.holder.close()


@pytest.mark.integration
@pytest.mark.skipif(
    not _SAMPLED_SEQUENCE_HAS_SEQUENCE_ID,
    reason="tinker < 0.26.2 has no SampledSequence.sequence_id",
)
def test_sample_sequences_carry_server_minted_ids(sdk_server: str) -> None:
    """tinker >= 0.26.2 asserts one sample sequence id per sample (D2 break B2).

    The client stamps the ids the server returned on the promise, so the returned
    sequence identities must be present, unique, and aligned with the sample index.
    """
    service_client = _service_client(sdk_server)
    try:
        sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)
        num_samples = 3
        result = sampling_client.sample(
            prompt=types.ModelInput.from_ints([7, 8, 9]),
            num_samples=num_samples,
            sampling_params=types.SamplingParams(max_tokens=4),
        ).result(timeout=CPU_TEST_TIMEOUT)

        ids = [seq.sequence_id for seq in result.sequences]
        assert len(ids) == num_samples
        assert all(ids), "every sampled sequence must carry an id"
        assert len(set(ids)) == num_samples, "sequence ids must be distinct"
        # Server mints f"{session}:{seq}:{index}"; the client attaches them in order.
        assert all(sid.endswith(f":{index}") for index, sid in enumerate(ids))
    finally:
        service_client.holder.close()


@pytest.mark.integration
def test_topk_prompt_logprobs_round_trip(sdk_server: str) -> None:
    """Top-k prompt logprobs survive the protobuf response (D2 break B1).

    tinker 0.28.0 renamed the top-k message field; a mismatched writer would raise
    server-side and this call would fail rather than return structured top-k rows.
    """
    service_client = _service_client(sdk_server)
    try:
        sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)
        topk = 2
        result = sampling_client.sample(
            prompt=types.ModelInput.from_ints([21, 22, 23]),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=3),
            include_prompt_logprobs=True,
            topk_prompt_logprobs=topk,
        ).result(timeout=CPU_TEST_TIMEOUT)

        assert result.prompt_logprobs is not None
        assert result.topk_prompt_logprobs is not None
        populated = [row for row in result.topk_prompt_logprobs if row]
        assert populated, "expected at least one populated top-k row"
        for row in populated:
            assert len(row) <= topk
            token_id, logprob = row[0]
            assert isinstance(token_id, int)
            assert isinstance(logprob, float)
    finally:
        service_client.holder.close()


@pytest.mark.integration
def test_pipelined_futures_resolve_out_of_order(sdk_server: str) -> None:
    """Several in-flight training futures resolve correctly when retrieved reordered."""
    service_client = _service_client(sdk_server)
    try:
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=8, train_unembed=False
        )
        # Submit three forward_backward requests before awaiting any (pipelining).
        futures = [
            training_client.forward_backward(
                [_datum([10 + i, 11 + i, 12 + i], [11 + i, 12 + i, 13 + i])], "cross_entropy"
            )
            for i in range(3)
        ]
        # Retrieve in reverse submission order; each future keeps its own payload.
        for future in reversed(futures):
            output = future.result(timeout=CPU_TEST_TIMEOUT)
            assert isinstance(output, types.ForwardBackwardOutput)
            assert output.metrics["loss:sum"] >= 0
    finally:
        service_client.holder.close()


def _post_proto_fwdbwd(
    client: httpx.Client, base_url: str, model_id: str, seq_id: int
) -> httpx.Response:
    """Submit a /forward_backward proto body built by the SDK's own encoder."""
    request = types.ForwardBackwardRequest(
        forward_backward_input=types.ForwardBackwardInput(
            data=[_datum([31, 32, 33], [32, 33, 34])],
            loss_fn="cross_entropy",
        ),
        model_id=model_id,
        seq_id=seq_id,
    )
    body = forward_backward_request_to_proto(request).SerializeToString()
    return client.post(
        f"{base_url}/api/v1/forward_backward",
        content=body,
        headers={"X-API-Key": API_KEY, "Content-Type": _PROTOBUF},
    )


@pytest.mark.integration
def test_proto_and_json_retrieve_wire_formats(sdk_server: str) -> None:
    """The same payload goes back as protobuf under an Accept: protobuf, JSON otherwise.

    Uses a fresh training client (no SDK ops, so the server expects seq_id=1) and a
    raw client so the wire headers are directly observable.
    """
    service_client = _service_client(sdk_server)
    try:
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=8, train_unembed=False
        )
        model_id = training_client.model_id
        assert model_id is not None

        with httpx.Client() as raw:
            caps = raw.get(
                f"{sdk_server}/api/v1/get_server_capabilities", headers={"X-API-Key": API_KEY}
            )
            assert caps.status_code == 200
            assert caps.headers["content-type"].startswith("application/json")

            submit = _post_proto_fwdbwd(raw, sdk_server, model_id, seq_id=1)
            assert submit.status_code == 202
            request_id = submit.json()["request_id"]

            proto_resp = raw.post(
                f"{sdk_server}/api/v1/retrieve_future",
                json={"request_id": request_id},
                headers={"X-API-Key": API_KEY, "Accept": _PROTOBUF},
            )
            assert proto_resp.status_code == 200
            assert proto_resp.headers["content-type"].startswith(_PROTOBUF)
            decoded = deserialize_forward_backward_output(proto_resp.content)
            assert decoded.loss_fn_outputs[0]["logprobs"].data

            json_resp = raw.post(
                f"{sdk_server}/api/v1/retrieve_future",
                json={"request_id": request_id},
                headers={"X-API-Key": API_KEY, "Accept": "application/json"},
            )
            assert json_resp.status_code == 200
            assert json_resp.headers["content-type"].startswith("application/json")
            assert json_resp.json()["loss_fn_outputs"]
    finally:
        service_client.holder.close()


@pytest.mark.integration
def test_duplicate_and_gapped_seq_ids_are_rejected(sdk_server: str) -> None:
    """The server enforces strict ascending seq_id per training run.

    A fresh model expects seq_id=1; a replayed or gapped seq_id must fail with a
    sequence conflict rather than silently applying an out-of-order optimizer step.
    """
    service_client = _service_client(sdk_server)
    try:
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=8, train_unembed=False
        )
        model_id = training_client.model_id
        assert model_id is not None

        with httpx.Client() as raw:
            first = _post_proto_fwdbwd(raw, sdk_server, model_id, seq_id=1)
            assert first.status_code == 202
            ok = raw.post(
                f"{sdk_server}/api/v1/retrieve_future",
                json={"request_id": first.json()["request_id"]},
                headers={"X-API-Key": API_KEY, "Accept": _PROTOBUF},
            )
            assert ok.status_code == 200  # consumes seq_id 1; server now expects 2

            for bad_seq_id in (1, 5):  # replayed, then gapped
                submit = _post_proto_fwdbwd(raw, sdk_server, model_id, seq_id=bad_seq_id)
                assert submit.status_code == 202
                conflict = raw.post(
                    f"{sdk_server}/api/v1/retrieve_future",
                    json={"request_id": submit.json()["request_id"]},
                    headers={"X-API-Key": API_KEY, "Accept": _PROTOBUF},
                )
                assert conflict.status_code == 409
                assert "Sequence conflict" in conflict.json()["detail"]
    finally:
        service_client.holder.close()


@pytest.mark.integration
def test_terminal_backend_failure_surfaces_without_hanging(sdk_server: str) -> None:
    """A terminal backend error surfaces to the SDK (no hang), leaving the server usable.

    ``DummyTrainingBackend`` raises when ``loss_fn_config`` carries the
    ``raise_missing_input`` sentinel. The SDK must raise a real error within the
    timeout rather than block (a hang would surface as ``TimeoutError``, which we
    reject). A fresh client then trains successfully, proving the process is live.
    """
    service_client = _service_client(sdk_server)
    try:
        training_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=8, train_unembed=False
        )
        datum = _datum([40, 41, 42], [41, 42, 43])

        with pytest.raises(Exception) as excinfo:
            training_client.forward_backward(
                [datum], "cross_entropy", loss_fn_config={"raise_missing_input": 1.0}
            ).result(timeout=CPU_TEST_TIMEOUT)
        assert not isinstance(excinfo.value, TimeoutError), "error must surface, not hang"

        healthy_client = service_client.create_lora_training_client(
            base_model=BASE_MODEL, rank=8, train_unembed=False
        )
        recovered = healthy_client.forward_backward([datum], "cross_entropy").result(
            timeout=CPU_TEST_TIMEOUT
        )
        assert isinstance(recovered, types.ForwardBackwardOutput)
    finally:
        service_client.holder.close()
