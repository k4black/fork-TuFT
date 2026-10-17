"""Image chunks: split for vLLM, decoded from the SDK proto, checked after generation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from tinker import types
from tinker.proto.request_conv import forward_backward_request_to_proto

from tuft.backends.image_prompt import split_prompt
from tuft.backends.vllm_engine import IMAGE_DECODE_ERROR
from tuft.compat import decode_forward_backward_request
from tuft.exceptions import InvalidRequestException

from .test_adapter_staging import _backend, _remote


IMAGE = types.ImageChunk(data=b"\x89PNG fake", format="png", expected_tokens=4)
PROMPT = types.ModelInput(
    chunks=[types.EncodedTextChunk(tokens=[1, 2]), IMAGE, types.EncodedTextChunk(tokens=[3])]
)


def test_text_only_matches_to_ints() -> None:
    prompt = types.ModelInput.from_ints([5, 6, 7])
    assert split_prompt(prompt) == ([5, 6, 7], [], 3)


def test_one_image_is_one_placeholder_and_expected_tokens_long() -> None:
    assert split_prompt(PROMPT) == ([1, 2, None, 3], [IMAGE.data], 7)


def test_image_without_expected_tokens_is_rejected() -> None:
    prompt = types.ModelInput(chunks=[types.ImageChunk(data=b"x", format="png")])
    with pytest.raises(InvalidRequestException, match="expected_tokens"):
        split_prompt(prompt)


def test_image_chunk_survives_proto_decoding() -> None:
    request = types.ForwardBackwardRequest(
        forward_backward_input=types.ForwardBackwardInput(
            data=[types.Datum(model_input=PROMPT, loss_fn_inputs={})],
            loss_fn="cross_entropy",
        ),
        model_id="model-1",
        seq_id=1,
    )
    body = forward_backward_request_to_proto(request).SerializeToString()
    decoded = decode_forward_backward_request(body).data[0]
    assert split_prompt(decoded.model_input) == split_prompt(PROMPT)


@pytest.mark.parametrize("engine_prompt_len, ok", [(7, True), (9, False)])
async def test_vllm_sends_one_image_token_and_checks_the_expansion(
    monkeypatch, engine_prompt_len, ok
) -> None:
    sent: list[dict] = []

    def _generate(prompt, **_kwargs):
        sent.append(prompt)
        logprobs = [{7: SimpleNamespace(logprob=-0.1)}]
        return SimpleNamespace(
            prompt_token_ids=[0] * engine_prompt_len,
            outputs=[SimpleNamespace(finish_reason="stop", token_ids=[7], logprobs=logprobs)],
        )

    backend = _backend(monkeypatch, [])
    backend.engine.image_token_id = _remote(lambda: 99)  # type: ignore[attr-defined]
    backend.engine.generate = _remote(_generate)  # type: ignore[attr-defined]
    backend._image_token_id = None
    sample = backend.sample(
        prompt=PROMPT, num_samples=1, sampling_params=types.SamplingParams(max_tokens=1)
    )
    if ok:
        assert (await sample).sequences[0].tokens == [7]
    else:
        with pytest.raises(InvalidRequestException, match="expands to 9 tokens, expected 7"):
            await sample
    assert sent == [{"prompt_token_ids": [1, 2, 99, 3], "images": [IMAGE.data]}]


async def test_undecodable_image_is_a_bad_request(monkeypatch) -> None:
    def _generate(**_kwargs):
        raise ValueError(IMAGE_DECODE_ERROR)

    backend = _backend(monkeypatch, [])
    backend.engine.generate = _remote(_generate)  # type: ignore[attr-defined]
    backend._image_token_id = 99
    with pytest.raises(InvalidRequestException, match="not a decodable image"):
        await backend.sample(
            prompt=PROMPT, num_samples=1, sampling_params=types.SamplingParams(max_tokens=1)
        )
