"""Split a tinker ``ModelInput`` with image chunks into vLLM prompt parts."""

from __future__ import annotations

from tinker import types

from ..exceptions import InvalidRequestException


def split_prompt(prompt: types.ModelInput) -> tuple[list[int | None], list[bytes], int]:
    """Return token ids with one ``None`` per image, the image bytes and the SDK length.

    Text-only input gives ``to_ints()``, ``[]`` and ``len(to_ints())``. The length
    counts ``expected_tokens`` per image: the prompt size after vLLM expands each
    image token into its placeholders.
    """
    tokens: list[int | None] = []
    images: list[bytes] = []
    length = 0
    for chunk in prompt.chunks:
        if isinstance(chunk, types.EncodedTextChunk):
            tokens.extend(chunk.tokens)
            length += len(chunk.tokens)
        elif isinstance(chunk, types.ImageChunk):
            if chunk.expected_tokens is None:
                raise InvalidRequestException("ImageChunk.expected_tokens is required")
            tokens.append(None)
            images.append(chunk.data)
            length += chunk.expected_tokens
        else:
            raise InvalidRequestException(f"{chunk.type} chunks are not supported")
    return tokens, images, length
