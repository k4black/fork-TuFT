"""Fetch and check LoRA adapters that copy_weights imports from hf:// and s3://."""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from .checkpoints import read_adapter_target_modules
from .exceptions import InvalidRequestException


# Pickled .bin weights run code on load, so only safetensors adapters are fetched.
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
_LORA_KEY = re.compile(r"base_model\.model\.(.+)\.lora_([AB])\.weight")


def s3_allowed(bucket: str, key: str, prefixes: list[str]) -> bool:
    """True when ``bucket/key`` lies under one of ``prefixes`` on a ``/`` boundary."""
    if ".." in key.split("/"):
        return False
    for prefix in prefixes:
        prefix_bucket, _, prefix_key = prefix.removeprefix("s3://").partition("/")
        prefix_key = prefix_key.rstrip("/") + "/" if prefix_key else ""
        if bucket == prefix_bucket and (key.rstrip("/") + "/").startswith(prefix_key):
            return True
    return False


def fetch(source: str, token: str | None, adapter_dir: Path, s3_prefixes: list[str]) -> None:
    """Download the two adapter files of ``source`` into ``adapter_dir``."""
    scheme, _, rest = source.partition("://")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    try:
        if scheme == "hf":
            _fetch_hf(rest, token, adapter_dir)
        elif scheme == "s3":
            _fetch_s3(rest, adapter_dir, s3_prefixes)
        else:
            raise InvalidRequestException(
                f"Unsupported source {source}: use tinker://, hf:// or s3://."
            )
    except InvalidRequestException:
        raise
    except Exception as exc:
        raise InvalidRequestException(f"Cannot download {source}: {exc}") from exc
    if not all((adapter_dir / name).is_file() for name in ADAPTER_FILES):
        raise InvalidRequestException(
            f"{source} must hold adapter_config.json and adapter_model.safetensors; "
            "only safetensors adapters are supported."
        )


def _fetch_hf(rest: str, token: str | None, adapter_dir: Path) -> None:
    """``org/repo[@rev][/subdir]``; only the request token is used, never the server's."""
    from huggingface_hub import snapshot_download

    parts = [part for part in rest.split("/") if part]
    if any(part in (".", "..") or "\\" in part for part in parts):
        raise InvalidRequestException(f"hf://{rest} must not hold . or .. segments.")
    repo, _, revision = "/".join(parts[:2]).partition("@")
    subdir = "".join(f"{part}/" for part in parts[2:])
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo,
            revision=revision or None,
            allow_patterns=[subdir + name for name in ADAPTER_FILES],
            token=token or False,
            local_dir=tmp,
        )
        for name in ADAPTER_FILES:
            source = (Path(tmp) / subdir / name).resolve()
            if source.is_relative_to(Path(tmp).resolve()) and source.is_file():
                shutil.copyfile(source, adapter_dir / name)


def _fetch_s3(rest: str, adapter_dir: Path, s3_prefixes: list[str]) -> None:
    """``bucket/key/`` under a configured prefix, with the default boto3 credentials."""
    bucket, _, key = rest.partition("/")
    if not s3_allowed(bucket, key, s3_prefixes):
        raise InvalidRequestException(
            f"s3://{rest} is not under a configured import_s3_prefixes entry."
        )
    import boto3

    client = boto3.client("s3")
    key = key.rstrip("/") + "/" if key else ""
    for name in ADAPTER_FILES:
        client.download_file(bucket, key + name, str(adapter_dir / name))


def check_shapes(adapter_dir: Path, model_path: Path, rank: int) -> None:
    """Reject LoRA tensors whose shapes do not fit the base model's modules."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path))
    try:
        handle = safe_open(adapter_dir / "adapter_model.safetensors", "pt")
    except Exception as exc:
        raise InvalidRequestException(f"Unreadable adapter_model.safetensors: {exc}") from exc
    pairs: dict[str, set[str]] = {}
    with handle as weights:
        for key in weights.keys():
            match = _LORA_KEY.fullmatch(key)
            try:
                weight = model.get_submodule(match[1]).weight if match else None
            except AttributeError:
                weight = None
            if match is None or weight is None or weight.dim() != 2:
                raise InvalidRequestException(
                    f"Adapter tensor {key} matches no linear module of {model_path}."
                )
            out_features, in_features = weight.shape
            expected = (rank, in_features) if match[2] == "A" else (out_features, rank)
            shape = tuple(weights.get_slice(key).get_shape())
            if shape != expected:
                raise InvalidRequestException(
                    f"Adapter tensor {key} has shape {shape}; the base model needs {expected}."
                )
            pairs.setdefault(match[1], set()).add(match[2])
    targets = read_adapter_target_modules(adapter_dir) or []
    # PEFT targets every Linear whose name equals a target or ends with ".<target>".
    expected = {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and any(name == t or name.endswith(f".{t}") for t in targets)
    }
    if not expected or any(pairs.get(name) != {"A", "B"} for name in expected | set(pairs)):
        raise InvalidRequestException(
            "adapter_model.safetensors must hold a lora_A and lora_B tensor for every "
            "module adapter_config.json targets."
        )
