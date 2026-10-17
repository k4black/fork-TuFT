import asyncio
import logging
from types import SimpleNamespace

import pytest
from tinker import types

from tuft.checkpoints import CheckpointRecord


@pytest.mark.asyncio
async def test_forward_empty_request_is_a_noop():
    from tuft.backends.hf_training_model import HFTrainingModel

    model = HFTrainingModel.__new__(HFTrainingModel)

    output = await model.forward(
        data=[],
        lora_id="test",
        loss_fn="cross_entropy",
        loss_fn_config=None,
    )

    assert output.loss_fn_output_type == "cross_entropy"
    assert output.loss_fn_outputs == []
    assert output.metrics == {}


@pytest.mark.asyncio
async def test_forward_backward_empties_cuda_cache_once_after_all_micro_batches(
    monkeypatch,
):
    import torch

    from tuft.backends.hf_training_model import HFTrainingModel

    class TinyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 8)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)

        def get_decoder(self):
            return lambda input_ids, **_: SimpleNamespace(last_hidden_state=self.embed(input_ids))

        def get_output_embeddings(self):
            return self.lm_head

    data = [
        types.Datum(
            model_input=types.ModelInput.from_ints(tokens=tokens),
            loss_fn_inputs={
                "target_tokens": types.TensorData(
                    data=[token + 1 for token in tokens], dtype="int64"
                ),
                "weights": types.TensorData(data=[1.0] * len(tokens), dtype="float32"),
            },
        )
        for tokens in ([1, 2, 3], [4, 5])
    ]

    model = HFTrainingModel.__new__(HFTrainingModel)
    model.config = SimpleNamespace(micro_batch_size=1)  # type: ignore[assignment]
    model.model = TinyCausalLM()  # type: ignore[assignment]
    model._lock = asyncio.Lock()
    model.logger = logging.getLogger(__name__)
    monkeypatch.setattr(model, "_activate_adapter", lambda _lora_id: None)

    empty_cache_calls = 0

    def record_empty_cache():
        nonlocal empty_cache_calls
        empty_cache_calls += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", record_empty_cache)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    output = await model.forward(
        data=data,
        lora_id="test",
        loss_fn="cross_entropy",
        loss_fn_config=None,
        backward=True,
    )

    assert len(output.loss_fn_outputs) == 2
    assert empty_cache_calls == 1


def _checkpoint_record(tmp_path, training_run_id: str) -> CheckpointRecord:
    return CheckpointRecord.from_training_run(
        training_run_id=training_run_id,
        checkpoint_name="ckpt",
        owner_name="tester",
        checkpoint_type="training",
        checkpoint_root_dir=tmp_path,
    )


@pytest.mark.parametrize("legacy_layout", [False, True])
def test_optimizer_state_resolves_across_runs(tmp_path, legacy_layout) -> None:
    """The optimizer file must be findable by a run other than the one that saved it.

    Naming it after the saving run made optimizer=True a silent no-op whenever a
    checkpoint was restored into a different training run.
    """
    from tuft.backends.hf_training_model import (
        OPTIMIZER_STATE_FILENAME,
        _resolve_optimizer_state_path,
    )

    record = _checkpoint_record(tmp_path, "source-run")
    record.optimizer_path.mkdir(parents=True, exist_ok=True)
    file_name = "source-run.pt" if legacy_layout else OPTIMIZER_STATE_FILENAME
    (record.optimizer_path / file_name).write_bytes(b"adam-state")

    # Resolution depends only on the checkpoint, never on the run loading it.
    resolved = _resolve_optimizer_state_path(record)

    assert resolved is not None
    assert resolved.name == file_name


def test_optimizer_state_absent_resolves_to_none(tmp_path) -> None:
    from tuft.backends.hf_training_model import _resolve_optimizer_state_path

    record = _checkpoint_record(tmp_path, "source-run")
    record.optimizer_path.mkdir(parents=True, exist_ok=True)

    assert _resolve_optimizer_state_path(record) is None


@pytest.mark.asyncio
async def test_load_state_restores_optimizer_into_a_different_run(tmp_path) -> None:
    import torch

    from tuft.backends.hf_training_model import OPTIMIZER_STATE_FILENAME, HFTrainingModel

    class FakePeftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.loaded_adapters: list[str] = []

        def load_adapter(self, model_id, adapter_name):
            self.loaded_adapters.append(adapter_name)

        def set_adapter(self, adapter_name):
            pass

    model = HFTrainingModel.__new__(HFTrainingModel)
    model.model = FakePeftModel()  # type: ignore[assignment]
    model.adapter_optimizer = {}
    model._lock = asyncio.Lock()
    model.logger = logging.getLogger(__name__)

    # Source run saves optimizer state with a non-zero step count.
    record = _checkpoint_record(tmp_path, "source-run")
    record.optimizer_path.mkdir(parents=True, exist_ok=True)
    saved = torch.optim.AdamW(list(model.model.parameters()))
    model.model.weight.grad = torch.ones(4)
    saved.step()
    torch.save(saved.state_dict(), record.optimizer_path / OPTIMIZER_STATE_FILENAME)

    await model.load_state(lora_id="destination-run", checkpoint_record=record, optimizer=True)

    restored = model.adapter_optimizer["destination-run"]
    assert model.model.loaded_adapters == ["destination-run"]
    # A fresh AdamW has an empty state dict; the restored one carries the step.
    assert restored.state_dict()["state"], "optimizer state was silently dropped"
    assert restored.state_dict()["state"][0]["step"] == saved.state_dict()["state"][0]["step"]


@pytest.mark.asyncio
async def test_create_adapter_rejects_rank_above_max_lora_rank():
    from tuft.backends.training_backend import HFTrainingBackend
    from tuft.exceptions import InvalidRequestException

    backend = HFTrainingBackend.__new__(HFTrainingBackend)
    backend.config = SimpleNamespace(max_lora_rank=16)  # type: ignore[assignment]

    with pytest.raises(InvalidRequestException) as excinfo:
        await backend.create_adapter("run", types.LoraConfig(rank=32))
    assert excinfo.value.status_code == 400
    assert "raise supported_models[].max_lora_rank" in str(excinfo.value)


@pytest.mark.asyncio
async def test_hf_backend_validates_inputs_before_the_actor(tmp_path):
    from transformers import Qwen3_5TextConfig

    from tuft.backends.training_backend import HFTrainingBackend

    Qwen3_5TextConfig(vocab_size=128).save_pretrained(tmp_path)
    backend = HFTrainingBackend.__new__(HFTrainingBackend)
    backend.config = SimpleNamespace(max_model_len=4, model_path=tmp_path)  # type: ignore[assignment]
    backend.model = None  # type: ignore[assignment]  # the actor must not be reached

    def datum(tokens: list[int]) -> types.Datum:
        return types.Datum(model_input=types.ModelInput.from_ints(tokens), loss_fn_inputs={})

    # No async_init (colocate=False skips it): the first forward loads the vocabulary.
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        await backend.forward([datum([1, 2, 3, 4, 5])], "run", "cross_entropy", None)
    assert backend._vocab_size == 128
    with pytest.raises(ValueError, match="out of range for vocab_size 128"):
        await backend.forward([datum([1, 2, 128])], "run", "cross_entropy", None)

    backend._vocab_size = None
    backend.config.model_path = tmp_path / "missing"
    with pytest.raises(OSError):
        await backend.forward([datum([1, 2, 3])], "run", "cross_entropy", None)


def _tiny_qwen3_5():
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    return Qwen3_5ForCausalLM(config).float()


def test_activate_adapter_keeps_gradient_checkpointing_on(tmp_path):
    from tuft.backends.hf_training_model import HFTrainingModel

    _tiny_qwen3_5().save_pretrained(tmp_path)
    model = HFTrainingModel.__new__(HFTrainingModel)
    config = SimpleNamespace(model_path=tmp_path, attn_implementation=None)
    model.model = model._init_peft_model(config)  # type: ignore[arg-type]
    model.adapter_optimizer = {"default": None}  # type: ignore[assignment]
    model.model.eval()  # as after load_adapter

    model._activate_adapter("default")

    # HF checkpoints a layer only when both flags are set.
    layers = model.model.get_decoder().layers  # pyright: ignore[reportCallIssue]
    assert all(layer.training and layer.gradient_checkpointing for layer in layers)


async def test_create_adapter_honours_lora_seed(tmp_path):
    import asyncio

    import torch

    from tuft.backends.hf_training_model import HFTrainingModel

    _tiny_qwen3_5().save_pretrained(tmp_path)
    model = HFTrainingModel.__new__(HFTrainingModel)
    model.config = SimpleNamespace(  # type: ignore[assignment]
        model_path=tmp_path,
        attn_implementation=None,
        lora_alpha_ratio=2,
        qwen_gated_deltanet_full_lora=False,
    )
    model.model = model._init_peft_model(model.config)  # type: ignore[arg-type]
    model.adapter_optimizer = {}  # type: ignore[assignment]
    model._lock = asyncio.Lock()
    seeded = types.LoraConfig(rank=4, seed=7, train_unembed=False)
    unseeded = types.LoraConfig(rank=4, train_unembed=False)
    await model.create_adapter("a", seeded)
    torch.manual_seed(0)
    expected_draw = torch.rand(3)
    torch.manual_seed(0)
    await model.create_adapter("b", seeded)
    assert torch.rand(3).equal(expected_draw)  # the seed does not leak into the caller's RNG
    await model.create_adapter("c", unseeded)
    await model.create_adapter("d", unseeded)

    params = dict(model.model.named_parameters())
    lora_a = [name for name in params if ".lora_A.a." in name]
    assert lora_a
    for name in lora_a:
        assert params[name].equal(params[name.replace(".a.", ".b.")])
        assert not params[name.replace(".a.", ".c.")].equal(params[name.replace(".a.", ".d.")])


def test_long_context_path_matches_full_logits(monkeypatch):
    import torch
    from peft import LoraConfig, get_peft_model

    from tuft.backends import hf_training_model

    model = get_peft_model(
        _tiny_qwen3_5(),
        LoraConfig(target_modules=["q_proj", "gate_proj", "lm_head"], init_lora_weights=False),
    )
    ids = torch.randint(0, 128, (2, 7))
    labels = torch.randint(0, 128, (2, 7))
    lora = [p for p in model.parameters() if p.requires_grad]

    logits = model(input_ids=ids).logits / 0.7
    expected = torch.log_softmax(logits, -1).gather(-1, labels[..., None]).squeeze(-1)
    expected_grads = torch.autograd.grad(expected.sum(), lora)

    # Tiled MLPs, gated norms, LoRA linears and lm_head: several chunks, a partial last one.
    monkeypatch.setattr(hf_training_model, "_CHUNK_TOKENS", 3)
    monkeypatch.setattr(hf_training_model, "_TILE_ELEMENTS", 3 * 16)  # 3 tokens of hidden 16
    monkeypatch.setattr(hf_training_model, "_OFFLOAD_MIN_TOKENS", 4)
    hf_training_model._tile_long_inputs(model)
    hidden = model.get_decoder()(input_ids=ids, use_cache=False).last_hidden_state  # pyright: ignore[reportCallIssue]
    actual = hf_training_model._chunked_target_logprobs(
        hidden,
        model.get_output_embeddings(),  # pyright: ignore[reportCallIssue]
        labels,
        0.7,
    )
    actual_grads = torch.autograd.grad(actual.sum(), lora)

    torch.testing.assert_close(actual, expected)
    for a, e in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, e)


def test_inactive_adapter_optimizer_moments_stay_on_host(monkeypatch):
    import torch

    from tuft.backends.hf_training_model import HFTrainingModel

    # The meta device stands in for the GPU; moments get placeholder copies.
    def fake_to(self, key, t, device, used):
        return torch.empty_like(t, device=device) if key.startswith("opt.") else t

    monkeypatch.setattr(HFTrainingModel, "_to", fake_to)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    model = HFTrainingModel.__new__(HFTrainingModel)
    model.model = torch.nn.Linear(2, 2, device="meta")  # type: ignore[assignment]
    model._host = {}
    model.adapter_optimizer = {}
    for lora_id in ("a", "b"):
        param = torch.nn.Parameter(torch.zeros(2, device="meta"))
        optimizer = torch.optim.AdamW([param])
        optimizer.state[param] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.zeros(2, device="meta"),
        }
        model.adapter_optimizer[lora_id] = optimizer

    def devices() -> dict[str, tuple[str, str]]:
        return {
            lora_id: next(
                (s["exp_avg"].device.type, s["step"].device.type) for s in opt.state.values()
            )
            for lora_id, opt in model.adapter_optimizer.items()
        }

    model._swap_opt("a")
    model._swap_opt("b")
    assert devices() == {"a": ("cpu", "cpu"), "b": ("meta", "cpu")}
    model._swap_opt("a")
    assert devices() == {"a": ("meta", "cpu"), "b": ("cpu", "cpu")}

    # colocate "sleep": onload brings back only the active adapter's moments.
    model._move(torch.device("cpu"))
    assert devices() == {"a": ("cpu", "cpu"), "b": ("cpu", "cpu")}
    model._move(torch.device("meta"))
    assert devices() == {"a": ("meta", "cpu"), "b": ("cpu", "cpu")}


def test_parked_optimizer_moments_survive_a_round_trip():
    import torch

    from tuft.backends.hf_training_model import HFTrainingModel

    def train(swap: bool) -> tuple[torch.Tensor, list[str]]:
        torch.manual_seed(0)
        model = HFTrainingModel.__new__(HFTrainingModel)
        model._host, model._pin = {}, True
        params = {k: torch.nn.Parameter(torch.randn(4)) for k in ("a", "b")}
        model.adapter_optimizer = {k: torch.optim.AdamW([p]) for k, p in params.items()}
        for _ in range(3):
            for k, p in params.items():
                p.grad = torch.randn(4)
                model.adapter_optimizer[k].step()
                if swap:  # park k, then unpark it
                    model._move_opt(k, torch.device("cpu"), model._host)
                    model._move_opt(k, torch.device("cpu"), model._host)
        return torch.cat([p.detach() for p in params.values()]), sorted(model._host)

    swapped, host = train(swap=True)
    plain, _ = train(swap=False)
    torch.testing.assert_close(swapped, plain)
    assert host == [
        "opt.a.0.exp_avg",
        "opt.a.0.exp_avg_sq",
        "opt.b.0.exp_avg",
        "opt.b.0.exp_avg_sq",
    ]
