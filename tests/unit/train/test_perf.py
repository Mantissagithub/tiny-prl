from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

from prime_rl.trainer.perf import PerfCounter


def _tiny_model() -> torch.nn.Module:
    model = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(
        vocab_size=8,
        hidden_size=4,
        intermediate_size=8,
        head_dim=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
    )
    return model


def test_perf_counter_rtx_4060_laptop_peak(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA GeForce RTX 4060 Laptop GPU")

    perf_counter = PerfCounter(_tiny_model(), seq_len=16)

    assert perf_counter.gpu_peak_flops == pytest.approx(29.12256e12)
    assert perf_counter.get_step_mfu(tokens=1, fwd_bwd_time=1.0) is not None


def test_perf_counter_unknown_gpu_disables_mfu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Unrecognized GPU")

    perf_counter = PerfCounter(_tiny_model(), seq_len=16)

    assert perf_counter.gpu_peak_flops is None
    assert perf_counter.get_step_mfu(tokens=1, fwd_bwd_time=1.0) is None


def test_perf_counter_without_cuda_disables_mfu(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    perf_counter = PerfCounter(_tiny_model(), seq_len=16)

    assert perf_counter.gpu_peak_flops is None
    assert perf_counter.get_step_mfu(tokens=1, fwd_bwd_time=1.0) is None


def test_perf_counter_preserves_known_a100_peak():
    counter = object.__new__(PerfCounter)
    assert counter._get_peak_flops("NVIDIA A100-SXM4-80GB") == 312e12


@pytest.mark.parametrize(
    "model_name, active_params, flops_per_token",
    [("Qwen/Qwen3-0.6B", 595_984_384, 4_280_549_376), ("Jackmin108/debug-moe-0.5B", 256_442_368, 1_840_644_096)],
)
def test_perf_counter(model_name: str, active_params: int, flops_per_token: int):
    # This speeds up the model loading as its a fake device
    with torch.device("meta"):
        config = AutoConfig.from_pretrained(model_name)
        if getattr(config, "pad_token_id", None) is None:
            gen_config = GenerationConfig.from_model_config(config)
            # Use `is not None` instead of truthiness: token ID 0 is valid.
            config.pad_token_id = next(
                (
                    v
                    for v in [gen_config.pad_token_id, gen_config.eos_token_id, getattr(config, "eos_token_id", None)]
                    if v is not None
                ),
                None,
            )
        model = AutoModelForCausalLM.from_config(config)
    perf_counter = PerfCounter(model, seq_len=1024)

    assert perf_counter.get_active_mm_params(config) == active_params, (
        f"Expected {active_params:,} active parameters, got {perf_counter.get_active_mm_params(config):,} active parameters"
    )
    assert perf_counter.num_flop_per_token == flops_per_token, (
        f"Expected {flops_per_token:,} FLOPS per token, got {perf_counter.num_flop_per_token:,} FLOPS per token"
    )
