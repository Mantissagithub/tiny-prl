from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.decode_backend import (
    KERNEL_NAME_135M,
    KERNEL_NAME_360M,
    _kernel_name_for_config,
    _metadata_eligible,
    _select_decode_step,
)


@pytest.mark.parametrize(
    ("geometry", "expected"),
    [
        ((576, 1536, 30, 9, 3, 64, 1e-5), KERNEL_NAME_135M),
        ((960, 2560, 32, 15, 5, 64, 1e-5), KERNEL_NAME_360M),
    ],
)
def test_exact_supported_decode_geometry(geometry, expected):
    hidden, intermediate, layers, heads, kv_heads, head_dim, eps = geometry
    config = SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        rms_norm_eps=eps,
    )

    assert _kernel_name_for_config(config) == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 961),
        ("intermediate_size", 2561),
        ("num_hidden_layers", 31),
        ("num_attention_heads", 16),
        ("num_key_value_heads", 4),
        ("head_dim", 32),
        ("rms_norm_eps", 1e-6),
    ],
)
def test_unsupported_geometry_uses_stock_forward(field, value):
    config = SimpleNamespace(
        hidden_size=960,
        intermediate_size=2560,
        num_hidden_layers=32,
        num_attention_heads=15,
        num_key_value_heads=5,
        head_dim=64,
        rms_norm_eps=1e-5,
    )
    setattr(config, field, value)

    assert _kernel_name_for_config(config) is None


class FlashAttentionMetadata:
    max_query_len = 1
    num_actual_tokens = 2

    def __init__(self, width, batch=2):
        self.num_actual_tokens = batch
        self.block_table = torch.empty((batch, width), dtype=torch.int32)


def test_decode_metadata_keeps_360m_batch_and_cache_limits():
    kv_cache = torch.empty((8, 5, 16, 128), dtype=torch.bfloat16)
    model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(kv_cache=kv_cache)))]
    )
    metadata = FlashAttentionMetadata(width=512)

    assert _metadata_eligible(metadata, model, KERNEL_NAME_360M, 2)
    assert not _metadata_eligible(metadata, model, KERNEL_NAME_360M, 3)
    assert not _metadata_eligible(FlashAttentionMetadata(width=513), model, KERNEL_NAME_360M, 2)


@pytest.mark.parametrize(
    ("shape", "width", "batch"),
    [((8, 5, 32, 128), 8, 2), ((8, 4, 16, 128), 8, 2), ((8, 5, 16, 128), 513, 2), ((8, 5, 16, 128), 8, 3)],
)
def test_unsupported_360m_metadata_never_selects_decode_kernel(shape, width, batch):
    kv_cache = torch.empty(shape, dtype=torch.bfloat16)
    model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(kv_cache=kv_cache)))]
    )
    metadata = FlashAttentionMetadata(width=width, batch=batch)
    calls = []
    steps = {KERNEL_NAME_360M: lambda **_: calls.append("called")}

    selected = _select_decode_step(steps, model, KERNEL_NAME_360M, metadata, batch)
    if selected is not None:
        selected()

    assert selected is None
    assert calls == []


def test_135m_batch_limit_is_unchanged():
    metadata = FlashAttentionMetadata(width=1)
    metadata.num_actual_tokens = 4

    assert _metadata_eligible(metadata, None, KERNEL_NAME_135M, 4)
    assert not _metadata_eligible(metadata, None, KERNEL_NAME_135M, 5)


def test_eligible_360m_batch_selects_registered_step():
    kv_cache = torch.empty((8, 5, 16, 128), dtype=torch.bfloat16)
    model = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(attn=SimpleNamespace(kv_cache=kv_cache)))]
    )
    metadata = FlashAttentionMetadata(width=8)
    decode_step = lambda **_: None

    assert _select_decode_step({KERNEL_NAME_360M: decode_step}, model, KERNEL_NAME_360M, metadata, 2) is decode_step
