import msgspec
import numpy as np
import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.configs.trainer import IcePopLossConfig
from prime_rl.orchestrator.batch import build_bin_cost, prepare_batch
from prime_rl.trainer.models import cast_float_and_contiguous
from prime_rl.trainer.models.layers.lm_head import FusedOutputLinear, VanillaOutputLinear, inject_prime_lm_head
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM
from prime_rl.trainer.rl.data import DataLoader, validate_score_centering_data
from prime_rl.trainer.rl.loss import (
    IcePopLoss,
    LossInputs,
    compute_entropy,
    compute_loss,
    selective_topk_log_softmax,
    shift_tensor_left,
    shift_tensor_right,
)
from prime_rl.transports.batch.types import MicroBatch, SamplingMask, TopLogprobs, TrainingSample
from prime_rl.utils.utils import default_dtype


def _baseline_logprobs_and_entropy(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, *, temperature: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline logprobs and entropy with per-token temperature tensor."""
    logits = hidden @ weight.t()
    # temperature is [b, s], logits is [b, s, v]
    logits = logits / temperature.unsqueeze(-1)
    logp = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    ent = compute_entropy(logits)
    return logp, ent


def test_fused_lm_head_matches_full_logits_forward_and_backward_cpu():
    torch.manual_seed(0)
    b, s, h, v = 2, 4, 8, 37
    temperature = torch.full((b, s), 1.7, dtype=torch.float32)
    chunk_size = 3

    hidden0 = torch.randn(b, s, h, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, v, (b, s), dtype=torch.long)
    weight0 = torch.randn(v, h, dtype=torch.float32, requires_grad=True)

    # Baseline
    logp0, ent0 = _baseline_logprobs_and_entropy(hidden0, weight0, labels, temperature=temperature)
    loss0 = logp0.sum()
    loss0.backward()
    grad_hidden0 = hidden0.grad.detach().clone()
    grad_weight0 = weight0.grad.detach().clone()

    # Fused
    hidden1 = hidden0.detach().clone().requires_grad_(True)
    weight1 = weight0.detach().clone().requires_grad_(True)
    lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    lm.weight = torch.nn.Parameter(weight1)

    out = lm(hidden1, labels, temperature=temperature)
    assert out.get("logits") is None
    assert out.get("logprobs") is not None
    assert out.get("entropy") is not None

    loss1 = out["logprobs"].sum()
    loss1.backward()
    grad_hidden1 = hidden1.grad.detach().clone()
    grad_weight1 = lm.weight.grad.detach().clone()

    torch.testing.assert_close(out["logprobs"], logp0, rtol=0, atol=1e-5)
    torch.testing.assert_close(out["entropy"], ent0, rtol=0, atol=1e-5)
    torch.testing.assert_close(grad_hidden1, grad_hidden0, rtol=0, atol=1e-5)
    torch.testing.assert_close(grad_weight1, grad_weight0, rtol=0, atol=1e-5)


def test_fused_lm_head_frozen_weight_backward_cpu():
    """Frozen LM-head weight: hidden gradient matches baseline, no weight gradient is produced."""
    torch.manual_seed(0)
    b, s, h, v = 2, 4, 8, 37
    temperature = torch.full((b, s), 1.7, dtype=torch.float32)
    chunk_size = 3

    hidden0 = torch.randn(b, s, h, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, v, (b, s), dtype=torch.long)
    weight0 = torch.randn(v, h, dtype=torch.float32)

    # Baseline (weight frozen)
    logp0, _ = _baseline_logprobs_and_entropy(hidden0, weight0, labels, temperature=temperature)
    logp0.sum().backward()
    grad_hidden0 = hidden0.grad.detach().clone()

    # Fused (weight frozen)
    hidden1 = hidden0.detach().clone().requires_grad_(True)
    lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    lm.weight = torch.nn.Parameter(weight0.clone(), requires_grad=False)

    out = lm(hidden1, labels, temperature=temperature)
    out["logprobs"].sum().backward()

    torch.testing.assert_close(hidden1.grad, grad_hidden0, rtol=0, atol=1e-5)
    assert lm.weight.grad is None


def test_fused_lm_head_topk_logprobs_forward_and_backward_cpu():
    torch.manual_seed(22)
    batch, seq, hidden_size, vocab, head = 2, 3, 5, 11, 4
    hidden0 = torch.randn(batch, seq, hidden_size, requires_grad=True)
    weight0 = torch.randn(vocab, hidden_size, requires_grad=True)
    labels = torch.randint(vocab, (batch, seq))
    topk_ids = torch.randint(vocab, (batch, seq, head))
    temperature = torch.linspace(0.8, 1.5, batch * seq).reshape(batch, seq)

    logits0 = hidden0 @ weight0.t() / temperature.unsqueeze(-1)
    logprobs0 = logits0.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    topk0 = logits0.log_softmax(-1).gather(-1, topk_ids)
    (logprobs0.sum() + topk0.sum()).backward()
    expected_hidden_grad = hidden0.grad.detach().clone()
    expected_weight_grad = weight0.grad.detach().clone()

    hidden1 = hidden0.detach().clone().requires_grad_(True)
    lm = FusedOutputLinear(hidden_size, vocab, chunk_size=2)
    lm.weight = torch.nn.Parameter(weight0.detach().clone())
    out = lm(hidden1, labels, temperature=temperature, topk_ids=topk_ids)
    assert out["topk_logprobs"].shape == (batch, seq, head)
    torch.testing.assert_close(out["logprobs"], logprobs0.detach(), rtol=0, atol=1e-5)
    torch.testing.assert_close(out["topk_logprobs"], topk0.detach(), rtol=0, atol=1e-5)
    (out["logprobs"].sum() + out["topk_logprobs"].sum()).backward()
    torch.testing.assert_close(hidden1.grad, expected_hidden_grad, rtol=0, atol=1e-5)
    torch.testing.assert_close(lm.weight.grad, expected_weight_grad, rtol=0, atol=1e-5)


def test_fused_lm_head_topk_replay_normalization_and_backward_cpu():
    torch.manual_seed(23)
    batch, seq, hidden_size, vocab = 1, 2, 4, 7
    hidden0 = torch.randn(batch, seq, hidden_size, requires_grad=True)
    weight0 = torch.randn(vocab, hidden_size, requires_grad=True)
    labels = torch.tensor([[2, 5]])
    support = torch.tensor([[1, 2, 4, 5]])
    sampling_mask = support.expand(batch, seq, -1)
    topk_ids = sampling_mask.clone()
    temperature = torch.ones(batch, seq)

    baseline_scaled = (hidden0 @ weight0.t()) / temperature.unsqueeze(-1)
    baseline_masked = torch.full_like(baseline_scaled, -torch.inf).scatter(
        -1, sampling_mask, baseline_scaled.gather(-1, sampling_mask)
    )
    baseline_logp = baseline_masked.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    baseline_head = baseline_masked.log_softmax(-1).gather(-1, topk_ids)
    (baseline_logp.sum() + baseline_head.sum()).backward()
    expected_hidden_grad = hidden0.grad.detach().clone()
    expected_weight_grad = weight0.grad.detach().clone()

    hidden = hidden0.detach().clone().requires_grad_(True)
    lm = FusedOutputLinear(hidden_size, vocab, chunk_size=1)
    lm.weight = torch.nn.Parameter(weight0.detach().clone())
    out = lm(hidden, labels, temperature=temperature, sampling_mask=sampling_mask, topk_ids=topk_ids)
    scaled = (hidden @ lm.weight.t()) / temperature.unsqueeze(-1)
    masked = torch.full_like(scaled, -torch.inf).scatter(-1, sampling_mask, scaled.gather(-1, sampling_mask))
    expected_logp = masked.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected_topk = masked.log_softmax(-1).gather(-1, topk_ids)
    torch.testing.assert_close(out["logprobs"], expected_logp, rtol=0, atol=1e-5)
    torch.testing.assert_close(out["topk_logprobs"], expected_topk, rtol=0, atol=1e-5)
    (out["logprobs"].sum() + out["topk_logprobs"].sum()).backward()
    torch.testing.assert_close(hidden.grad, expected_hidden_grad, rtol=0, atol=1e-5)
    torch.testing.assert_close(lm.weight.grad, expected_weight_grad, rtol=0, atol=1e-5)


def test_selective_topk_log_softmax_uses_replay_normalizer_cpu():
    logits = torch.tensor([[[0.5, 1.0, -0.4, 0.2]]])
    labels = torch.tensor([[2]])
    sampling_mask = torch.tensor([[[0, 2, -1]]])
    topk_ids = torch.tensor([[[0, 2, -1]]])
    actual = selective_topk_log_softmax(logits, topk_ids, sampling_mask, labels)
    expected = torch.log_softmax(torch.tensor([[0.5, -0.4]]), dim=-1)
    torch.testing.assert_close(actual[0, 0, :2], expected[0])
    assert actual[0, 0, 2] == 0


def test_score_centering_data_validation_mixed_replay_and_untruncated_rows_cpu():
    q = torch.tensor([[0.75, 0.25], [0.6, 0.3]])
    batch = {
        "loss_mask": torch.tensor([[True, True]]),
        "rl_weights": None,
        "top_logprobs_ids": torch.tensor([[[5, 6], [1, 2]]]),
        "top_logprobs_values": q.log().unsqueeze(0),
        "sampling_mask": torch.tensor([[[5, 6], [-1, -1]]]),
        "input_ids": torch.tensor([[5, 8]]),
    }
    validate_score_centering_data(batch, vocab_size=10)
    bad = {**batch, "input_ids": torch.tensor([[7, 8]])}
    with pytest.raises(ValueError, match="Sampled token is missing"):
        validate_score_centering_data(bad, vocab_size=10)


def test_data_loader_merges_replay_probabilities_and_untruncated_heads_cpu():
    sampler_probs = np.array([0.75, 0.25], dtype=np.float32)
    head_probs = np.array([0.6, 0.3], dtype=np.float32)
    sampling_mask = SamplingMask(
        ids=np.array([5, 6], dtype=np.int32).tobytes(),
        counts=np.array([2, 0], dtype=np.int32).tobytes(),
        logprobs=np.log(sampler_probs).tobytes(),
    )
    top_logprobs = TopLogprobs(
        ids=np.array([1, 2], dtype=np.int32).tobytes(),
        logprobs=np.log(head_probs).tobytes(),
        counts=np.array([0, 2], dtype=np.int32).tobytes(),
    )
    sample = TrainingSample(
        token_ids=[5, 8],
        mask=[True, True],
        logprobs=[float(np.log(sampler_probs[0])), float(np.log(head_probs[0]))],
        temperatures=[1.0, 1.0],
        env_name="env",
        advantages=[1.0, 1.0],
        sampling_mask=sampling_mask,
        top_logprobs=top_logprobs,
    )
    grid = prepare_batch([sample], seq_len=2, num_train_workers=1, bin_cost=build_bin_cost(None))
    payload = msgspec.msgpack.encode(grid)
    batch = msgspec.msgpack.Decoder(type=list[list[MicroBatch]]).decode(payload)[0][0]
    assert batch.sampling_mask is not None
    assert batch.sampling_mask.logprobs == sampling_mask.logprobs
    assert batch.top_logprobs is not None

    tensor_batch = DataLoader._micro_batch_to_tensor(object.__new__(DataLoader), batch)
    torch.testing.assert_close(tensor_batch["top_logprobs_ids"], torch.tensor([[[5, 6], [1, 2]]], dtype=torch.int32))
    torch.testing.assert_close(
        tensor_batch["top_logprobs_values"],
        torch.tensor([[sampler_probs.tolist(), head_probs.tolist()]]).log(),
    )
    validate_score_centering_data(tensor_batch, vocab_size=10)


def test_icepop_score_centering_partial_head_matches_dense_gradient_cpu():
    logits = torch.tensor([0.3, -0.4, 0.1, -0.9], requires_grad=True)
    logp = logits.log_softmax(-1)
    p = logp.detach().exp()
    head_size = 2
    head_ids = torch.arange(head_size)
    sampler_head = torch.tensor([0.2, 0.6])
    q = torch.cat((sampler_head, (1 - sampler_head.sum()) * p[head_size:] / p[head_size:].sum()))
    advantages = torch.tensor([1.3, -0.7, 0.2, -1.0])
    top_ids = head_ids.expand(logits.numel(), -1)
    inputs = LossInputs(
        trainer_logprobs=logp,
        inference_logprobs=q.log(),
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=torch.ones_like(q, dtype=torch.bool),
        loss_weights=q,
        trainer_topk_logprobs=logp[head_ids].expand(logits.numel(), -1),
        sampler_topk_logprobs=q[head_ids].log().expand(logits.numel(), -1),
        topk_valid=torch.ones_like(top_ids, dtype=torch.bool),
    )
    config = IcePopLossConfig(ratio_low=0.5, ratio_high=2.0, score_centering=True, score_centering_topk=head_size)
    actual, metrics = compute_loss(
        trainer_logprobs=[logp],
        inference_logprobs=[q.log()],
        ref_logprobs=None,
        advantages=[advantages],
        loss_mask=[inputs.loss_mask],
        rl_weights=[q],
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=IcePopLoss(config),
        rl_scale=1.0,
        ce_scale=1.0,
        ref_kl_scale=1.0,
        trainer_topk_logprobs=[inputs.trainer_topk_logprobs],
        sampler_topk_logprobs=[inputs.sampler_topk_logprobs],
        topk_valid=[inputs.topk_valid],
    )
    ratio = p / q
    weight = ratio * ((ratio >= config.ratio_low) & (ratio <= config.ratio_high))
    assert (weight > 0).tolist() == [True, False, False, False]
    baseline = (q * weight * logp).sum()
    oracle = -(q * advantages * (weight * logp - baseline)).sum()
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(oracle, logits)[0],
        atol=2e-7,
        rtol=2e-6,
    )
    assert "score_centering/tail_scale" in metrics
    assert not metrics["score_centering/tail_scale"].bool().any()


def test_icepop_score_centering_complete_support_replay_gradient_cpu():
    logits = torch.tensor([0.2, -0.3, 0.7, -0.2], requires_grad=True)
    support = torch.tensor([0, 2, 3])
    masked_logits = torch.full_like(logits, -torch.inf).scatter(0, support, logits[support])
    logp = masked_logits.log_softmax(-1)
    q = torch.tensor([0.4, 0.0, 0.35, 0.25])
    advantages = torch.tensor([1.0, 0.0, -0.5, 2.0])
    inputs = LossInputs(
        trainer_logprobs=logp[support],
        inference_logprobs=q[support].log(),
        ref_logprobs=None,
        advantages=advantages[support],
        loss_mask=torch.ones(support.numel(), dtype=torch.bool),
        loss_weights=q[support],
        trainer_topk_logprobs=logp[support].expand(support.numel(), -1),
        sampler_topk_logprobs=q[support].log().expand(support.numel(), -1),
        topk_valid=torch.ones((support.numel(), support.numel()), dtype=torch.bool),
        sampler_support_complete=torch.ones(support.numel(), dtype=torch.bool),
    )
    config = IcePopLossConfig(ratio_low=0.5, ratio_high=2.0, score_centering=True)
    output = IcePopLoss(config).loss(inputs)
    actual = output.loss
    ratio = logp[support].detach().exp() / q[support]
    weight = ratio * ((ratio >= config.ratio_low) & (ratio <= config.ratio_high))
    baseline = (q[support] * weight * logp[support]).sum()
    oracle = -(q[support] * advantages[support] * (weight * logp[support] - baseline)).sum()
    actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    oracle_grad = torch.autograd.grad(oracle, logits)[0]
    torch.testing.assert_close(actual_grad, oracle_grad, atol=2e-7, rtol=2e-6)
    assert actual_grad[1] == 0
    assert torch.equal(output.metrics["score_centering/tail_scale"], torch.zeros(support.numel()))


def test_backward_epilogue_receives_contiguous_broadcast_upstream_grad_cpu():
    torch.manual_seed(4)
    rows, hidden_size, vocab = 4, 5, 17
    hidden = torch.randn(1, rows, hidden_size, requires_grad=True)
    weight = torch.randn(vocab, hidden_size, requires_grad=True)
    labels = torch.randint(vocab, (1, rows))
    temperature = torch.linspace(0.7, 1.8, rows).reshape(1, rows)
    seen_strides = []

    baseline_hidden = hidden.detach().clone().requires_grad_(True)
    baseline_weight = weight.detach().clone().requires_grad_(True)
    baseline_logprobs, _ = _baseline_logprobs_and_entropy(
        baseline_hidden, baseline_weight, labels, temperature=temperature
    )
    baseline_logprobs.sum().backward()

    def backward_epilogue(logits, inv_temperature, logz, grad_logprobs, chunk_labels, vocab_start):
        seen_strides.append(grad_logprobs.stride())
        assert grad_logprobs.is_contiguous()
        scaled = logits.float() * inv_temperature.float().unsqueeze(-1)
        grad_logits = -grad_logprobs.unsqueeze(-1) * torch.exp(scaled - logz.unsqueeze(-1))
        in_range = (chunk_labels >= vocab_start) & (chunk_labels < vocab_start + logits.shape[-1])
        local = (chunk_labels - vocab_start).clamp(0, logits.shape[-1] - 1).unsqueeze(-1)
        grad_logits.scatter_add_(1, local, (grad_logprobs * in_range).unsqueeze(-1))
        return grad_logits * inv_temperature.float().unsqueeze(-1)

    lm = FusedOutputLinear(hidden_size, vocab, chunk_size=rows, backward_epilogue_fn=backward_epilogue)
    lm.weight = torch.nn.Parameter(weight.detach().clone())
    out = lm(hidden, labels, temperature=temperature)
    output_gradient_strides = []
    out["logprobs"].register_hook(lambda grad: output_gradient_strides.append(grad.stride()) or grad)
    out["logprobs"].sum().backward()

    assert any(0 in stride for stride in output_gradient_strides)
    assert seen_strides == [(1,)]
    assert hidden.grad is not None
    assert lm.weight.grad is not None
    torch.testing.assert_close(out["logprobs"], baseline_logprobs, rtol=0, atol=1e-5)
    torch.testing.assert_close(hidden.grad, baseline_hidden.grad, rtol=0, atol=1e-5)
    torch.testing.assert_close(lm.weight.grad, baseline_weight.grad, rtol=0, atol=1e-5)


def test_fused_lm_head_requires_labels():
    """Test that FusedOutputLinear raises assertion error when labels is None."""
    torch.manual_seed(0)
    b, s, h, v = 2, 3, 4, 9

    hidden = torch.randn(b, s, h, dtype=torch.float32)
    weight = torch.randn(v, h, dtype=torch.float32)
    temperature = torch.full((b, s), 1.0, dtype=torch.float32)

    lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=5)
    lm.weight = torch.nn.Parameter(weight)

    with pytest.raises(AssertionError, match="FusedOutputLinear requires labels"):
        lm(hidden, labels=None, temperature=temperature)


def test_vanilla_lm_head_returns_logits():
    """Test that VanillaOutputLinear returns logits."""
    torch.manual_seed(0)
    b, s, h, v = 2, 3, 4, 9

    hidden = torch.randn(b, s, h, dtype=torch.float32)
    weight = torch.randn(v, h, dtype=torch.float32)

    lm = VanillaOutputLinear(in_features=h, out_features=v)
    lm.weight = torch.nn.Parameter(weight)

    # VanillaOutputLinear doesn't use temperature - it just returns logits
    out = lm(hidden, labels=None, temperature=None)
    assert out.get("logits") is not None
    assert out.get("logprobs") is None
    assert out.get("entropy") is None

    logits_ref = hidden @ weight.t()
    torch.testing.assert_close(out["logits"], logits_ref, rtol=0, atol=1e-6)


def test_fused_vs_vanilla_integration():
    """Integration test comparing fused and vanilla outputs after postprocessing."""
    torch.manual_seed(42)
    b, s, h, v = 2, 4, 8, 37
    temp_value = 1.7
    temperature = torch.full((b, s), temp_value, dtype=torch.float32)
    chunk_size = 3

    hidden = torch.randn(b, s, h, dtype=torch.float16)
    labels = torch.randint(0, v, (b, s), dtype=torch.long)
    weight = torch.randn(v, h, dtype=torch.float16)

    # Vanilla path: get logits, compute logprobs manually
    vanilla_lm = VanillaOutputLinear(in_features=h, out_features=v)
    vanilla_lm.weight = torch.nn.Parameter(weight.clone())
    vanilla_out = cast_float_and_contiguous(vanilla_lm(hidden, labels=None, temperature=None))

    assert vanilla_out.get("logits") is not None
    logits = vanilla_out["logits"] / temp_value
    vanilla_logprobs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    vanilla_entropy = compute_entropy(logits)

    # Fused path: get logprobs and entropy directly
    fused_lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    fused_lm.weight = torch.nn.Parameter(weight.clone())
    fused_out = cast_float_and_contiguous(fused_lm(hidden, labels=labels, temperature=temperature))

    assert fused_out.get("logprobs") is not None
    assert fused_out.get("entropy") is not None

    # Compare: fused should match vanilla within tolerance
    torch.testing.assert_close(fused_out["logprobs"], vanilla_logprobs, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(fused_out["entropy"], vanilla_entropy, rtol=1e-3, atol=1e-4)


def test_fused_lm_head_correct_shift():
    """
    End-to-end test that the fused LM head with shifted labels, after shift_tensor_right,
    produces logprobs aligned with the inference convention.

    This simulates the full training loop behavior and verifies the importance ratio
    (trainer_logprobs - inference_logprobs) is ~0 for positions that matter in training.
    """
    torch.manual_seed(999)
    b, s, h, v = 2, 16, 32, 50
    temp_value = 1.5
    temperature = torch.full((b, s), temp_value, dtype=torch.float32)
    chunk_size = 13

    hidden = torch.randn(b, s, h, dtype=torch.float32)
    weight = torch.randn(v, h, dtype=torch.float32)
    input_ids = torch.randint(0, v, (b, s), dtype=torch.long)

    # Create shifted labels as done in training
    labels = shift_tensor_left(input_ids)

    # === Fused path (as in training) ===
    fused_lm = FusedOutputLinear(in_features=h, out_features=v, chunk_size=chunk_size)
    fused_lm.weight = torch.nn.Parameter(weight.clone())
    fused_out = fused_lm(hidden, labels=labels, temperature=temperature)
    trainer_logprobs = shift_tensor_right(fused_out["logprobs"])

    # === Inference convention (baseline) ===
    logits = hidden @ weight.t()
    logits = logits / temp_value
    # Shift logits right (prepend zeros, drop last) to get inference convention
    shifted_logits = torch.cat([torch.zeros(b, 1, v, dtype=logits.dtype), logits[:, :-1, :]], dim=1)
    inference_logprobs = (
        torch.log_softmax(shifted_logits, dim=-1).gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    )

    assert torch.all(trainer_logprobs[:, 0] == 0), "Position 0 should be 0 after shift_tensor_right"

    importance_ratio = trainer_logprobs[:, 1:] - inference_logprobs[:, 1:]
    torch.testing.assert_close(
        importance_ratio,
        torch.zeros(b, s - 1),
        rtol=0,
        atol=1e-4,
        msg="Importance ratio at positions 1 to s-1 should be ~0 (same token probs)",
    )


@pytest.mark.gpu
def test_inject_prime_lm_head_vanilla():
    """Test that inject_prime_lm_head correctly wraps the model with VanillaOutputLinear."""
    torch.manual_seed(123)

    config = Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_hidden_layers=2,
        vocab_size=1000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
    )

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3ForCausalLM._from_config(config)

    # Wrap with VanillaOutputLinear (chunk_size=None)
    inject_prime_lm_head(model, chunk_size=None)

    assert isinstance(model.lm_head, VanillaOutputLinear), "lm_head should be VanillaOutputLinear"

    # Test forward with labels and temperature
    batch_size, seq_len = 2, 64

    with torch.device("cuda"):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        temperature = torch.full((batch_size, seq_len), 1.5, dtype=torch.float32)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            temperature=temperature,
            seq_lens=torch.tensor([seq_len], device="cuda"),
        )

    # VanillaOutputLinear returns logits
    assert isinstance(out, dict), "Output should be PrimeLmOutput (dict)"
    assert out.get("logits") is not None, "Vanilla path should return logits"
    assert out.get("logprobs") is None, "Vanilla path should not return logprobs"
    assert out["logits"].shape == (batch_size, seq_len, config.vocab_size), "Logits shape mismatch"


@pytest.mark.gpu
def test_inject_prime_lm_head_fused():
    """Test that inject_prime_lm_head correctly wraps the model with FusedOutputLinear."""
    torch.manual_seed(123)

    config = Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        num_hidden_layers=2,
        vocab_size=1000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
    )

    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3ForCausalLM._from_config(config)

    # Wrap with FusedOutputLinear
    inject_prime_lm_head(model, chunk_size=512)

    assert isinstance(model.lm_head, FusedOutputLinear), "lm_head should be FusedOutputLinear"

    # Test forward with labels and temperature
    batch_size, seq_len = 2, 64

    with torch.device("cuda"):
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        temperature = torch.full((batch_size, seq_len), 1.5, dtype=torch.float32)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            temperature=temperature,
            seq_lens=torch.tensor([seq_len], device="cuda"),
        )

    # FusedOutputLinear returns logprobs and entropy
    assert isinstance(out, dict), "Output should be PrimeLmOutput (dict)"
    assert out.get("logprobs") is not None, "Fused path should return logprobs"
    assert out.get("entropy") is not None, "Fused path should return entropy"
    assert out.get("logits") is None, "Fused path should not return logits"
    assert out["logprobs"].shape == (batch_size, seq_len), "Logprobs shape mismatch"
    assert out["entropy"].shape == (batch_size, seq_len), "Entropy shape mismatch"
