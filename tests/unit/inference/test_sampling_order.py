import torch

from prime_rl.inference.vllm.sampling_order import apply_top_p_then_top_k, reference_top_p_then_top_k


def test_bounded_top_p_then_top_k_matches_full_sort():
    torch.manual_seed(7)
    source = torch.randn(4, 128)
    k = torch.tensor([8, 16, 32, 48])
    p = torch.tensor([0.8, 0.9, 0.95, 0.99])

    expected = reference_top_p_then_top_k(source.clone(), k, p)
    actual = apply_top_p_then_top_k(source.clone(), k, p)

    torch.testing.assert_close(actual, expected)


def test_top_p_then_top_k_falls_back_at_candidate_tie():
    source = torch.tensor([[5.0, 4.0, 4.0, 3.0], [2.0, 1.0, 0.0, -1.0]])
    k = torch.tensor([1, 2])
    p = torch.tensor([0.7, 0.8])

    expected = reference_top_p_then_top_k(source.clone(), k, p)
    actual = apply_top_p_then_top_k(source.clone(), k, p)

    torch.testing.assert_close(actual, expected)
