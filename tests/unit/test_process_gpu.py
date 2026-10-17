import pytest

from prime_rl.utils.process import partition_gpus


def test_partition_gpus_shares_the_only_visible_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    inference_gpus, trainer_gpus = partition_gpus(1, 1, share=True)

    assert inference_gpus == [3]
    assert trainer_gpus == [3]


def test_partition_gpus_keeps_default_gpu_partition_disjoint(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    inference_gpus, trainer_gpus = partition_gpus(1, 1)

    assert inference_gpus == [3]
    assert trainer_gpus == [5]


def test_partition_gpus_rejects_sharing_when_multiple_gpus_are_visible(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    with pytest.raises(ValueError, match="exactly one physical GPU"):
        partition_gpus(1, 1, share=True)
