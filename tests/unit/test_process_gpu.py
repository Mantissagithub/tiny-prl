import os
import sys
import time
from pathlib import Path

import pytest

from prime_rl.utils.process import ProcessGroup, partition_gpus


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


def test_process_group_cleanup_terminates_children_before_joining_monitors(tmp_path: Path):
    processes = ProcessGroup()
    child = processes.start(
        "sleeping-child",
        [sys.executable, "-c", "import time; time.sleep(60)"],
        dict(os.environ),
        tmp_path / "sleeping-child.log",
    )

    started = time.monotonic()
    try:
        processes.cleanup()
    finally:
        processes.cleanup()
    elapsed = time.monotonic() - started

    assert child.poll() is not None
    assert all(not thread.is_alive() for thread in processes.monitor_threads)
    assert elapsed < 3
