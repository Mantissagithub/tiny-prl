"""Launcher boundaries that must hold before a real GPU run."""

import importlib.util
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("tiny_prl")
guard = load("tiny_guard")


def test_stock_preset_does_not_enable_lossless():
    args = launcher.config_args("smollm2-360m-stock", ["--max-steps", "100"])
    assert str(launcher.EXAMPLES / "smollm2-360m-lossless.toml") not in args
    assert args[-2:] == ["--max-steps", "100"]
    optimized = launcher.config_args("smollm2-360m", [])
    assert str(launcher.EXAMPLES / "smollm2-360m-lossless.toml") in optimized
    assert str(launcher.EXAMPLES / "laptop.toml") in optimized


def test_custom_config_keeps_native_overrides(tmp_path):
    config = tmp_path / "my training.toml"
    args = launcher.config_args("@", [str(config), "--max-steps", "3"])
    assert args[:2] == ["@", str(config)]
    assert args[-2:] == ["--max-steps", "3"]
    assert str(launcher.EXAMPLES / "laptop.toml") not in args
    with pytest.raises(ValueError, match="requires a config"):
        launcher.config_args("@", [])


def test_offline_resolution_refuses_ambiguous_snapshots(tmp_path):
    snapshots = tmp_path / "models--org--model/snapshots"
    one = snapshots / "first"
    one.mkdir(parents=True)
    (one / "config.json").write_text("{}")
    assert launcher.cached_model("org/model", tmp_path) == str(one)
    two = snapshots / "second"
    two.mkdir()
    (two / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="unambiguous"):
        launcher.cached_model("org/model", tmp_path)
    refs = snapshots.parent / "refs"
    refs.mkdir()
    (refs / "main").write_text("first")
    assert launcher.cached_model("org/model", tmp_path) == "org/model"


def test_environment_drops_private_experiment_preloads(monkeypatch):
    monkeypatch.setenv("PRIME_R30_PRIVATE_PRELOAD", "/some/private/hook.py")
    monkeypatch.setenv("PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS", "1")
    env = launcher.environment()
    assert "PRIME_R30_PRIVATE_PRELOAD" not in env
    assert "PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS" not in env
    assert str(launcher.REPO / "deps/prime-kernels") in env["PYTHONPATH"]
    assert env["MAX_JOBS"] == "1"


def test_guard_checks_resource_and_time_boundaries():
    gib = 1024**3
    assert guard.stop_reason(3 * gib, 3 * gib, 7600, 5, 10) is None
    assert guard.stop_reason(gib, 3 * gib, 7600, 5, 10) == "available host RAM below 2 GiB"
    assert guard.stop_reason(3 * gib, gib, 7600, 5, 10) == "free disk below 2 GiB"
    assert guard.stop_reason(3 * gib, 3 * gib, 7677, 5, 10) == "GPU memory exceeded 7676 MiB"
    assert guard.stop_reason(3 * gib, 3 * gib, 7600, 11, 10) == "run timeout"
