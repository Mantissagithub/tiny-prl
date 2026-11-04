"""Launcher boundaries that must hold before a real GPU run."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("tiny_prl")
guard = load("tiny_guard")
hf = load("tiny_hf")


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


def test_wordle_overlay_follows_model_preset_and_keeps_overrides():
    args = launcher.config_args("smollm2-360m", ["--max-steps", "1"], "primeintellect/wordle")
    overlay = args.index(str(launcher.EXAMPLES / "wordle.toml"))
    assert args.index(str(launcher.EXAMPLES / "smollm2-360m-full-stream64.toml")) < overlay
    assert overlay < args.index("--no-dashboard")
    assert args[-2:] == ["--max-steps", "1"]


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


def test_model_cleanup_requires_verification_and_run_ownership(tmp_path):
    run = tmp_path / "run"
    model = tmp_path / "owned-model"
    for path in [run / "checkpoints", run / "broadcasts", run / "export", model]:
        path.mkdir(parents=True)
        (path / "weights").write_bytes(b"keep until verified")
    state = {"download_dir": str(model), "status": "prepared"}
    with pytest.raises(ValueError, match="before a verified"):
        hf.cleanup_models(run, state)
    assert (model / "weights").exists()
    state.update(status="verified", uploaded_files={"weights": "checked-hash"})
    with pytest.raises(ValueError, match="not owned"):
        hf.cleanup_models(run, state)
    assert (run / "checkpoints/weights").exists()
    (model / ".tiny-prl-owner").write_text(str(run.resolve()))
    (run / "metrics.json").write_text(json.dumps({"loss": 1.0}))
    hf.cleanup_models(run, state)
    assert not model.exists() and not (run / "checkpoints").exists()
    assert (run / "metrics.json").exists()


def test_removed_model_cache_is_recreated_for_an_initialized_run(tmp_path, monkeypatch):
    import huggingface_hub

    run = tmp_path / "run"
    model = tmp_path / "cleared-cache"
    hf.receipt_path(run).parent.mkdir(parents=True)
    hf.receipt_path(run).write_text(
        json.dumps({"base_model": "owner/base", "base_revision": "pinned", "download_dir": str(model)})
    )

    def download(repo, *, local_dir, revision, **kwargs):
        assert revision == "pinned"
        (local_dir / "model.safetensors").write_bytes(b"downloaded again")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert hf.prepare_run(run, "owner/base", 10, None) == model
    assert (model / ".tiny-prl-owner").read_text() == str(run.resolve())
    assert (model / "model.safetensors").read_bytes() == b"downloaded again"


def test_remote_verification_rejects_wrong_content_before_cleanup(tmp_path):
    import hashlib

    (tmp_path / "config.json").write_bytes(b"{}")
    digest = hashlib.sha1(b"blob 2\0{}").hexdigest()
    item = SimpleNamespace(path="config.json", size=2, lfs=None, blob_id=digest)
    api = SimpleNamespace(list_repo_tree=lambda *args, **kwargs: [item])
    assert hf.verify_files(api, "owner/model", "commit", tmp_path) == {"config.json": hashlib.sha256(b"{}").hexdigest()}
    item.blob_id = "wrong"
    with pytest.raises(ValueError, match="content mismatch"):
        hf.verify_files(api, "owner/model", "commit", tmp_path)
    assert (tmp_path / "config.json").exists()


def test_failed_hf_upload_preserves_checkpoint_and_download(tmp_path, monkeypatch):
    import huggingface_hub

    run = tmp_path / "run"
    download = tmp_path / "download"
    adapter = run / "broadcasts/step_1"
    checkpoint = run / "checkpoints/step_1/trainer"
    for directory in [download, adapter, checkpoint, run / "configs"]:
        directory.mkdir(parents=True, exist_ok=True)
    (checkpoint / ".metadata").write_bytes(b"checkpoint")
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (download / ".tiny-prl-owner").write_text(str(run.resolve()))
    (download / "model.safetensors").write_bytes(b"base weights")
    hf.receipt_path(run).write_text(
        json.dumps({"base_model": "owner/base", "max_steps": 1, "download_dir": str(download), "status": "prepared"})
    )

    def fail_upload(**kwargs):
        raise OSError("network unavailable")

    api = SimpleNamespace(create_repo=lambda *args, **kwargs: None, upload_folder=fail_upload)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: api)
    with pytest.raises(OSError, match="network unavailable"):
        hf.finish_run(run, "owner/model", {}, None)
    assert (checkpoint / ".metadata").exists()
    assert (adapter / "adapter_model.safetensors").exists()
    assert (download / "model.safetensors").exists()
    assert (run / "export/adapter_model.safetensors").exists()
