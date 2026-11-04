"""Own run-local model files until their HF export is verified remotely."""

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def receipt_path(run):
    return run / "configs/tiny-prl.json"


def prepare_run(run, model, steps, repo, environment=None):
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import validate_repo_id

    from prime_rl.utils.pathing import has_run_artifacts

    if has_run_artifacts(run):
        raise ValueError(f"Run folder already contains training artifacts: {run}")
    if repo:
        validate_repo_id(repo)
        HfApi().whoami()
    record = receipt_path(run)
    if record.exists():
        state = json.loads(record.read_text())
        if state["base_model"] != model:
            raise ValueError("Initialized model differs from requested model; choose a new output folder")
        directory = Path(state["download_dir"])
        if not directory.exists():
            directory.mkdir(parents=True)
            (directory / ".tiny-prl-owner").write_text(str(run.resolve()))
        elif not (directory / ".tiny-prl-owner").is_file():
            raise ValueError("Model cache has no ownership marker; refusing to reuse it")
    else:
        directory = Path.home() / ".cache/tiny-prl/models" / uuid.uuid4().hex
        directory.mkdir(parents=True)
        (directory / ".tiny-prl-owner").write_text(str(run.resolve()))
        state = {"base_model": model, "download_dir": str(directory), "run_dir": str(run.resolve())}
    if "base_revision" not in state:
        state["base_revision"] = HfApi().model_info(model).sha
    state.update(max_steps=steps, hf_repo=repo, environment=environment, status="prepared")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(state, indent=2) + "\n")
    print(f"Preparing {model} in {directory}", flush=True)
    snapshot_download(
        model,
        revision=state["base_revision"],
        local_dir=directory,
        max_workers=2,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
    )
    return directory


def verify_files(api, repo, revision, folder):
    """Check remote content identities at the immutable upload commit."""
    remote = {
        p.path: p
        for p in api.list_repo_tree(repo, repo_type="model", revision=revision, recursive=True)
        if hasattr(p, "blob_id")
    }
    hashes = {}
    for local in sorted(folder.rglob("*")):
        if not local.is_file():
            continue
        name = local.relative_to(folder).as_posix()
        item = remote.get(name)
        if item is None or item.size != local.stat().st_size:
            raise ValueError(f"HF verification failed for {name}: absent or wrong size")
        with local.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if item.lfs:
            matches = item.lfs.sha256 == digest
        else:
            data = local.read_bytes()
            matches = item.blob_id == hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if not matches:
            raise ValueError(f"HF verification failed for {name}: content mismatch")
        hashes[name] = digest
    if not hashes:
        raise ValueError("Refusing to verify an empty export")
    return hashes


def cleanup_models(run, state):
    if state.get("status") != "verified" or not state.get("uploaded_files"):
        raise ValueError("Refusing model deletion before a verified HF upload")
    directory = Path(state["download_dir"])
    owner = directory / ".tiny-prl-owner"
    if directory.exists() and (not owner.is_file() or owner.read_text() != str(run.resolve())):
        raise ValueError("Refusing to delete a model directory not owned by this run")
    for path in [run / "checkpoints", run / "broadcasts", run / "export", directory]:
        if path.exists():
            shutil.rmtree(path)


def finish_run(run, repo, env, guarded):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(repo)
    record = receipt_path(run)
    state = json.loads(record.read_text())
    if state.get("status") == "verified":
        cleanup_models(run, state)
        print(f"Already uploaded and verified: {state['url']}")
        return 0
    step = state["max_steps"]
    checkpoint = run / "checkpoints" / f"step_{step}"
    if not (checkpoint / "trainer/.metadata").is_file():
        raise ValueError(f"Final checkpoint is missing: {checkpoint}. Local weights retained.")
    export = run / "export"
    if export.exists():
        shutil.rmtree(export)
    adapter = run / "broadcasts" / f"step_{step}"
    base = Path(state["download_dir"])
    if (adapter / "adapter_config.json").is_file():
        export.mkdir()
        for path in adapter.iterdir():
            if path.suffix in {".json", ".safetensors"}:
                shutil.copy2(path, export / path.name)
        path = export / "adapter_config.json"
        config = json.loads(path.read_text())
        config["base_model_name_or_path"] = state["base_model"]
        if state.get("base_revision"):
            config["revision"] = state["base_revision"]
        path.write_text(json.dumps(config, indent=2) + "\n")
        for path in base.iterdir():
            if path.is_file() and (
                path.name.startswith("tokenizer") or path.suffix == ".model" or path.name == "special_tokens_map.json"
            ):
                shutil.copy2(path, export / path.name)
    else:
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if active:
            raise ValueError("GPU is busy; retry upload after the current workload finishes")
        command = [
            "uv",
            "run",
            "--offline",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(ROOT / "tools/convert_dcp_to_bf16.py"),
            str(checkpoint),
            str(export),
        ]
        guard = run.parent / ".tiny-prl" / uuid.uuid4().hex
        if guarded(command, guard, 600, env):
            raise ValueError("Checkpoint export failed. Local weights retained.")
    if not list(export.glob("*.safetensors")):
        raise ValueError("No model weights in export. Local checkpoints retained.")
    (export / "README.md").write_text(
        f"---\nbase_model: {state['base_model']}\ntags:\n- reinforcement-learning\n---\n\n"
        f"# Tiny PRL checkpoint\n\nEnvironment: {state.get('environment', 'see run configs')}. "
        f"Export after {step} RL updates. "
        "This upload does not establish improved task accuracy. "
        "Weights are provided for inference or new training; optimizer/resume state is not included.\n"
    )
    algorithm = run / "configs/algo.py"
    if algorithm.is_file():
        shutil.copy2(algorithm, export / "algo.py")
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    print(f"Uploading final weights to {repo}; local weights stay until verification completes.", flush=True)
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(export),
        commit_message=f"Upload final checkpoint at step {step}",
    )
    hashes = verify_files(api, repo, commit.oid, export)
    state.update(
        status="verified",
        hf_repo=repo,
        commit=commit.oid,
        uploaded_files=hashes,
        url=f"https://huggingface.co/{repo}/tree/{commit.oid}",
    )
    record.write_text(json.dumps(state, indent=2) + "\n")
    cleanup_models(run, state)
    print(f"Verified upload: {state['url']}\nLocal model weights removed; logs and configs retained.")
    return 0
