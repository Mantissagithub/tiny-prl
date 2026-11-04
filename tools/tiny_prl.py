"""Laptop launcher; Prime-RL owns config parsing and the training pipeline."""

import argparse
import fcntl
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples/tiny"
GPU_LOCK = REPO / "outputs/tiny-kernel-20261020/triton/gpu.lock"
MODEL_IDS = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "qwen3-600m-lora": "Qwen/Qwen3-0.6B",
}
ENVIRONMENTS = {
    "primeintellect/wordle": "wordle.toml",
    "primeintellect/gsm8k": "gsm8k.toml",
    "primeintellect/reverse-text": "reverse-text.toml",
    "primeintellect/alphabet-sort": None,
}
PRESETS = {
    "smollm2-360m": (
        "360M full parameters, lossless decode, stream64",
        ["smollm2-360m-full-stream64.toml", "smollm2-360m-lossless.toml"],
    ),
    "smollm2-360m-stock": ("360M full parameters, stock vLLM, stream64", ["smollm2-360m-full-stream64.toml"]),
    "smollm2-135m": ("135M full parameters, CUDA IPC, early trainer start", ["smollm2-full-ipc.toml"]),
    "qwen3-600m-lora": ("600M LoRA rank 8, stock vLLM", ["rl.toml"]),
}


def environment():
    env = os.environ.copy()
    # Use this checkout and its pinned kernel submodule, independent of editable installs.
    paths = [
        REPO / "src",
        REPO / "packages/prime-rl-configs/src",
        REPO / "deps/prime-kernels",
        REPO / "deps/verifiers",
        REPO / "deps/renderers",
        REPO / "deps/pydantic-config/src",
        REPO / "tools",
    ]
    paths += sorted((REPO / "deps/verifiers/environments").glob("*"))
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["MAX_JOBS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9+PTX")
    env.setdefault("TORCH_EXTENSIONS_DIR", str(REPO / "outputs/kernel-cache"))
    for name in (
        "PRIME_DECODE_MEGAKERNEL_360M",
        "PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS",
        "PRIME_R30_PRIVATE_PRELOAD",
        "PRIME_R30_PRELOAD_TRACE_DIR",
        "PRL_RUN_ID",
        "PRL_RUN_NAME",
    ):
        env.pop(name, None)
    return env


def config_args(preset, overrides, environment=None):
    run_name = f"{preset if preset != '@' else 'custom'}-{uuid.uuid4().hex[:8]}"
    defaults = ["--no-dashboard", "--output-dir", str(REPO / "outputs/training"), "--run.name", run_name]
    if preset == "@":
        if not overrides:
            raise ValueError("train @ requires a config path")
        files = [Path(overrides[0]).resolve()]
        overrides = overrides[1:]
    else:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; run tiny-prl presets")
        files = [EXAMPLES / name for name in PRESETS[preset][1]] + [EXAMPLES / "laptop.toml"]
    if environment and ENVIRONMENTS[environment]:
        files.append(EXAMPLES / ENVIRONMENTS[environment])
    return [arg for path in files for arg in ("@", str(path))] + defaults + overrides


def cached_model(name, cache):
    """Resolve the user's single cached snapshot when HF refs/main is absent."""
    if Path(name).exists():
        return str(Path(name).resolve())
    directory = cache / ("models--" + name.replace("/", "--"))
    if (directory / "refs/main").exists():
        return name
    snapshots = sorted(p.parent for p in (directory / "snapshots").glob("*/config.json"))
    if len(snapshots) == 1:
        return str(snapshots[0])
    raise ValueError(
        f"No unambiguous cached snapshot for {name}. Pass --model.name /path/to/model, or set HF_HUB_OFFLINE=0 to download."
    )


def preflight():
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < 8 * 1024**3:
        raise ValueError(f"Need 8 GiB available host RAM; currently {available / 1024**3:.2f} GiB")
    apps = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    if apps:
        raise ValueError(f"GPU 0 has active compute processes: {apps}")
    gpu = subprocess.check_output(
        ["nvidia-smi", "-i", "0", "--query-gpu=name,memory.free,compute_cap", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    name, free, capability = [v.strip() for v in gpu.split(",")]
    if int(free) < 7680:
        raise ValueError(f"Need 7680 MiB free GPU memory; currently {free} MiB")
    return {"available_host_bytes": available, "gpu": name, "free_gpu_mib": int(free), "compute_capability": capability}


def build_kernels(env, model="smollm2-360m"):
    command = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--expand-environment=no",
        "--property=MemoryMax=2G",
        "--property=MemorySwapMax=0",
        "--property=CPUQuota=100%",
        "--property=RuntimeMaxSec=120",
        shutil.which("uv"),
        "run",
        "--offline",
        "--no-project",
        "--python",
        sys.executable,
        "python",
        "-c",
        (
            "from prime_kernels.decode_megakernel import _library; _library(); print('135M kernel ready')"
            if model == "smollm2-135m"
            else "from prime_kernels.lossless_bf16 import _load_library; "
            "from prime_kernels.decode_megakernel_360m_lossless import _library; "
            "_load_library(); _library(); print('Lossless kernels ready')"
        ),
    ]
    subprocess.run(command, env={**env, "CUDA_VISIBLE_DEVICES": ""}, check=True)


def main():
    parser = argparse.ArgumentParser(prog="tiny-prl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("presets", help="list tested laptop configurations")
    sub.add_parser("doctor", help="check environment, tools and free resources")
    sub.add_parser("build-kernels", help="compile/cache the lossless kernels without GPU allocation")
    init = sub.add_parser("initialize", help="choose a model and environment; create an editable training workspace")
    init.add_argument("folder", type=Path, nargs="?", default=Path.cwd())
    init.add_argument("--model", choices=["qwen3-600m", "smollm2-135m", "smollm2-360m"])
    init.add_argument("--env")
    init.add_argument("--hf-repo")
    hub = sub.add_parser("env", help="prepare a supported Hub environment without changing the pinned venv")
    hub.add_argument("action", choices=["add"])
    hub.add_argument("environment", choices=["primeintellect/wordle"])
    upload = sub.add_parser("upload", help="retry a finished run's HF upload, verify it, then remove local weights")
    upload.add_argument("output", type=Path)
    upload.add_argument("--hf-repo", required=True, help="explicit Hugging Face model repository, owner/name")
    train = sub.add_parser("train", help="launch guarded RL; remaining arguments go to Prime-RL")
    train.add_argument("preset", nargs="?", help="model preset; omit inside an initialized workspace")
    train.add_argument("--env", help="reviewed environment, e.g. primeintellect:gsm8k")
    train.add_argument(
        "--output", type=Path, help="run folder; download base weights into an isolated disposable cache"
    )
    train.add_argument(
        "--hf-repo", help="upload final weights here, verify remote files, then remove local model files"
    )
    train.add_argument(
        "--guard-timeout", type=int, default=3600, help="maximum run duration in seconds (default: 3600)"
    )
    train.add_argument("--dry-run", action="store_true", help="resolve and write native configs without GPU work")
    argv = sys.argv[1:]
    if argv[:1] == ["train"] and (len(argv) == 1 or argv[1].startswith("-")):
        argv.insert(1, "__workspace__")
    args, overrides = parser.parse_known_args(argv)
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    if args.command == "initialize":
        if overrides:
            parser.error("initialize takes no native training overrides")
        from tiny_workspace import initialize

        return initialize(args, environment(), build_kernels)
    if args.command == "presets":
        if overrides:
            parser.error("presets takes no additional arguments")
        for name, (description, _) in PRESETS.items():
            print(f"{name:24} {description}")
        return 0
    if args.command == "upload":
        if overrides:
            parser.error("upload takes no additional arguments")
        from tiny_hf import finish_run

        GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
        with GPU_LOCK.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return finish_run(args.output.resolve(), args.hf_repo, environment(), run_guarded)
    if args.command == "build-kernels":
        if overrides:
            parser.error("build-kernels takes no additional arguments")
        build_kernels(environment())
        return 0
    if args.command == "env":
        if overrides:
            parser.error("env add takes no additional arguments")
        command = [
            "systemd-run",
            "--user",
            "--scope",
            "--quiet",
            "--collect",
            "--expand-environment=no",
            "--property=MemoryMax=2G",
            "--property=MemorySwapMax=0",
            "--property=CPUQuota=100%",
            "--property=RuntimeMaxSec=180",
            shutil.which("uv"),
            "run",
            "--offline",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(REPO / "tools/tiny_wordle_setup.py"),
        ]
        return subprocess.call(command, env=environment())
    if args.command == "doctor":
        if overrides:
            parser.error("doctor takes no additional arguments")
        print(f"Python: {sys.executable}\nCheckout: {REPO}\nKernels: {REPO / 'deps/prime-kernels'}")
        for name in ("torch", "vllm", "transformers", "prime-rl"):
            print(f"{name}: {importlib.metadata.version(name)}")
        for name in ("uv", "systemd-run", "nvidia-smi", "nvcc"):
            if shutil.which(name) is None:
                raise ValueError(f"Missing command: {name}")
        print(json.dumps(preflight(), indent=2))
        return 0
    if args.guard_timeout < 1:
        parser.error("--guard-timeout must be positive")
    workspace = Path.cwd() / "tiny-prl.json"
    if args.output and (args.output / "tiny-prl.json").is_file():
        workspace = args.output.resolve() / "tiny-prl.json"
        args.output = workspace.parent / "output"
    if workspace.is_file():
        settings = json.loads(workspace.read_text())
        if args.preset in (None, "__workspace__"):
            args.preset = settings["model"]
        args.env = args.env or settings["environment"]
        args.output = args.output or Path.cwd() / "output"
        args.hf_repo = args.hf_repo or settings.get("hf_repo")
        defaults = ["--orchestrator.train.algo.advantage-fn", "algo.compute_advantages"]
        if settings.get("decode_megakernel") and args.preset == settings["model"]:
            defaults += ["--inference.decode-megakernel"]
        overrides = defaults + overrides
    elif args.preset in (None, "__workspace__"):
        parser.error("Choose a model, or run tiny-prl initialize first")
    args.preset = {**{v: k for k, v in MODEL_IDS.items()}, "qwen3-600m": "qwen3-600m-lora"}.get(
        args.preset, args.preset
    )
    if args.env:
        args.env = args.env.replace(":", "/", 1)
        if args.env not in ENVIRONMENTS:
            parser.error("Supported environments: " + ", ".join(ENVIRONMENTS))
    if args.hf_repo and not args.output:
        parser.error("--hf-repo requires --output so cleanup has an explicit run folder")
    env = environment()
    if workspace.is_file() and args.preset:
        env["PYTHONPATH"] = str(workspace.parent) + os.pathsep + env["PYTHONPATH"]
    if args.output:
        env["HF_HUB_OFFLINE"] = "0"
        overrides += ["--output-dir", str(args.output.resolve().parent), "--run.name", args.output.resolve().name]
    if args.env == "primeintellect/wordle":
        directory = REPO / "outputs/environments/primeintellect-wordle-0.1.7"
        if not (directory / "dependencies/textarena").is_dir():
            setup = subprocess.run([sys.executable, str(REPO / "tools/tiny_wordle_setup.py")], env=env, check=False)
            if setup.returncode:
                raise ValueError("Wordle setup failed; retry tiny-prl env add primeintellect/wordle")
        env["PYTHONPATH"] += os.pathsep + str(directory / "dependencies")
        env["NLTK_DATA"] = str(directory / "nltk_data")
    os.environ.update(env)
    for directory in reversed(env["PYTHONPATH"].split(os.pathsep)):
        sys.path.insert(0, directory)
    from pydantic_config import cli

    from prime_rl.configs.rl import RLConfig

    flags = config_args(args.preset, overrides, args.env)
    config = cli(RLConfig, args=flags)
    if args.hf_repo and config.ckpt is None:
        raise ValueError("--hf-repo requires checkpoint saving; remove --no-ckpt")
    if args.output and config.run_dir.resolve() != args.output.resolve():
        raise ValueError("--output cannot be combined with a conflicting --run.dir override")
    if env["HF_HUB_OFFLINE"] == "1":
        cache = Path(
            env.get("HF_HUB_CACHE", str(Path(env.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"))
        )
        local = cached_model(config.model.name, cache)
        if local != config.model.name:
            flags += ["--model.name", local]
            config = cli(RLConfig, args=flags)
    command = [shutil.which("uv"), "run", "--offline", "--no-project", "--python", sys.executable, "rl", *flags]
    if args.dry_run:
        print(shlex.join(command + ["--dry-run"]), flush=True)
        return subprocess.call(command + ["--dry-run"], env=env)
    # The same lock is used by existing guarded research runs.
    lock = GPU_LOCK
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another tiny-prl GPU workload holds the lock") from None
        resources = preflight()
        if config.inference and config.inference.env_vars.get("PRIME_DECODE_MEGAKERNEL_360M_LOSSLESS") == "1":
            if resources["compute_capability"] != "8.9":
                raise ValueError("The lossless preset requires SM 8.9; use smollm2-360m-stock on other GPUs")
            build_kernels(env)
            resources = preflight()
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(config.output_dir).free < 8 * 1024**3:
            raise ValueError("Need 8 GiB free disk for training/checkpoints")
        if args.output:
            from tiny_hf import prepare_run

            model = MODEL_IDS.get(args.preset)
            if model is None:
                raise ValueError(
                    "--output supports the three tested model presets; use native configs for other models"
                )
            local = prepare_run(config.run_dir, model, config.max_steps, args.hf_repo, args.env)
            if workspace.is_file():
                shutil.copy2(workspace.parent / "algo.py", config.run_dir / "configs/algo.py")
            flags += ["--model.name", str(local)]
            command = [shutil.which("uv"), "run", "--offline", "--no-project", "--python", sys.executable, "rl", *flags]
        guard = config.output_dir / ".tiny-prl" / uuid.uuid4().hex
        guard.mkdir(parents=True)
        (guard / "launch.json").write_text(
            json.dumps(
                {
                    "command": command,
                    "resources": resources,
                    "run_dir": str(config.run_dir),
                    "guard_timeout": args.guard_timeout,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Run: {config.run_dir}\nGuard: {guard}\nStop: Ctrl-C", flush=True)
        result = run_guarded(command, guard, args.guard_timeout, env)
        if result == 0 and args.hf_repo:
            from tiny_hf import finish_run

            return finish_run(config.run_dir, args.hf_repo, env, run_guarded)
        return result


def run_guarded(command, guard, timeout, env):
    guard.mkdir(parents=True, exist_ok=True)
    scope = [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        "--expand-environment=no",
        "--unit=tiny-prl-" + guard.name,
        "--property=MemoryMax=7G",
        "--property=MemorySwapMax=0",
        f"--property=RuntimeMaxSec={timeout + 30}",
        sys.executable,
        str(REPO / "tools/tiny_guard.py"),
        str(guard),
        str(timeout),
        *command,
    ]
    return subprocess.call(scope, env=env)


if __name__ == "__main__":
    bootstrap = environment()
    os.environ.update(bootstrap)
    for directory in reversed(bootstrap["PYTHONPATH"].split(os.pathsep)):
        sys.path.insert(0, directory)
    from pydantic_config.cli import ConfigFileError

    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError, ConfigFileError) as error:
        print(f"tiny-prl: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        raise SystemExit(130)
