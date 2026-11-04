"""Install runtime source outside the checkout, reusing the pinned home venv."""

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = Path.home() / ".local/share/tiny-prl"
    staging = destination.with_name("tiny-prl-installing")
    if staging.exists():
        raise ValueError(f"Staging directory already exists: {staging}")
    venv = Path(os.environ.get("TINY_PRL_VENV", str(Path.home() / ".venv"))).resolve()
    if not (venv / "bin/python").is_file():
        raise ValueError(f"Pinned Python missing: {venv}")
    sources = [
        "src",
        "packages/prime-rl-configs/src",
        "examples/tiny",
        "deps/verifiers/verifiers",
        "deps/verifiers/environments",
        "deps/renderers/renderers",
        "deps/pydantic-config/src",
        "deps/prime-kernels/prime_kernels",
        "outputs/environments/primeintellect-wordle-0.1.7",
    ]
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".venv", "hillclimb*.png", "hillclimb*.pdf")
    for name in sources:
        source = ROOT / name
        if source.exists():
            shutil.copytree(source, staging / name, ignore=ignore)
        elif not name.startswith("outputs/"):
            raise ValueError(f"Initialize the pinned dependency before installing: {source}")
    for name in [
        "tiny_prl.py",
        "tiny_guard.py",
        "tiny_hf.py",
        "tiny_workspace.py",
        "tiny_wordle_setup.py",
        "convert_dcp_to_bf16.py",
    ]:
        target = staging / "tools" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "tools" / name, target)
    for name in [
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE",
        "docs/tiny-prl.md",
        "docs/tiny-prl-cleanup-handoff.md",
    ]:
        if (ROOT / name).is_file():
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, staging / name)
    (staging / "installation.json").write_text(
        json.dumps(
            {
                "source_revision": subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
                "venv": str(venv),
                "runtime": str(destination),
            },
            indent=2,
        )
        + "\n"
    )
    if destination.exists():
        if not (destination / "installation.json").is_file():
            raise ValueError("Refusing to replace an unmanaged runtime directory")
        shutil.rmtree(destination)
    staging.rename(destination)
    command = Path.home() / ".local/bin/tiny-prl"
    if command.is_symlink():
        command.unlink()
    elif command.exists() and "tiny-prl" not in command.read_text():
        raise ValueError("Refusing to replace an unrelated executable")
    command.parent.mkdir(parents=True, exist_ok=True)
    command.write_text(
        "#!/bin/sh\nset -eu\n"
        f'export UV_CACHE_DIR="${{UV_CACHE_DIR:-$HOME/.cache/tiny-prl/uv}}"\n'
        f"exec uv run --offline --no-project --python {shlex.quote(str(venv / 'bin/python'))} "
        f'python {shlex.quote(str(destination / "tools/tiny_prl.py"))} "$@"\n'
    )
    command.chmod(0o755)
    print(f"Installed {command}\nRuntime: {destination}\nKeep the pinned environment: {venv}")


if __name__ == "__main__":
    main()
