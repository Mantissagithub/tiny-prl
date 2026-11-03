"""Fetch the pinned Hub Wordle package and isolated optional dependencies."""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "outputs/environments/primeintellect-wordle-0.1.7"
SOURCE_SHA256 = "8567b975122280ef3e29105592dcd0bc0fd11dc4f065be65596c1de074f8d87b"


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    prefix = ["uv", "run", "--offline", "--no-project", "--python", sys.executable]
    source = DIRECTORY / "upstream/wordle.py"
    if not source.exists():
        subprocess.run(
            prefix + ["prime", "env", "pull", "primeintellect/wordle@0.1.7", "--target", str(source.parent), "--plain"],
            check=True,
            env={**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"},
        )
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Hub Wordle 0.1.7 source hash does not match the reviewed adapter")
    dependencies = DIRECTORY / "dependencies"
    versions = {
        d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions(path=[str(dependencies)])
    }
    expected = {"textarena": "0.7.4", "nltk": "3.9.2"}
    if any(versions.get(name) != version for name, version in expected.items()):
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(dependencies),
                "--no-deps",
                "--reinstall",
                "textarena==0.7.4",
                "nltk==3.9.2",
            ],
            check=True,
        )
    data = DIRECTORY / "nltk_data"
    data.mkdir(exist_ok=True)
    os.environ["NLTK_DATA"] = str(data)
    sys.path.insert(0, str(dependencies))
    import nltk

    for corpus in ("words", "averaged_perceptron_tagger_eng"):
        if not nltk.download(corpus, download_dir=str(data), quiet=True, raise_on_error=True):
            raise RuntimeError(f"Could not prepare NLTK corpus: {corpus}")
    (DIRECTORY / "manifest.json").write_text(
        json.dumps(
            {
                "hub_id": "primeintellect/wordle",
                "package_version": "0.1.7",
                "version_id": "eletiahb7kg3mkh4qbh6ixlw",
                "source_sha256": SOURCE_SHA256,
                "dependencies": expected,
            },
            indent=2,
        )
        + "\n"
    )
    print("Wordle ready. Run: tiny-prl train smollm2-360m --env primeintellect/wordle")


if __name__ == "__main__":
    main()
