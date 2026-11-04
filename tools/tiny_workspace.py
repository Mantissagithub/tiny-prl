"""Create a model-specific workspace with an editable advantage function."""

import json
import os
import sys

from tiny_hf import prepare_run

ALGORITHM = '''"""Edit compute_advantages to change group credit assignment.

Inputs are one-dimensional CPU FP32 tensors. Return one finite scalar per
rollout. The starter delegates to the tested, length-weighted GRPO baseline.
"""
from prime_rl.orchestrator.algo.grpo import default_advantages


def compute_advantages(rewards, lengths):
    return default_advantages(rewards, lengths)
'''


def choose(prompt, choices):
    for i, (name, description) in enumerate(choices, 1):
        print(f"  {i}. {name}: {description}")
    answer = input(f"{prompt} [1]: ").strip() or "1"
    if not answer.isdigit() or not 1 <= int(answer) <= len(choices):
        raise ValueError("Choose one of the displayed numbers")
    return choices[int(answer) - 1][0]


def initialize(args, env, build_kernels):
    from tiny_prl import ENVIRONMENTS, MODEL_IDS

    folder = args.folder.resolve()
    if (folder / "tiny-prl.json").exists() or (folder / "algo.py").exists():
        raise ValueError("Workspace already initialized; existing algo.py will not be overwritten")
    model = args.model or choose(
        "Model",
        [
            ("qwen3-600m", "LoRA rank 8; stock vLLM"),
            ("smollm2-135m", "full-parameter training; packaged decode megakernel"),
            ("smollm2-360m", "full-parameter training; lossless decode megakernel"),
        ],
    )
    environment = args.env or choose(
        "Environment",
        [
            ("primeintellect:gsm8k", "grade-school math; 1024-token profile, quality not yet measured"),
            ("primeintellect:reverse-text", "single-turn text reversal; integration verified"),
            ("primeintellect:alphabet-sort", "short sorting task; laptop baseline"),
            ("primeintellect:wordle", "multi-turn game; no measured wins yet"),
        ],
    )
    if environment.replace(":", "/", 1) not in ENVIRONMENTS:
        raise ValueError("Unsupported environment: " + environment)
    repo = args.hf_repo
    if repo is None and args.model is None:
        repo = input("HF destination owner/name, blank to decide later: ").strip() or None
    env["HF_HUB_OFFLINE"] = "0"
    os.environ.update(env)
    for path in reversed(env["PYTHONPATH"].split(os.pathsep)):
        sys.path.insert(0, path)
    preset = "qwen3-600m-lora" if model == "qwen3-600m" else model
    prepare_run(folder / "output", MODEL_IDS[preset], 10, repo, environment)
    if model != "qwen3-600m":
        build_kernels(env, model)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "algo.py").write_text(ALGORITHM)
    (folder / "tiny-prl.json").write_text(
        json.dumps(
            {"model": model, "environment": environment, "hf_repo": repo, "decode_megakernel": model != "qwen3-600m"},
            indent=2,
        )
        + "\n"
    )
    print(f"Workspace: {folder}\nEdit {folder / 'algo.py'} to change compute_advantages(rewards, lengths).")
    print("From that folder, run tiny-prl train --max-steps 100 --hf-repo owner/name")
    print("A verified upload removes local model files. Without an HF destination, weights stay local.")
    return 0
