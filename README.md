# tiny-prl

A [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl) fork for RL experiments
on small NVIDIA GPUs. The current local workflow has been tested on an
**RTX 4060 Laptop GPU with 8 GiB VRAM and about 15 GiB system RAM**.

The stack is **PyTorch/FSDP2 for training, vLLM for inference, and Verifiers for
environments and rewards**. The laptop launcher adds model presets, resource
checks, checkpoint/resume commands, and opt-in kernels from the pinned
[prime-kernels fork](https://github.com/Mantissagithub/prime-kernels).

The verified path is short-context SmolLM2-360M full-parameter RL. A local
Wordle integration also runs, but the first trial did **not** demonstrate
useful learning.

## Run locally

The launcher requires an existing compatible Python 3.12 environment, Linux
with a systemd user session, NVIDIA drivers, CUDA/nvcc, ninja, and `uv`.
The tested environment contains Torch **2.13.0+cu130**, vLLM **0.31.0**, and
Transformers **5.6.2**.

For a new checkout:

```bash
git clone https://github.com/Mantissagithub/tiny-prl.git
cd tiny-prl
git submodule update --init -- deps/verifiers deps/renderers deps/pydantic-config deps/prime-kernels
```

The command uses `$HOME/.venv` by default. To use another existing environment:

```bash
export TINY_PRL_VENV=/absolute/path/to/your/venv
```

It runs through `uv run --offline --no-project`; it does not install or
synchronize the training stack. A fresh clone alone is not a complete GPU
installation. Models must already be cached or supplied as local paths;
`HF_HUB_OFFLINE=0` allows model/dataset downloads.

```bash
./bin/tiny-prl presets
./bin/tiny-prl doctor
./bin/tiny-prl train smollm2-360m --dry-run
./bin/tiny-prl train smollm2-360m --max-steps 2 --run.name first-run
```

The default environment is alphabet-sort. Named presets save a final checkpoint
and retain the last checkpoint. To resume the same run:

```bash
./bin/tiny-prl train smollm2-360m --run.name first-run --resume --max-steps 3
```

Native Prime-RL config files and overrides are supported:

```bash
./bin/tiny-prl train @ /path/to/rl.toml --dry-run
./bin/tiny-prl train @ /path/to/rl.toml --max-steps 10 --run.name custom-run
```

Outputs default to `outputs/training/<run-name>/`, with native logs, resolved
configs, metrics and DCP checkpoints. `--output-dir` overrides the output root.
Custom configs retain their own checkpoint settings.

To use `tiny-prl` from any directory, link the launcher from the checkout:

```bash
mkdir -p ~/.local/bin
ln -s "$(pwd)/bin/tiny-prl" ~/.local/bin/tiny-prl
```

See [the command guide](examples/tiny/README.md) for checkpoint export,
longer run timeouts, model-cache handling and dashboard options.

## Available presets

| Preset | Trainer | Inference / weight transfer | Default updates |
|---|---|---|---|
| `smollm2-360m` | Full parameters, FP32 optimization/reduction, full activation checkpointing, chunked optimizer offload | vLLM with opt-in lossless SM89 decode / stream64 filesystem updates | 2 |
| `smollm2-360m-stock` | Same 360M trainer | Stock vLLM / stream64 filesystem updates | 2 |
| `smollm2-135m` | Full parameters, FP32 optimization/reduction, RoPE/RMSNorm kernels | Stock vLLM / CUDA IPC, early trainer startup | 10 |
| `qwen3-600m-lora` | LoRA rank 8 | Stock vLLM / filesystem updates | 10 |

These configs use 512-token sequences, up to two inference sequences, and small
training batches. The packaged command was exercised end to end on the 360M
lossless preset. The other entries expose existing configs; they are not new
capacity or performance validations. No 1B full-training or long-context coding
capacity claim is established here.

## Verified results

These measurements describe the recorded workloads on this laptop.

| Check | Recorded result | Scope |
|---|---|---|
| 360M full-parameter alphabet-sort RL through the CLI | Two updates, checkpoint saved; resume loaded step 2, completed step 3 and saved again | Functional training/checkpoint validation |
| GPU memory in those two launches | 7,224 MiB and 7,218 MiB peak; zero job-cgroup swap | The tested short-context config |
| B1 lossless decoder change, Experiment 233 | vLLM generation latency fell **0.75–0.88%** versus the previous lossless package | Five fresh processes per arm; fixed 64-token generation at prompt lengths 32/128/1024; identical tokens and sampled-token logprobs |
| B2 in Experiment 233 | Machine code unchanged; serving timing ranges overlap | No B2 improvement claimed |
| Latest kernel's whole-RL benefit | No verified additional RL wall-time gain | Functional runs performed different rollout work and cannot establish a speedup |

The cached decoder's B1/B2 instructions match the tested Experiment 233 variant.
The lossless path targets **SmolLM2-360M on SM89, batches 1–2**. It preserves
weight bits and floating-point operations; its exact comparisons are against
the preceding lossless implementation, not a claim of universal native-vLLM
parity. Unsupported steps use the existing inference fallback.

Evidence: [CLI validation](examples/tiny/cli-validation.json),
[experiment ledger](examples/tiny/EXPERIMENTS.md), and the
[full hillclimb figure](examples/tiny/hillclimb.png). Component, serving, and
verified RL wall-time results are recorded separately.

## Prime Environments Hub: Wordle

The supported Hub package is **`primeintellect/wordle@0.1.7`**. Its legacy API
is adapted to the pinned taskset API, preserving its prompt, parser and four
reward components. Setup verifies the reviewed source hash and installs the
optional TextArena/NLTK packages into a local directory, leaving the pinned
training venv unchanged.

```bash
./bin/tiny-prl env add primeintellect/wordle
./bin/tiny-prl train smollm2-360m --env primeintellect/wordle --dry-run
./bin/tiny-prl train smollm2-360m --env primeintellect/wordle \
  --max-steps 1 --run.name wordle-first --guard-timeout 240
```

The recorded trial found:

- **2,010** parser/reward fixtures matched the original Hub functions and the
  legacy parser reference.
- Four local evaluation episodes completed without infrastructure errors:
  **0 wins**, all ending in invalid guesses.
- One full-parameter RL update and checkpoint save completed. Finalized metrics
  recorded 32 rollouts, 28 discarded for no signal, and four admitted traces.
- Training peak GPU memory was **7,170 MiB**, with zero job swap. Gradient norm
  was **1.66e-8**: this near-zero gradient does not demonstrate useful learning.

This trial used 32 generated tokens per turn and a 512-token context; it is not
a full-budget Wordle capability benchmark. Formatting reward alone can be
nonzero for malformed text in the original package. Generic Hub packages and
Harbor tasks are not yet wired into this launcher.

See [Wordle setup and limitations](examples/tiny/WORDLE.md) and
[the validation report](examples/tiny/wordle-validation.json).

## Resource limits

Training preflight requires 8 GiB available host RAM, 7,680 MiB free on GPU 0,
no active GPU compute process, and 8 GiB free disk. Only one guarded GPU workload
runs at a time. The training scope caps host memory at 7 GiB and job swap at
zero; a monitor checks host RAM, disk, GPU usage and the run timeout.

Lossless kernels build before GPU allocation in a separate one-CPU, 2 GiB
scope and are cached under `outputs/kernel-cache/` by default. The training
timeout is one hour; `--guard-timeout` changes it. These limits reduce resource
pressure; they do not guarantee arbitrary model/context/batch overrides will fit.

## Repository and upstream

This repository contains the trainer fork and laptop launcher.
`deps/prime-kernels` pins the separate kernel fork. Downloaded environments,
compiled libraries, model weights and run outputs are not committed.

Prime-RL's broader framework guides remain in [docs/](docs). Their distributed
model support is separate from this fork's tested laptop configurations.
Upstream projects: [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl),
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers), and
[Prime Kernels](https://github.com/PrimeIntellect-ai/prime-kernels).
