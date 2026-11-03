# tiny-prl

I'm trying to make RL training work well on my laptop.

This is my work-in-progress fork of
[Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl), built around my
**RTX 4060 Laptop GPU with 8 GiB VRAM and about 15 GiB system RAM**. I want to
see how much I can get out of this hardware, from the training loop and weight
transfers down to the decode kernels. And most of it is driven by codex!

**Still in progress.** So far, I've run short-context SmolLM2-360M
full-parameter RL, saved checkpoints, and resumed training on this laptop.
I've also connected Wordle from the Prime Environments Hub. That integration
runs, but the first trial hasn't shown useful learning yet.

I'm using **PyTorch/FSDP2 for training, vLLM for inference, and Verifiers for
environments and rewards**. I've put the laptop configs behind a `tiny-prl`
command, with resource checks, checkpoint/resume support, and opt-in kernels
from my pinned
[prime-kernels fork](https://github.com/Mantissagithub/prime-kernels).

My approach is to keep a baseline, profile it, try one change, check correctness,
and measure again. I record the failed attempts too. A faster kernel only counts
as an RL wall-time improvement once I've measured that in the whole run.

## My setup and how to run it

My current environment uses Python 3.12, Torch **2.13.0+cu130**, vLLM **0.31.0**,
and Transformers **5.6.2**. The launcher needs Linux with a systemd user session,
NVIDIA drivers, CUDA/nvcc, ninja, `uv`, and an existing compatible training
environment.

For a new checkout:

```bash
git clone https://github.com/Mantissagithub/tiny-prl.git
cd tiny-prl
git submodule update --init -- deps/verifiers deps/renderers deps/pydantic-config deps/prime-kernels
```

I keep my pinned environment at `$HOME/.venv`, which is the launcher's default.
To use another existing environment:

```bash
export TINY_PRL_VENV=/absolute/path/to/your/venv
```

I run through `uv run --offline --no-project` to preserve that environment.
The launcher doesn't install or synchronize the training stack, so a fresh clone
still needs a compatible GPU environment. Models must already be cached or
supplied as local paths; `HF_HUB_OFFLINE=0` allows model/dataset downloads.

```bash
./bin/tiny-prl presets
./bin/tiny-prl doctor
./bin/tiny-prl train smollm2-360m --dry-run
./bin/tiny-prl train smollm2-360m --max-steps 2 --run.name first-run
```

I've been using alphabet-sort as the default environment. Named presets save a final checkpoint
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

## Presets I'm using

| Preset | Trainer | Inference / weight transfer | Default updates |
|---|---|---|---|
| `smollm2-360m` | Full parameters, FP32 optimization/reduction, full activation checkpointing, chunked optimizer offload | vLLM with opt-in lossless SM89 decode / stream64 filesystem updates | 2 |
| `smollm2-360m-stock` | Same 360M trainer | Stock vLLM / stream64 filesystem updates | 2 |
| `smollm2-135m` | Full parameters, FP32 optimization/reduction, RoPE/RMSNorm kernels | Stock vLLM / CUDA IPC, early trainer startup | 10 |
| `qwen3-600m-lora` | LoRA rank 8 | Stock vLLM / filesystem updates | 10 |

These configs use 512-token sequences, up to two inference sequences, and small
training batches. I've tested the packaged command end to end with the 360M
lossless preset. The other entries expose my existing configs; their presence
in this table isn't a new capacity or performance result. Full-parameter 1B
training and long-context coding still need separate capacity tests on my laptop.

## What I've verified so far

These are results from the recorded workloads on my laptop, with the scope of
each measurement kept explicit.

| Check | Recorded result | Scope |
|---|---|---|
| 360M full-parameter alphabet-sort RL through the CLI | Two updates, checkpoint saved; resume loaded step 2, completed step 3 and saved again | Functional training/checkpoint validation |
| GPU memory in those two launches | 7,224 MiB and 7,218 MiB peak; zero job-cgroup swap | The tested short-context config |
| B1 lossless decoder change, Experiment 233 | vLLM generation latency fell **0.75–0.88%** versus the previous lossless package | Five fresh processes per arm; fixed 64-token generation at prompt lengths 32/128/1024; identical tokens and sampled-token logprobs |
| B2 in Experiment 233 | Machine code unchanged; serving timing ranges overlap | No B2 improvement claimed |
| Kernel-only whole-RL benefit | No verified additional RL wall-time gain | A single matched pair cannot isolate the small kernel effect from timing variation |
| Early trainer startup with the latest kernel, Experiment 235 | Median **53.56 → 46.71 s (12.8% lower)**, five runs per arm | One-update B1-only RL with identical request/output hashes and training work; both ABBA blocks pass, but individual timing ranges overlap |
| Checkpointed two-sequence startup, Experiment 238 | Median **62.55 to 55.20 s (11.74% lower)**, five runs per arm | Two full updates with invariant serving and mixed-decode repair in both arms; finalized training work/metrics match exactly; every early run beats every control |

I've checked that the cached decoder's B1/B2 instructions match the tested
Experiment 233 variant. My lossless path targets
**SmolLM2-360M on SM89, batches 1–2**. It preserves
weight bits and floating-point operations; its exact comparisons are against
the preceding lossless implementation, not a claim of universal native-vLLM
parity. Unsupported steps use the existing inference fallback.

I keep the evidence in the [CLI validation](examples/tiny/cli-validation.json),
[experiment ledger](examples/tiny/EXPERIMENTS.md), and
[full hillclimb figure](examples/tiny/hillclimb.png). Component, serving, and
verified RL wall-time results have separate records.

I've kept the startup result as an [opt-in B1 profile](examples/tiny/smollm2-360m-b1-early.toml).
It compares serial and overlapping startup on the same kernel. The gain varied
between blocks. I've now also confirmed startup overlap for a two-sequence,
two-update run with checkpoints, using the [batch-invariant overlay](examples/tiny/smollm2-360m-batch-invariant.toml)
and the mixed-decode repair. That profile's medians are 62.55 to 55.20 s, with
exact finalized training metrics. Neither result establishes a gain over the
unchanged default preset or longer training. Some discarded raw rollout
logprobs still vary. My ten-update follow-up completed in both startup modes,
but their training diverged at update 5 and generated work differed, so I
haven't verified a longer-run speedup. The B2 integer-add trial in
Experiment 234 had no decoder gain, so I rejected it before spending more runs on serving.

## Trying Wordle from the Prime Environments Hub

I've connected **`primeintellect/wordle@0.1.7`**. I adapted its legacy API to
the pinned taskset API, keeping its prompt, parser and four reward components.
Setup verifies the reviewed source hash and installs the optional TextArena/NLTK
packages into a local directory, leaving my pinned training venv unchanged.

```bash
./bin/tiny-prl env add primeintellect/wordle
./bin/tiny-prl train smollm2-360m --env primeintellect/wordle --dry-run
./bin/tiny-prl train smollm2-360m --env primeintellect/wordle \
  --max-steps 1 --run.name wordle-first --guard-timeout 240
```

Here's what happened in my first bounded trial:

- **2,010** parser/reward fixtures matched the original Hub functions and the
  legacy parser reference.
- Four local evaluation episodes completed without infrastructure errors:
  **0 wins**, all ending in invalid guesses.
- One full-parameter RL update and checkpoint save completed. Finalized metrics
  recorded 32 rollouts, 28 discarded for no signal, and four admitted traces.
- Training peak GPU memory was **7,170 MiB**, with zero job swap. Gradient norm
  was **1.66e-8**: this near-zero gradient does not demonstrate useful learning.

I used 32 generated tokens per turn and a 512-token context, so this doesn't
establish how the model would perform with a larger budget. The original
package also gives some formatting reward to malformed text. I still need to
establish valid guesses and meaningful reward contrast before calling this
useful Wordle training. Generic Hub packages and Harbor tasks are still on the
integration list.

See [Wordle setup and limitations](examples/tiny/WORDLE.md) and
[the validation report](examples/tiny/wordle-validation.json).

## Keeping my laptop usable

I've added resource checks because the trainer and inference engine share my
GPU. Training preflight requires 8 GiB available host RAM, 7,680 MiB free on GPU 0,
no active GPU compute process, and 8 GiB free disk. Only one guarded GPU workload
runs at a time. The training scope caps host memory at 7 GiB and job swap at
zero; a monitor checks host RAM, disk, GPU usage and the run timeout.

Lossless kernels build before GPU allocation in a separate one-CPU, 2 GiB
scope and are cached under `outputs/kernel-cache/` by default. The training
timeout is one hour; `--guard-timeout` changes it. These limits reduce resource
pressure; they do not guarantee arbitrary model/context/batch overrides will fit.

## How I've organized the repo

I keep the trainer fork and laptop launcher here, with `deps/prime-kernels`
pinning my separate kernel fork. I leave downloaded environments, compiled
libraries, model weights and run outputs out of Git.

Prime-RL's broader framework guides remain in [docs/](docs). I'm only making
laptop-specific claims for the configurations I've actually tested here.
Upstream projects: [Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl),
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers), and
[Prime Kernels](https://github.com/PrimeIntellect-ai/prime-kernels).
