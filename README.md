# tiny-prl

I'm making RL training work on my **RTX 4060 laptop: 8 GiB VRAM and about
15 GiB RAM**. This is my work-in-progress fork of
[Prime-RL](https://github.com/PrimeIntellect-ai/prime-rl), mostly built with
Codex coding agents.

The stack is **PyTorch/FSDP2 for training, vLLM for inference, and Verifiers for
environments and rewards**. My separate
[prime-kernels fork](https://github.com/Mantissagithub/prime-kernels) is pinned
under `deps/prime-kernels`.

## Run

The launcher needs Linux/systemd, NVIDIA drivers, CUDA/nvcc, ninja, `uv`, a
compatible training environment and cached models. My pinned stack uses Python
3.12, Torch 2.13.0+cu130, vLLM 0.31.0 and Transformers 5.6.2. A clone alone
doesn't install it.

```bash
git clone https://github.com/Mantissagithub/tiny-prl.git
cd tiny-prl
git submodule update --init -- deps/verifiers deps/renderers deps/pydantic-config deps/prime-kernels

# Defaults to ~/.venv; override if needed.
export TINY_PRL_VENV="$HOME/.venv"
./bin/tiny-prl doctor
./bin/tiny-prl train smollm2-360m --dry-run
./bin/tiny-prl train smollm2-360m --max-steps 2 --run.name first-run
```

The default task is short-context alphabet-sort. The launcher checks resources,
limits job RAM to 7 GiB with zero job swap, and saves checkpoints. Outputs go to
`outputs/training/<run-name>/`. These guards don't guarantee every configuration
will fit.

| Preset | Training |
|---|---|
| `smollm2-360m` | Full parameters, opt-in lossless SM89 decode |
| `smollm2-360m-stock` | Full parameters, stock vLLM |
| `smollm2-135m` | Full parameters, CUDA IPC weight updates |
| `qwen3-600m-lora` | LoRA rank 8, stock vLLM |

I've verified the packaged 360M command through training, checkpoint saving and
resume. The other presets expose existing configs; 1B full training and long
contexts still need capacity tests. See the [command guide](examples/tiny/README.md)
for setup, resume, export and custom configs.

## Progress

- **Training works; Wordle solving hasn't improved yet.** The original 360M,
  50-step SFT and SFT-plus-two-RL checkpoints all scored **0% pass@1 and pass@4**
  over 20 evaluation words with four attempts each. A longer run stalled on
  invalid guesses. I've fixed a GRPO rounding bug that admitted tied-reward
  groups as trainable. [Wordle results and diagnosis](research/wordle-20261103/README.md).
- **Measured startup gain:** 62.55 → 55.20 seconds, or 11.74% lower median wall
  time, in matched two-update checkpointed runs with an opt-in startup profile.
  This doesn't establish a longer-training speedup.
- **Measured decoder gain:** 0.75–0.88% lower B1 serving latency against the
  preceding lossless decoder. No additional kernel-only whole-RL gain verified.

For the Hub adapter, start with:

```bash
./bin/tiny-prl env add primeintellect/wordle
./bin/tiny-prl train smollm2-360m --env primeintellect/wordle --dry-run
```

I keep detailed measurements in the [experiment ledger](examples/tiny/EXPERIMENTS.md)
and [hillclimb graph](examples/tiny/hillclimb.png). See [Wordle setup](examples/tiny/WORDLE.md)
for its unchanged Hub rewards and current limitations, or [docs/](docs) for the
broader Prime-RL framework. Weights, downloaded environments and bulk run outputs
stay out of Git.
