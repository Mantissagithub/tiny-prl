# Laptop training commands

`tiny-prl` launches the existing Prime-RL pipeline with the laptop presets, your
pinned Python environment, and resource guards. It uses this checkout and the
`deps/prime-kernels` submodule; private experiment scripts are not required.

From the repository:

```bash
./bin/tiny-prl presets
./bin/tiny-prl doctor
./bin/tiny-prl train smollm2-360m --dry-run
./bin/tiny-prl train smollm2-360m --max-steps 100 --run.name my-run
```

For a command available from any directory, link the tracked launcher once:

```bash
mkdir -p ~/.local/bin
ln -s "$(pwd)/bin/tiny-prl" ~/.local/bin/tiny-prl
```

The launcher uses `uv run --offline --no-project` with `$HOME/.venv`, preserving
the installed stack. Set `TINY_PRL_VENV` to select another existing environment.
It does not install or synchronize dependencies. Linux, a systemd user session,
CUDA/nvcc, ninja, and the installed Prime-RL stack are required. This machine has
Torch 2.13.0+cu130, vLLM 0.31.0 and Transformers 5.6.2.

| Preset | Training | Inference and weight transfer | Default updates |
|---|---|---|---|
| `smollm2-360m` | Full parameters, FP32 optimization/reduction, activation checkpointing, chunked optimizer offload | vLLM with opt-in lossless SM89 decode, stream64 filesystem updates | 2 |
| `smollm2-360m-stock` | Same 360M trainer | Stock vLLM, stream64 filesystem updates | 2 |
| `smollm2-135m` | Full parameters, FP32 optimization/reduction, RoPE/RMSNorm kernels | Stock vLLM, CUDA IPC, early trainer startup | 10 |
| `qwen3-600m-lora` | LoRA rank 8 | Stock vLLM, filesystem updates | 10 |

These are short-context alphabet-sort RL presets: 512-token sequences, up to
64 generated tokens, two inference sequences, and four usable traces/update.
`--max-steps` sets the number of updates. They do not configure math or coding
training automatically. The model must be cached or supplied as a local path.
The new command is validated end to end on the 360M lossless preset; the other
entries expose the existing configs, not new capacity or performance claims.

For the reviewed `primeintellect/wordle` Hub package, see [Wordle setup and trial results](WORDLE.md).

Experiment 235 adds an opt-in B1 profile with early trainer startup:

```bash
./bin/tiny-prl train smollm2-360m @ examples/tiny/smollm2-360m-b1-early.toml --dry-run
```

It overlaps trainer imports with inference startup, limits inference to one
sequence and fixes episode concurrency at two. In the private matched-work
one-update benchmark, five processes per arm gave medians 53.56 to 46.71 s;
both ABBA blocks passed, but individual timing ranges overlap. The benchmark
disabled checkpoint saving and used distinct deterministic request seeds;
normal training retains its checkpoint and stochastic sampling settings.
This does not establish a gain over the default two-sequence preset. See
[Experiment 235](EXPERIMENTS.md#experiment-235-startup-overlap-reduces-matched-work-b1-rl-wall-time)
for the scope and raw evidence location.

For the tested short-context 360M workload, native batch-invariant serving is
available as an overlay:

```bash
./bin/tiny-prl train smollm2-360m @ examples/tiny/smollm2-360m-batch-invariant.toml --dry-run
```

It enables vLLM's invariant projections and attention. The lossless backend
also recomputes decode rows when a prefill joins their batch, keeping their
decode arithmetic consistent. This adds work to mixed steps. It retains two
sequence slots and checkpoint saving; normal sampling is still stochastic.
Fixed seeds alone do not guarantee identical training schedules or long-context
results. The default preset is unchanged.

To also overlap trainer startup:

```bash
./bin/tiny-prl train smollm2-360m @ examples/tiny/smollm2-360m-batch-invariant.toml --deployment.early-trainer-start
```

Experiment 238 compares startup modes with invariant serving and the mixed
repair in both arms. Five runs per arm gave medians 62.55 to 55.20 seconds,
including two updates and a full checkpoint. All early runs beat all controls;
finalized training work and both steps' metrics matched exactly. Shutdown-tail
work varied slightly and was included in timing. This is a short-run startup
result, not an isolated kernel or long-run throughput gain. Some discarded
raw rollout logprobs still vary. See [Experiment 238](EXPERIMENTS.md#experiment-238-checkpointed-b2-capable-rl-startup-gain-confirmed).

Normal Prime-RL arguments pass through:

```bash
# Keep a predictable run name and retain the last checkpoint.
tiny-prl train smollm2-360m --max-steps 100 --run.name my-run

# Resolve your own task/model/training configuration.
tiny-prl train @ /path/to/my-rl.toml --dry-run

tiny-prl train @ /path/to/my-rl.toml --max-steps 100 --run.name custom-run

# Resume a saved named run, using the same preset and output directory.
tiny-prl train smollm2-360m --run.name my-run --resume --max-steps 200

# Disable checkpoint saving only when you want a disposable benchmark.
tiny-prl train smollm2-360m --no-ckpt
```

Named presets layer `laptop.toml` to enable final checkpoints, one file per
tensor, retaining the last checkpoint. Custom configs keep their own checkpoint
settings. `--no-dashboard` is the launcher default; pass `--dashboard` to enable
Prime-RL's dashboard. Outputs default to this checkout's `outputs/training/`;
`--output-dir` overrides it. Each new launch gets a unique run name unless you
pass `--run.name`. Native logs, resolved configs, metrics and checkpoints remain
in the normal `<output-dir>/<run-name>/` layout. Trainer checkpoints are DCP
checkpoints; use Prime-RL's `tools/convert_dcp_to_bf16.py` to export HF weights.

Cached models work offline by default, including this laptop's single cached
snapshot without `refs/main`. Ambiguous snapshots require an explicit
`--model.name /path/to/snapshot`. Set `HF_HUB_OFFLINE=0` if you intend to download
models or datasets. The launcher does not delete or modify cached models.

The 360M lossless kernels compile automatically before GPU allocation, under a
separate one-CPU/2 GiB build limit. To prepare them explicitly:

```bash
tiny-prl build-kernels
```

They use Torch's extension cache, defaulting in this launcher to
`outputs/kernel-cache/`; `TORCH_EXTENSIONS_DIR` overrides it. CUDA sources and
the codec header are included in the kernel package. The lossless decode path
is specific to SmolLM2-360M on SM89, with batch sizes 1–2; unsupported shapes
follow the existing inference fallback. Other GPUs should use the stock preset.

One GPU job runs at a time, sharing the research runner's lock. Before training,
the command requires 8 GiB available host RAM, 7,680 MiB free GPU memory, no
active GPU compute process and 8 GiB free disk. The training scope caps host
memory at 7 GiB and job swap at zero. Its monitor stops the job below 2 GiB
available host RAM/free disk, above 7,676 MiB GPU usage, or on timeout. These
limits reduce crash risk; changing batch size/context/model can still fail.

The default timeout is one hour. For longer training:

```bash
tiny-prl train smollm2-360m --max-steps 1000 --guard-timeout 14400
```

The command prints the run directory and guard directory. Ctrl-C stops the run.
Guard launch metadata, resource samples and the exit result are stored under
`<output-dir>/.tiny-prl/<id>/`. A killed/OOM scope can exit before writing its
final result; a missing result is not a successful run. Native `rl @ config.toml`
remains available if you want to operate Prime-RL directly.

Validation results are recorded in [cli-validation.json](cli-validation.json):
five launcher tests, native dry runs, a fresh kernel build, instruction identity
with Experiment 233, real two-update training with a final checkpoint, and
resume from step 2 through step 3 with another checkpoint. Both RL guards
passed with zero job swap. These are functional checks, not speed measurements.
