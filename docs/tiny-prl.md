# Tiny single-GPU RL baseline

## Workspace layout

The training fork and kernel fork are separate repositories on `main`:

```text
tiny-prl/
  prime-rl/       # Mantissagithub/tiny-prl
  prime-kernels/  # Mantissagithub/prime-kernels
```

Run the commands below from `prime-rl/`. Each repository's `origin` points to
the personal fork; `upstream` points to the corresponding Prime Intellect
repository. Develop kernels in the sibling checkout, then update the pinned
`deps/prime-kernels` submodule after testing the kernel commit. The submodule
remains the reproducible kernel source dependency for the training repository.

## Baseline

[`examples/tiny/rl.toml`](../examples/tiny/rl.toml) is a small starting run for
Qwen3-0.6B with LoRA on one 8 GiB GPU. It uses the deterministic, seeded
`alphabet-sort` taskset with one turn and two names, a 64-token completion
limit, orchestrator batch size 4, and group size 2. The trainer uses a
128-token fused LM-head chunk and disables `torch.compile` for this first run.

The trainer and vLLM stay resident concurrently on the same physical GPU. The
launcher waits for vLLM's engine liveness check before starting the trainer, so
trainer allocations do not interfere with vLLM's startup memory profile. The
launcher downloads model files without importing the trainer's Torch runtime.
In shared-GPU mode, the launcher starts one trainer process directly after
inference is live; the trainer initializes its one-rank distributed process
group from the environment and writes output to `trainer.log`. This mode has no
torchrun per-rank logs.
The tiny profile runs its environment server inline with one worker, avoiding a
separate broker process while keeping the task's network protocol unchanged.
This recipe uses LoRA and filesystem weight broadcast. It uses the opt-in
`deployment.share_train_infer_gpu` setting with one logical trainer GPU and one
logical inference GPU mapped to that device. vLLM is bounded to a 512-token model
length, two sequences, a 512-token scheduler budget, and 25% GPU memory
utilization. Torch compilation stays disabled. vLLM captures full CUDA graphs
for decode batches of one and two sequences; prefill runs without CUDA graphs.
JIT kernel warmup is disabled. These settings define a bounded profile; they do
not guarantee that the whole run fits in 8 GiB.

Use uv 0.11.1 or newer and initialize the repository's `verifiers`,
`renderers`, `prime-envs`, and `pydantic-config` submodules. Install the GPU
runtime and this environment with:

```bash
uv sync --extra gpu --extra flash-attn --extra kernels --package prime-rl --package alphabet-sort
```

This installs CUDA, vLLM, FlashAttention, the pinned `prime-kernels` wheel, and
environment dependencies, which can require substantial disk space. Check
available disk space before syncing. On GPUs outside the wheel's supported
architecture, the package reports its kernels unavailable and prime-rl uses
its regular fallback paths.
The config sets `inference.router = "None"` so this local run uses the bare
vLLM engine and does not require the separate `disagg` router extra. Use
`uv run --no-sync` below to keep the selected environment intact.

Validate the resolved config, then start the run:

```bash
uv run --no-sync rl @ examples/tiny/rl.toml --dry-run --run.name tiny-qwen3-0.6b-lora
uv run --no-sync rl @ examples/tiny/rl.toml --run.name tiny-qwen3-0.6b-lora
```

The `[ckpt]` section saves a final checkpoint for later resume checks. Ensure
the output filesystem has room for the checkpoint and run artifacts.
The trainer sets `ckpt.single_file_per_rank = false` to write each tensor or
object separately. This limits how many CPU tensor copies the pinned PyTorch
writer retains during saving and disables GPU-to-CPU copy-ahead, at the cost
of producing more files. The
checkpoint still contains the model, optimizer, scheduler, and progress.

The config enables the file monitor. The run directory records metrics and
rollout traces; trainer logs also report step time, throughput, and trainer
peak memory. Before treating a result as the frozen baseline, record the exact
command, resolved config, git revision and any dirty diff, `uv.lock` revision,
and model snapshot identity. Confirm that all 10 steps completed, at least one
trainable LoRA parameter received a nonzero update, the matching adapter
version was applied for rollouts, the final checkpoint exists, and a resume
smoke test succeeds. Record total run time and peak whole-GPU memory while both
processes are resident; the trainer's peak-memory metric alone does not include
vLLM's allocation. Fit and performance must be measured on the target machine;
the recipe alone is not a runtime guarantee.

## Trainer-only full-parameter capacity profile

[`examples/tiny/qwen3-full-trainer.toml`](../examples/tiny/qwen3-full-trainer.toml)
configures five full-parameter Qwen3-0.6B trainer updates with fake batch size 1,
sequence length 64, full activation checkpointing, FP32 optimization/reduction,
and chunked AdamW with a 1 GiB GPU moment budget. Equivalent trainer settings
completed four guarded five-update comparisons, which found 9.5% lower warm
step medians at 1 GiB than 512 MiB, with identical hashes for all 1,271 final
parameter, gradient, buffer and optimizer-state records. This short fake-data
profile does not configure inference or rollouts and does not establish RL
quality, co-residency capacity, or longer context support. GPU and host-memory
fit still depend on the target machine.

The chunked optimizer also has an opt-in CuTe slice backend. Keep the default
Torch backend unless the installed `prime-kernels` build reports
`cute_sliced_adamw` available. For the profile above, enable it with:

```bash
uv run --no-sync rl @ examples/tiny/qwen3-full-trainer.toml \
  --trainer.optim.chunked-adamw-backend cute
```

CuTe applies only to supported contiguous FP32 CUDA slices with CPU FP32
moments and step state. Unsupported optimizer groups and unavailable kernel
builds fall back to Torch. The registered kernel currently targets SM 8.9; on
other compute capabilities the profile uses Torch. On the tested SM 8.9
machine, the five-update prototype measured 2.88% lower warm step medians with
zero swap and an identical 1,271-entry final state hash. This result is specific
to this short fake-data profile and is not a general training-speed guarantee.

## Opt-in CUDA IPC weight updates

Run the ten-update SmolLM2-135M full-parameter profile with:

```bash
uv run --no-sync rl @ examples/tiny/smollm2-full-ipc.toml --no-dashboard
```

For full-parameter RL runs on a single shared GPU, set `weight_broadcast.type`
to `cuda_ipc` and `weight_broadcast.timeout` to `120`. Configure a single-node
deployment with one GPU, one trainer GPU, one inference GPU, and
`share_train_infer_gpu = true`:

```toml
[weight_broadcast]
type = "cuda_ipc"
timeout = 120

[deployment]
type = "single_node"
gpus_per_node = 1
num_train_gpus = 1
num_infer_gpus = 1
share_train_infer_gpu = true
```

Set both allocator variables in the trainer and inference component configs. The
RL entrypoint otherwise supplies allocator defaults to child processes:

```toml
[trainer.env_vars]
PYTORCH_ALLOC_CONF = "expandable_segments:False"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:False"

[inference.env_vars]
PYTORCH_ALLOC_CONF = "expandable_segments:False"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:False"
```

CUDA IPC supports HF-layout Llama models with an identity-tied output head, an
FP32 trainer, and a BF16 vLLM model. Use one inference worker (TP=1, DP=1) and
one API server. LoRA, quantized inference, Dynamo clients, multi-GPU, and
multi-node deployments are unsupported. Startup checks reject incompatible
models and allocator settings.

The sender streams tensors through a fixed 64 MB CUDA slab and waits for an
acknowledgement before reusing it. vLLM defers fused projection loading, so the
receiver snapshots each tensor before passing it to the layerwise loader. This
keeps buffered Q/K/V and gate/up tensors intact without a full-model wire
buffer; snapshots add transient GPU memory while a layer is assembled.

Production parity matched all 182 unique vLLM parameter hashes and deterministic
token IDs and logprobs against a filesystem update. The run also preserved the
tied embedding and proved the model does not retain IPC staging storage. A
guarded SmolLM2-135M RL smoke completed ten actual updates, applied final
policy v10, and returned HTTP 200 for all 11 update requests. The IPC weight
transfer sent no checkpoint payload to disk. These checks establish correctness
for this small run, not RL quality or an end-to-end speedup; measure performance
on the target workload.
