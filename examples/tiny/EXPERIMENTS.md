# tiny-prl experiment ledger

Optimize one change at a time on the RTX 4060 Laptop (8 GiB VRAM), preserving the model, task, numerical settings and RL objective. A candidate must pass numerical and gradient checks where applicable, complete a protected real run, and show a repeatable benefit before a speedup is claimed.

## Frozen baseline

`outputs/tiny-validation10-swap2-20261017/BASELINE.md` and `validation-summary.json` record the successful 10-update Qwen3-0.6B rank-8 LoRA baseline and real checkpoint resume through update 11. Runtime is vLLM inference with PrimeRL's native FSDP2 trainer, AdamW with CPU optimizer offload, and filesystem adapter broadcasts on the same GPU. The orchestrator computes GRPO advantages; the trainer uses the upstream IPO surrogate.

The original run took 115.13 seconds including startup, checkpoint and shutdown. Trainer step time summed to 49.07 seconds, of which 36.81 seconds was batch waiting. Whole-device sampled peak was 6688 MiB. Minimum available system RAM was 2.04 GiB. These are one-run observations, not a throughput guarantee. GNOME Software was closed during that baseline, so it is not an interchangeable control for later timing runs.

## Experiment 1: remove trainer imports from the orchestrator

The tokenizer helper previously imported the trainer model setup module. Moving it alone still loaded the model registry through eager weight-transport imports. Selecting transport implementations inside their factories, and importing sender-only LoRA helpers only when broadcasting adapters, removes that second path.

Isolated full-orchestrator imports, measured sequentially in separate processes:

| Stage | RSS after import | Import time | Trainer model modules loaded |
| --- | ---: | ---: | ---: |
| Frozen control | 1111.98 MiB | 7.67 s | 102 |
| Tokenizer extraction only | 1108.50 MiB | 7.43 s | 99 |
| Tokenizer + lazy transport imports | 788.86 MiB | 4.04 s | 0 |

The final candidate reduces observed import RSS by about 323 MiB (29.1%). This is process RSS after import, not peak training memory or a proven training throughput improvement. Import times are single observations and may reflect cache effects.

Full-run trials use 10 updates, no checkpoint in either arm, the same recipe and safety limits, and frozen source snapshots. The planned comparison order is control/candidate/candidate/control. The first control stopped after 4 updates when available system RAM fell below the 2 GiB reserve. Its elapsed duration is not a valid 10-update comparison. The candidate completed one trial, then its repeat also hit the reserve. The fourth control trial was cancelled rather than repeating an unsafe comparison. No training speedup or stable full-run RAM reduction is established.

| Trial | Result | Updates | Elapsed | Minimum available RAM |
| --- | --- | ---: | ---: | ---: |
| A1 control | Watchdog stop | 4/10 | 76.16 s | 1.965 GiB |
| B1 candidate | Completed, finite loss and gradients | 10/10 | 113.93 s | 2.024 GiB |
| B2 candidate | Watchdog stop | 1/10 | 65.61 s | 1.995 GiB |

Stop durations are not comparable to completed 10-update durations. The watchdog detected threshold crossings and terminated the jobs; neither stopped service reports an OOM result. GPU memory returned to 15 MiB afterward. More host-memory headroom is required before testing higher concurrency or profiler overhead. The five focused port/config tests pass; the tokenizer helper has identical function AST before and after extraction. Ruff and whitespace checks cover the changed optimization source.

Artifacts: `outputs/tiny-opt-tokenizer-20261017/`, including complete control/candidate source trees, SHA-256 manifests, unchanged dependency comparison, import measurements, per-trial logs, host/GPU samples, and `comparison.json`. The original full checkpoint is preserved.

## Next experiments

1. First increase measured host-RAM headroom without lowering the watchdog reserve. If headroom is adequate, compare inference `max_num_seqs=2` against `4`, leaving the model, group size, batch size, task, reward filtering and off-policy rules fixed. Batch waiting is the largest observed recurring cost; additional concurrency may also increase contention and memory use.
2. Profile one or two real trainer steps using the existing `trainer.trace_path` option. Profiling adds overhead; use it for attribution and shapes, not throughput claims.
3. Consider a CuTeDSL kernel only if the profile identifies a substantial hotspot. The temperature-dependent chunked LM-head logprob/entropy function performs vocabulary-tiled GEMMs and separate normalization, entropy and target-gather operations. The frozen LM-head weight skips weight-gradient accumulation but still requires hidden-state gradients. Each microbatch is capped at 512 tokens, split into chunks of 128; capture actual packed shapes before designing a kernel.
4. Benchmark a proposed kernel against the current operation on identical inputs, including logprobs, entropy and hidden-state gradients, edge shapes and sampling-mask behavior. Include compilation overhead separately. Keep only improvements that carry through to trainer and full-run measurements. Direct PTX is optional and needs a concrete measured benefit. Follow `skills/kernels/SKILL.md`: production kernels belong in prime-kernels, with architecture gating and a supported fallback.

## Laptop protection and graph

Run only one GPU workload at a time in a systemd user service with `MemoryMax=7G`, `MemorySwapMax=2G`, `OOMPolicy=stop`, a 2 GiB system-RAM reserve watchdog and a 600-second deadline. Never lower the reserve to complete a benchmark. Temporary swap makes total host swap 6 GiB; its 2 GiB addition is not persistent across reboot. A safe watchdog stop is recorded as a rejected or incomplete trial, not a performance result.

Regenerate the single-image progress chart after recording results:

```bash
MPLBACKEND=Agg uv run --isolated --no-project --python 3.12 --with matplotlib \
  tools/plot_tiny_hillclimb.py \
  outputs/tiny-opt-tokenizer-20261017 \
  outputs/tiny-opt-tokenizer-20261017/hillclimb.png
```

The graph separates measured import memory, complete training trials and future experiments. Future gains are never drawn as measured points. No commit, push or publication is part of these experiments.

## Experiment 2: disable activation offloading, retain checkpointing

The isolated pilot used `--trainer.model.ac-offloading None --max-steps 2 --no-ckpt`. Comparing the resolved trainer model configs confirms that `ac_offloading` is the only model-field difference. Activation checkpointing remains enabled; optimizer CPU offload, dtypes, workload and inference concurrency remain fixed. The recipe default is unchanged.

The pilot was stopped by the system-RAM watchdog after 62.70 seconds, before any completed update (0/2). Available RAM crossed the 2 GiB reserve at 1.996 GiB. Sampled whole-GPU usage peaked at 6658 MiB, below the additional 512 MiB free-VRAM watchdog threshold. The service reports an ordinary exit-code failure from supervised termination, not an OOM result; GPU usage returned to 15 MiB.

This experiment is inconclusive: there are no completed losses/gradients to validate and no meaningful performance comparison. It does not establish that activation offloading caused the remaining RAM pressure or that disabling it improves memory. The full 10-update trial was skipped after the pilot stopped. Investigate per-process host memory during startup and the first step before retrying, increasing concurrency or adding profiler overhead.

Evidence: `outputs/tiny-opt-tokenizer-20261017/activation-offload/` contains the recipe, protocol, extra VRAM-aware guard, resolved model config diff, pilot logs, GPU/host samples and summary. `activation-comparison.json` feeds the failed pilot into the single-image chart with its actual 0/2 update count. The plotting dependencies run in an isolated temporary uv environment; the pinned training environment is unchanged.

## Experiment 3: process-memory attribution and direct trainer launch

A separate two-second process-tree sampler recorded RSS, USS, PSS, swap and anonymous RSS while the independent 100-ms watchdog remained active. The unchanged-recipe, two-update diagnostic stopped at the RAM reserve before completing an update. At the largest sampled aggregate PSS (59.38 seconds), the main processes were:

| Process | PSS | USS |
| --- | ---: | ---: |
| Trainer | 1828 MiB | 1760 MiB |
| vLLM engine | 1691 MiB | 1621 MiB |
| vLLM API / inference | 1152 MiB | 1093 MiB |
| Orchestrator | 708 MiB | 653 MiB |
| Environment-server worker | 687 MiB | 634 MiB |
| Torchrun elastic launcher | 326 MiB | 292 MiB |
| RL launcher | 236 MiB | 232 MiB |
| Environment-server broker | 229 MiB | 225 MiB |

These values describe one common snapshot, not separate per-process maxima. PSS apportions shared pages; RSS should not be summed as if every page were unique. The sampler omits its own wrapper and measurements are not a throughput benchmark. Smaller agent-runtime and supervisory processes also contribute. Evidence is in `process-memory/peak-process-summary.json` and `process-memory/process-memory.jsonl`.

The local shared-GPU launcher now starts its single trainer directly with explicit rank-zero/world-size-one rendezvous environment variables, using a free TCP port. The trainer still initializes its distributed process group and FSDP normally. Other deployment modes keep Torchrun. This removes the measured elastic parent; its 292 MiB USS is not a proven reduction in peak total job RAM. Direct mode writes `trainer.log`; the launcher omits the unavailable Torchrun per-rank log hint.

| Direct-launch validation | Result | Minimum available RAM | Sampled GPU peak |
| --- | --- | ---: | ---: |
| Two-update pilot | Completed, finite losses and gradients | 2.025 GiB | 6658 MiB |
| Ten-update attempt | RAM-reserve stop before first completed update | 1.981 GiB | 6658 MiB |

The two-update pilot also confirms that no Torchrun parent appears in the sampled process tree. The full attempt did not establish a reliable memory fit. Both runs retained activation offloading, activation checkpointing and the existing numerical settings; no activation-offloading change was combined with this experiment. The source remains a local candidate, not a newly frozen successful full baseline. No training speedup is established, and no additional GPU trials were launched after the full attempt stopped.

The five focused port/config tests pass. Both default and direct-mode log hints were checked, and Ruff/whitespace checks pass. Whole-GPU memory returned to 15 MiB and no training service remained running. Total swap was almost fully used at the end; this is another reason to avoid repeated training attempts without more memory headroom.

Evidence: `direct-launch/` contains the complete frozen candidate source, SHA-256 manifest, protocol, guard, per-process samples, run logs and summaries. `launch-comparison.json` feeds the pilot and full attempt into the graph. The small log-hint and documentation correction follows the runtime source snapshot and is checked separately. Checkpoint save/resume has not been revalidated for direct launch; the original verified checkpoint remains preserved.

## Experiment 4: one inline environment server

The text task's default elastic serving pool starts an environment broker plus a spawned worker. The existing `serve.pool.max_workers = 1` path runs the same EnvServer protocol directly, avoiding the broker. The explicit pool type remains `elastic`; its tag is needed when specifying the pool table. Resolved-config comparison confirms `max_workers: null -> 1` is the only behavioral difference. The task, reward, batch/group sizes, optimizer, activation settings and inference concurrency remain fixed.

A guarded CPU-only health comparison passed for both arms. Aggregate environment-server PSS fell from 514.3 to 277.6 MiB (236.7 MiB less). The control had broker, worker and resource-tracker processes; the candidate had one server process. Neither mapped libtorch at the health-check point. This does not by itself prove the same reduction in the full RL workload.

The host had materially changed since the earlier RAM stops: about 11 GiB was available and most of the 6 GiB swap was free. The following control/candidate runs therefore establish behavior under the newer conditions, not proof that the pool change alone fixed the earlier laptop-wide pressure.

| Order | Arm | Updates / finite loss and gradients | End-to-end time |
| --- | --- | --- | ---: |
| A1 | Elastic pool | 10 / yes | 114.75 s |
| B1 | One inline server | 10 / yes | 104.75 s |
| B2 | One inline server | 10 / yes | 103.11 s |
| A2 | Elastic pool | 10 / yes | 102.32 s |

The apparent initial speed advantage did not repeat against the final control, so no repeatable throughput gain is claimed. All runs used direct trainer launch, no checkpoint saving, the same frozen source and the same guards. These runs also validate the direct launch through ten real updates under the newer host conditions.

Resident PSS alone initially looked worse for the candidate. Process samples showed that the control had paged substantially more memory into swap, so comparing resident pages alone misrepresented its allocated footprint. The confirmation pair added `SwapPss` per process and `memory.swap.current` per cgroup; it left all training settings and safety limits unchanged.

| Confirmation metric, sampled peaks | A2 pool | B2 one server | Reduction |
| --- | ---: | ---: | ---: |
| Cgroup anonymous memory + swap | 6386.4 MiB | 6179.6 MiB | 206.8 MiB |
| Process PSS + proportional swap | 7204.2 MiB | 6990.2 MiB | 214.0 MiB |

The cgroup metric excludes file cache; process PSS apportions shared pages and includes mapped file pages, so the two totals are different quantities. Peaks can occur at different times. Only the confirmation pair has proportional swap data. The candidate is retained for the lower memory footprint, supported by the isolated health comparison and completed real updates, not for a proven speedup. The base tiny recipe now uses one inline environment server. The temporary duplicate profile was removed; both tested recipes remain frozen under `single-worker/`.

Artifacts: `single-worker/` holds the CPU health probe/results, validated config difference, exact source and recipes, guards, protocol, all process samples and run summaries. `pool-comparison.json` feeds the graph. The original checkpoint remains intact. The GPU returned to 15 MiB after the comparison.

## Autonomous continuation

The user explicitly made this an active goal: continue improving without requesting a new “proceed” between experiments. Use Luna for coding, keep one GPU workload at a time, preserve the pinned home environment and frozen baseline, retain the RAM/VRAM reserves, and keep this ledger and the single image current. Do not commit, push or publish without authorization.

Next experiment: vLLM `max_num_seqs = 2 -> 4` with the retained single-server profile. Baseline telemetry showed two running requests and two waiting requests, with the orchestrator's concurrency cap reaching 25, so there is demand beyond the engine's two-sequence limit. Run a short safety pilot, then A-B-B-A ten-update comparisons. Record generated-token counter deltas, filtered work, trainer batch waiting, actual engine concurrency and preemptions as well as wall time. Treat the five-second inference scrape window as coarse corroboration. Keep the workload and reward/off-policy rules unchanged. No gain is claimed before these runs finish.

## Experiment 5 outcome: keep two inference sequences

The four-sequence safety pilot completed both updates. The full A-B-B-A comparison then completed every update with finite losses/gradients and no preemptions:

| Arm | Max sequences | End-to-end | Trainer batch wait | Observed running requests | Generated tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 2 | 92.90 s | 29.08 s | 2 | 1907 |
| B1 | 4 | 108.50 s | 41.55 s | 4 | 2456 |
| B2 | 4 | 101.51 s | 38.68 s | 4 | 2281 |
| A2 | 2 | 99.29 s | 35.31 s | 2 | 2215 |

Four sequences averaged 105.01 seconds versus 96.09 seconds for two (9.3% longer in these short trials). Mean batch waiting increased from 32.19 to 40.12 seconds. Coarse generated-token rates were 54.3/55.2 tokens/s for A and 54.4/56.9 for B. Thus the higher capacity was exercised but did not produce a useful gain for this workload. Different filtered/generated work prevents interpreting this as a general engine throughput law. Reject the change and retain `max_num_seqs = 2`; the temporary example was removed, with exact recipes and source preserved under `rollout-seqs/`.

The summarizer joins rows by producer and step, takes cumulative engine counters as last-minus-first deltas, and excludes short pilots from full comparisons. Actual preemption counters and histogram sums were zero; interpolated histogram p99 values are not used as evidence of preemption. The five focused port/config tests still pass. No checkpoint was written during these comparisons.

## Trainer profile and next candidates

A separate two-update run used the native `trainer.trace_path` profiler, keeping all safety limits. It completed successfully and produced `trainer-profile/trace/trace_0.json.gz` plus `summary.json`. Profiling changes timings; its runtime is not a speed comparison.

The trace contains trainer GPU kernel events only, not the separate inference process. Kernel durations summed to 299 ms across 40,971 events, with an approximately 5.99-second kernel-event interval. This excludes GPU memcpy time and is not GPU utilization for the entire RL job. CPU-scope/external-ID attribution identifies the chunked LM-head forward/backward; the reduction-only share is approximate because it has no dedicated named scope.

Two actual head calls had 286 and 298 tokens, hidden dimension 1024 and vocabulary 151936. BF16 GEMMs feed FP32 scaled-logit/state reductions. With token chunks of 128, each forward makes 57 vocabulary-tile GEMMs; the last vocabulary tile is 4480 wide. Larger existing token-chunk settings can reduce these loops without introducing a new kernel. A guarded standalone benchmark will compare 128/256/512 on matched BF16 inputs and frozen head weights, checking logprobs, entropy and hidden-state gradients before timing. No chunk-size change is retained yet.

CuTeDSL feasibility is established only at the source/metadata level: installed CUTLASS DSL 4.7.1 includes SM89 in its architecture map, and CUDA 13 tools are present. A possible standalone online-state reduction must preserve previous m/s/t rescaling across vocabulary tiles; a tile-local reduction would be incorrect. It must not replace the training function without its backward semantics and measured end-to-end value. Prefer the simpler chunk-size experiment first. Useful official references: [CuTe reduction API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html), [framework integration](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/framework_integration.html). No custom kernel has been integrated.

## Experiment 6 result: retain bounded decode graphs

The next inference candidate keeps two sequences and sets eager false, compilation mode NONE, CUDA graph mode FULL_DECODE_ONLY, capture sizes [1,2], and JIT warmup false. Eager mode already disables JIT warmup in the control; the explicit candidate value avoids introducing that compiler/warmup change. No Inductor compilation or numerical dtype change is enabled.

The two-update pilot passed. Native logs confirmed decode graph capture with the requested sizes, approximately 0.03 GiB in each of two capture phases, and adapter versions 0, 1 and 2 loaded successfully. That is a functional pilot, not proof of complete LoRA hot-reload parity. The full A-B-B-A comparison completed all updates with finite losses and gradients:

| Arm | Mode | End-to-end | Trainer wait | Forward/backward | Trainer time |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | Eager | 105.36 s | 37.71 s | 9.95 s | 49.65 s |
| B1 | Decode graph | 90.90 s | 24.07 s | 8.48 s | 35.13 s |
| B2 | Decode graph | 90.51 s | 24.57 s | 8.96 s | 36.21 s |
| A2 | Eager | 102.29 s | 37.22 s | 9.96 s | 49.10 s |

Across these two runs per arm, decode graph averaged 90.70 seconds end-to-end versus 103.82 seconds eager (12.6% lower). Mean trainer time was 35.67 versus 49.37 seconds, trainer wait was 24.32 versus 37.47 seconds, and forward/backward time was 8.72 versus 9.95 seconds. Separate A/B/A reload checks within both eager and graph engines verified adapter restoration; cross-engine parity passed strict 1e-3 for sampled selected and top-20 logprobs over four-token prompts, and changing scaled adapter B changed output. This bounded reload check supports retaining the graph candidate; it is not a full-distribution or general workload proof. The single ABBA cycle is promising evidence, not a broad throughput claim. No checkpoint was written.

All four full runs stayed above the 2 GiB host-RAM reserve; the lowest sampled available RAM was 3.43 GiB, and whole-GPU sampled peak was 6690 MiB. They ran with the existing 7 GiB cgroup memory cap, 2 GiB swap cap and 100 ms watchdog. The graph comparison did not record paired anonymous-memory-plus-swap peaks, so it does not establish a memory reduction. Artifacts are under `decode-graph/`, with `graph-comparison.json`, frozen configurations/source and per-run logs.

## Experiment 7 result: keep the LM-head token chunk at 128

A standalone benchmark compared token chunks 128, 256 and 512 on the frozen BF16 Qwen3-0.6B head weights, using two warmups and seven timed iterations. Larger chunks reduced isolated median time on some measured shapes, while entropy and hidden-state-gradient comparisons passed. Log-probability parity failed for several comparisons: maximum absolute error was 0.001953 at 286 tokens and reached 0.007813 at 512 tokens. Since the faster settings did not meet the benchmark's existing log-probability parity check, their timings are not grounds to change the training recipe. Keep chunk size 128 and do not loosen the tolerance. No full RL run used the larger chunks.

The isolated benchmark completed without a guard stop; sampled minimum available host RAM was 8.35 GiB and whole-GPU peak was 591 MiB. Exact per-shape timings, allocation peaks and parity results are in `lm-head-chunks/results.json`. Further CuTeDSL/PTX work remains gated on profiling evidence and must preserve log-probability, entropy and gradient correctness before measuring end-to-end benefit.

## Experiment 8 result: reject 16384-wide vocabulary tiles

A follow-up kept token chunk 128 and BF16 inputs, replacing only the two vocabulary-tile limits from 8192 to 16384 in an isolated source copy. Log-probability parity failed at all three token counts: maximum absolute errors were 0.007856 (286), 0.015654 (298) and 0.015576 (512). Entropy and hidden-gradient checks passed. Candidate timing was correctly skipped after the parity failures; no production source changed. Keep vocabulary tiles at 8192 and token chunks at 128. The isolated run stayed within the guard: minimum available RAM was 8.36 GiB and GPU peak was 541 MiB. The exact two-replacement source, hashes, errors and decision are under `lm-head-vocab-tiles/`.

## Experiment 9 result: retain activation offloading

After an earlier pilot stopped at the host-RAM reserve, a two-update retest passed with activation checkpointing still enabled and only activation offloading disabled. It took 71.31 seconds, with minimum available RAM 3.43 GiB, cgroup peak 7.00 GiB and GPU peak 6688 MiB. The pilot time is not part of the full-run comparison. The guarded A-B-B-A ten-update sequence completed all updates with finite losses and gradients:

| Order | Arm | Offloading | Elapsed | Trainer wait | Forward/backward | Minimum RAM | GPU peak |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A1 | Control | On | 85.88 s | 17.59 s | 10.84 s | 3.10 GiB | 6688 MiB |
| B1 | Candidate | Off | 89.75 s | 21.30 s | 11.17 s | 3.32 GiB | 6672 MiB |
| B2 | Candidate | Off | 87.14 s | 17.69 s | 11.94 s | 3.22 GiB | 6688 MiB |
| A2 | Control | On | 90.00 s | 21.89 s | 10.66 s | 3.24 GiB | 6678 MiB |

With activation offloading on, mean elapsed time was 87.94 s versus 88.45 s with it off (0.58% slower); mean trainer wait was 19.74 versus 19.50 s, while forward/backward was 10.75 versus 11.55 s. The arms generated different token totals, so this single ABBA is not evidence of a throughput gain. Mean sampled cgroup anonymous memory plus swap was 6139.1 MiB with offloading on and 6136.3 MiB with it off. The two-second process sampler's peak aggregate PSS plus SwapPss was 7027.8 versus 6971.4 MiB. These measures point to only a small, inconsistent memory difference and do not demonstrate a reliable benefit to disabling offloading. Retain the existing setting. The lowest available system RAM across the trials was 3.10 GiB; all remained under the 7 GiB cgroup and GPU guards.

Artifacts include frozen source/config, each trial's guard and host/GPU logs, two-second process-tree samples, and `activation-retest-comparison.json`. Derived paired means and memory metrics are recorded in `activation-retest/decision.json`; the comparison feeds the progress chart.

## Experiment 10 result: retain the 0.25 vLLM KV-cache budget for GPU headroom

The candidate changes only `inference.vllm.gpu_memory_utilization` from 0.35 to 0.25; activation offloading remains enabled and the retained recipe's two-sequence decode-graph settings are unchanged. The two-update RL pilot passed in 69.66 seconds, with minimum available RAM 3.22 GiB and sampled GPU peak 5908 MiB.

The candidate engine reported a 0.59 GiB KV cache, 5,552 tokens, or 10.84 simultaneous 512-token contexts. A separate guarded stress service sent two concurrent requests with 448 prompt tokens and 64 generated tokens each, using distinct cache salts. Both returned all 64 tokens; the vLLM preemption counter delta was zero, with 896 prompt and 128 generation tokens counted. That standalone stress service ran for 30.65 seconds, with minimum available RAM 6.63 GiB, cgroup peak 4.15 GiB and GPU peak 2079 MiB. It was not an RL throughput comparison.

The ten-update A-B-B-A comparison completed with the same task, batch, checkpointing, activation offloading and guards. All four runs completed 10 updates with finite losses and gradients:

| Order | GPU budget | Elapsed | GPU peak | Generated tokens | Actual preemptions |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 0.35 | 88.28 s | 6684 MiB | 2055 | 0 |
| B1 | 0.25 | 89.34 s | 5908 MiB | 2459 | 0 |
| B2 | 0.25 | 89.74 s | 5908 MiB | 2129 | 0 |
| A2 | 0.35 | 89.94 s | 6656 MiB | 2271 | 0 |

Mean elapsed time was 89.11 s at 0.35 and 89.54 s at 0.25 (+0.48%, or 0.43 s); this is not a speed gain. Mean sampled whole-device GPU peak fell from 6670 to 5908 MiB (762 MiB, 11.4%). The actual `vllm:num_preemptions_total` counter and request-preemption sum remained zero in all four runs, with no counter resets. The four-request capacity was not exercised during these RL runs. Mean cgroup anonymous memory plus swap differed by 9.6 MiB and process PSS plus SwapPss by 26.1 MiB, both small relative to the measurements. Generated-token totals also differed between arms. Retain 0.25 for additional GPU headroom on this tested workload, not for a demonstrated throughput or host-memory improvement.

The candidate capacity and standalone two-request stress passed as described above. Per-run guard, host/GPU samples and process-tree samples are under `kv-budget/`; `kv-comparison.json` and `kv-budget/decision.json` record the paired results and decision. This is one ABBA cycle, not a general performance guarantee.

Current single-image chart: [hillclimb.png](hillclimb.png). The rendered source artifact remains `outputs/tiny-opt-tokenizer-20261017/hillclimb.png`.

## Optimized checkpoint resume validation

The retained decode-graph, inline-environment and direct-trainer profile restored the frozen baseline step-10 checkpoint and completed update 11 with finite loss (-0.0023) and gradient norm (1.3913). Guard return code 0, no reserve stop; elapsed 65.08 seconds, minimum available RAM 3.37 GiB, whole-GPU peak 6688 MiB. No new full checkpoint was written. Native logs confirm resume from step 10; the GPU returned to 15 MiB after cleanup. Five focused port/config tests and Ruff checks passed. The one-loss sample standard deviation remains undefined and is omitted by the metrics writer; the actual loss and gradient norm were finite. Evidence: optimized-resume/validation.json and outputs/tiny-optimized-resume11-20261017/.
