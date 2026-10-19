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
3. Continue CuTeDSL work on the profiled LM-head online-update hotspot. The temperature-dependent chunked LM-head logprob/entropy function performs vocabulary-tiled GEMMs and separate normalization, entropy and target-gather operations. The frozen LM-head weight skips weight-gradient accumulation but still requires hidden-state gradients. Each microbatch is capped at 512 tokens, split into chunks of 128; capture actual packed shapes before designing a kernel.
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

## Experiment 11 result: retain full activation checkpointing

This comparison used the retained 0.25 KV budget and decode-graph settings. The control keeps full activation checkpointing (`ac.mode = full`, frequency 1) and its dependent activation-offloading limit of five in-flight activations. The candidate sets `trainer.model.ac = None`; config validation consequently clears `ac_offloading`. These dependent settings were tested together. No other recipe change was intended.

The two-update pilot passed in 65.88 seconds, with minimum available RAM 3.54 GiB and GPU peak 6232 MiB. Across the full A-B-B-A, all runs completed 10 updates with finite losses and gradients:

| Order | AC setting | Elapsed | Trainer wait | Forward/backward | GPU peak | Generated tokens | Actual preemptions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | Full | 87.90 s | 20.35 s | 11.42 s | 5908 MiB | 2743 | 0 |
| B1 | Off | 88.13 s | 23.57 s | 8.10 s | 6258 MiB | 2415 | 0 |
| B2 | Off | 88.76 s | 26.11 s | 6.38 s | 6278 MiB | 2268 | 0 |
| A2 | Full | 89.55 s | 20.75 s | 12.43 s | 5876 MiB | 2604 | 0 |

The AC-off arm averaged 88.45 s versus 88.73 s with full AC (0.31% lower), while mean generated tokens were 2342 versus 2674. Trainer time was nearly unchanged (35.80 versus 35.73 s), batch wait increased by 4.29 s, and forward/backward time fell by 39.3%. Sampled GPU peak rose by 376 MiB (6268 versus 5892 MiB). The small wall-time difference came with less generated work, so it does not establish an end-to-end speed gain. The actual vLLM preemption counter and request-preemption sum were zero in every run, with no resets.

CPU inspection found that all 392 F32 adapter tensors changed between steps 1 and 2 and none were nonfinite. This verifies that adapter values changed and remained finite; it is not a gradient-equivalence check. Retain full activation checkpointing and its dependent offloading for this recipe because the AC-off setting did not show a meaningful end-to-end benefit and raised GPU peak memory. Details are in `activation-retest/ac-recompute/decision.json`, `adapter-validation.json`, `config-diff.json`, and `ac-comparison.json`.

Current single-image chart: [hillclimb.png](hillclimb.png). The rendered source artifact remains `outputs/tiny-opt-tokenizer-20261017/hillclimb.png`.

## Retained profile 32-update stability acceptance

The retained recipe completed one guarded 32-update run with finite loss and gradient values on all updates, in 156.59 seconds. Minimum available RAM was 3.52 GiB, whole-GPU peak was 5908 MiB, and preemption counters remained zero. The step-32 adapter loaded in inference; CPU inspection found all 392 F32 adapter tensors changed from step 31 and none were nonfinite. This is a stability check, not a speed comparison, convergence result, or arbitrary-duration guarantee. During teardown, the API server sent SIGTERM and vLLM logged an `EngineDeadError`; all 32 updates completed, and the step-32 adapter load returned HTTP 200. The original step-10 checkpoint remains preserved and no checkpoint was written by this run. Evidence: `outputs/tiny-opt-tokenizer-20261017/retained-acceptance32/` and `outputs/tiny-retained32-20261018/`.

## Optimized checkpoint resume validation

The retained decode-graph, inline-environment and direct-trainer profile restored the frozen baseline step-10 checkpoint and completed update 11 with finite loss (-0.0023) and gradient norm (1.3913). Guard return code 0, no reserve stop; elapsed 65.08 seconds, minimum available RAM 3.37 GiB, whole-GPU peak 6688 MiB. No new full checkpoint was written. Native logs confirm resume from step 10; the GPU returned to 15 MiB after cleanup. Five focused port/config tests and Ruff checks passed. The one-loss sample standard deviation remains undefined and is omitted by the metrics writer; the actual loss and gradient norm were finite. Evidence: optimized-resume/validation.json and outputs/tiny-optimized-resume11-20261017/.

A separate resume using the retained 0.25 GPU-memory profile also restored frozen step 10 and finished update 11: guard return code 0 with no stop reason, 72.95 seconds elapsed, 4,008,435,712 bytes minimum available RAM, and 5908 MiB whole-GPU peak. Step 11 had finite loss 0.00067075 and gradient norm 2.23842; native inference logs show the step-10 and step-11 adapters loaded. No new full checkpoint was written. Evidence: `outputs/tiny-profile-resume25-20261018/validation.json` and its native logs.

## Experiment 12 result: retain adaptive rollout concurrency

The candidate changed only the orchestrator's `max_inflight` ceiling from 1024 to 4. `initial_inflight` stayed unset and `min_inflight` stayed 1, so the controller remained adaptive. Task, reward, sampling, batch/group sizes, model, and numerical settings were unchanged. The two-update candidate pilot passed in 74.35 seconds, with minimum available RAM 4.29 GiB and GPU peak 5908 MiB. All four guarded A-B-B-A runs completed 10 updates with finite loss and gradients:

| Order | Ceiling | Elapsed | Trainer wait | Forward/backward | Generated tokens | Attempts / no-signal / queued | GPU peak | Preemptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | Adaptive | 89.69 s | 23.18 s | 10.48 s | 2496 | 110 / 70 / 40 | 5876 MiB | 0 |
| B1 | 4 | 92.17 s | 23.19 s | 10.52 s | 2280 | 100 / 60 / 40 | 5908 MiB | 0 |
| B2 | 4 | 91.94 s | 25.87 s | 8.82 s | 2015 | 98 / 58 / 40 | 5908 MiB | 0 |
| A2 | Adaptive | 92.94 s | 22.15 s | 10.55 s | 2423 | 102 / 62 / 40 | 5908 MiB | 0 |

Mean elapsed time was 91.32 s with the adaptive ceiling and 92.05 s with cap 4 (+0.81%). Mean trainer wait rose from 22.66 to 24.53 s (+8.24%), while mean generated tokens fell from 2459.5 to 2147.5 (-12.7%). The cap arm averaged seven fewer attempts and no-signal discards per run for the same 40 queued traces, but this stochastic difference did not produce a speed gain. No stale, errored, cancelled, pruned, or preempted traces were recorded. Actual inference running peaked at two in both arms; the observed controller ceiling reached 10 in the adaptive runs and 4 in candidate runs. Swap-aware guard and process samples showed no clear host-memory reduction, and GPU peak was similar (5892 versus 5908 MiB mean). Retain adaptive concurrency; do not promote cap 4 as a rollout optimization. Detailed comparison and copied guard/process evidence are in `outputs/tiny-opt-tokenizer-20261017/rollout-queue/decision.json`; the full comparison is in `outputs/tiny-opt-tokenizer-20261017/queue-comparison.json`.

## Experiment 13: compile compatibility screen only

The candidate used the retained profile with only `trainer.model.compile = { mode = "reduce-overhead", fullgraph = false }`; both two-step recipes passed direct config validation. The guarded candidate completed two updates with finite losses and gradients, return code 0, no stop reason, 4.15 GiB minimum available RAM, 7.00 GiB peak cgroup use, and 5908 MiB sampled GPU peak. It took 73.53 seconds including startup. The first forward/backward step was 4.63 seconds and the second was 0.66 seconds; without an eager control, these values do not establish a speed effect.

The trainer logged “Compiled 28 layers,” and the first forward emitted a Dynamo tracing warning at `norms.py:37`. In-place `nn.Module.compile` is set before FSDP2 changes module classes; FSDP2 adds hooks and class mixins, while PyTorch's module call wrapper still dispatches `_compiled_call_impl`. These facts show compile setup and a trace attempt, not the amount of graph coverage or successful compiled-kernel execution. The isolated Inductor cache remained empty and no Triton cache was created; this is inconclusive rather than proof that no compiled graph ran. There was no eager-versus-compiled fixed-batch loss, logprob, gradient, or adapter parity comparison. Do not promote this setting and make no speed claim. The screen only establishes that this two-step configuration completed under its guards; compiled numerical parity remains unproven.

Evidence: `outputs/tiny-opt-tokenizer-20261017/compile-screen/decision.json`, `summary.json`, `protocol.md`, `source-manifest.json`, trainer log under `outputs/tiny-compile-pilot-20261018/`, and the isolated cache paths recorded in the decision artifact.

## Experiment 14: CuTe online-update optimization in progress

The optional dynamic-row `cute_online_update` implementation is registered and passed the guarded direct-binding parity check through the autograd function's seventh argument. The test covered log-probabilities, entropy, hidden-state gradients and weight gradients, including masked and unmasked replay with the real BF16 Qwen checkpoint weight, non-unit inverse temperatures, a V=8209 weight-gradient case, carried-state maxima that rise and fall, and a non-default Torch stream. The comparison used the current stream, and parity passed. The kernel remains optional; `torch` stays the default while CuTeDSL/PTX optimization continues.

In the exploratory 298-row local head benchmark, three warm samples had a 27.52 ms Torch median and a 24.34 ms CuTe median (11.6% lower). This is one shape and one short sample set, so it is a local result, not a general head speed claim. The parity guard returned 0 without a stop in 8.45 seconds. Evidence: `outputs/tiny-kernel-20261018/dynamic-full.json` and `comparison/registered-parity/result.json`.

The A-B-B-A ten-update runs all completed with finite losses and gradients: A1 106.36 s, B1 85.37 s, B2 92.79 s, and A2 88.14 s. Raw means were 97.25 seconds for control and 89.08 seconds for candidate (-8.4%), but A1 spent 31 seconds waiting for shared-GPU inference readiness while the other runs each waited 20 seconds. Candidate forward/backward time averaged 0.648 seconds over steps 2–10, versus 0.611 seconds for controls; candidate repeat medians also disagreed (0.488 and 0.647 seconds). Inference request counts and generated-token totals varied. These runs do not establish an attributable full-RL speed gain and do not justify a raw -8.4% optimization bar. Keep the Torch default and continue controlled CuTeDSL/PTX optimization; the kernel phase is not complete.

Evidence: `outputs/tiny-kernel-20261018/comparison/decision.json`, `analysis.md`, `summary.json`, and `dynamic-full.json`. Small-model full-parameter training (135M first) remains pending and has not started; kernel optimization remains active.

## Experiment 15: CuTe/Triton v2 head components and shared-GPU profile

Two isolated v2 candidates passed parity on the 298-row, unmasked real-Qwen-weight head case, with 57 actual online-update calls each. The CuTe R2 run measured five warm wall samples: its own Torch median was 27.1614 ms and candidate median 23.0906 ms (CUDA-event medians 27.1430 and 23.0757 ms). The separate Triton v2 run measured five warm wall samples: its own Torch median was 29.1078 ms and candidate median 22.7513 ms (CUDA-event medians 29.0878 and 22.7367 ms). Both also passed log-probability, entropy, hidden-gradient and V=8209 weight-gradient checks; masked cases used the existing fallback. The per-run Torch baselines were collected separately, so these results do not rank CuTe against Triton. They are local component timings, not RL speed evidence. Guarded runs completed successfully with sampled GPU peaks of 865 MiB (CuTe) and 845 MiB (Triton).

The resolved RL profile configuration (`top_p=1`, `top_k=null`, no inference sampling mask) is eligible for the fused path, but the run did not record direct tensor-level proof that it executed there. In the recovered 15-second Nsight Systems window, the captured-process CUDA kernel interval union was 3.569 seconds: 3.219 seconds for vLLM and 356 ms for the trainer, with about 6 ms overlap. These are time-interval unions from the profiled processes, not whole-device utilization. The capture overlaps active inference and trainer work after startup, but the trainer reported variable batch waits and the orchestrator discarded many no-signal episodes; it is not a clean steady-state workload.

The RL process reached update 6/10, then the cgroup OOM killer stopped it during Nsight report generation. The 15-second raw capture was recovered and exported offline, so its attribution data is usable; the RL trial is incomplete and does not establish a pipeline gain. No CUDA-event timing, kernel-duration sum, or raw aggregate GPU percentage should be presented as utilization or an RL speedup. Keep Torch as the default and continue the split-K optimization work; this kernel phase remains active. Small-model full-parameter training (135M first) remains on hold and has not started.

The later balanced single-process comparison used one shared fixture and four alternating order cycles. With 8 timed samples per arm, median synchronized wall times were 25.0746 ms for Torch, 20.7200 ms for CuTe v2 (-17.4%), and 20.3419 ms for Triton v2 (-18.9%); CUDA-event medians were 25.0594, 20.7008, and 20.3202 ms. All arms passed the same strict log-probability, entropy, and hidden-gradient parity checks, with 57 active fused-update calls per candidate execution. Triton’s median was 0.38 ms lower than CuTe’s in this fixture; the result supports a comparison only on this one shape and not a general backend ranking. It does not establish training or whole-RL speed. The guard returned 0 with no stop in 10.27 seconds, peak cgroup use 1.44 GiB, minimum available RAM 6.58 GiB (7.06 GB), and sampled GPU peak 521 MiB. Profiling was disabled and first-call compilation timings were kept separate. Evidence: `outputs/tiny-kernel-20261018/head-balanced/result.json`, `guard/result.json`, `source/manifest.json`, and `guard/unit-properties-start.txt`.

The recovered Nsight capture's 15-second CUDA kernel interval union was 3.569 seconds across the captured processes. vLLM's per-process union was 3.219 seconds and the trainer's was 356 ms, with about 6 ms overlap. These interval unions are not whole-device utilization. The partial capture reached update 6/10 before cgroup OOM during report generation; it remains attribution evidence, and no full-RL gain is established. The upstream review in `outputs/tiny-kernel-20261018/analysis/upstream-lora-inspiration.md` identifies further kernel ideas, but none has been implemented or measured. Keep Torch as the default while CuTeDSL/PTX and Triton kernel optimization continues. Small-model full-parameter training (135M first) remains on hold and has not started.

Evidence: `outputs/tiny-kernel-20261018/candidate-fused-v2/result-r2.json` and `guard-r2/result.json`; `candidate-triton-v2/result.json` and `guard/result.json`; `head-balanced/result.json`, `guard/result.json`, `source/manifest.json`, and `guard/unit-properties-start.txt`; `outputs/tiny-profile-20261018/analysis/profile-analysis.md`, `offline-export/profile-recovered.sqlite`, and `outputs/tiny-nsys-baseline-20261018/logs/attempt_1/`. The full chronology chart is `hillclimb.png`.

## Experiment 16: small-M LoRA split-K component check

An isolated native-Triton component benchmark changed split-K from 64 to 4 only for the rank-8 LoRA shrink/expand GEMMs with K=1024 and M≤2. The original mixed-adapter fixtures measured 100 CUDA-graph replays per sample in ABBA order, with two samples per arm: QKV M=2 (`[0,-1]`) was 19.111→13.942 µs (27.0% lower), gate/up M=2 (`[0,-1]`) was 17.332→14.449 µs (16.6% lower), and gate/up M=1 was 21.525→19.020 µs (11.6% lower). An expanded boundary run added QKV M=1 at 23.024→18.452 µs (19.9% lower) and QKV M=2 with both rows using the adapter (`[0,0]`) at 18.188→13.583 µs (25.3% lower); the earlier mixed fixtures reproduced. Stock split-K 64 remains on the K=2048/3072 o_proj and down_proj cases.

Parity passed with the cached real rank-8 A/B weights at scale 2: FP32 intermediate error stayed below 1e-5, final BF16 outputs matched exactly, the graph-captured outputs and intermediates matched eager results, and current/non-default stream checks passed. The cached adapter tensors are read-only inputs; no derived adapter values are cached. This is a LoRA shrink/expand graph-component result, not whole-model or end-to-end inference evidence. Fixed-token inference and adapter-reload parity remain pending. No production default or model-level performance claim changes from this result.

The guarded component runs completed with no stop reason: the first in 5.84 seconds (187 MiB sampled GPU peak; 7.81 GB minimum available RAM), and the expanded boundary run in 4.03 seconds (187 MiB sampled GPU peak; 7.55 GB minimum available RAM). The added cases preserve FP32-intermediate error below 1e-5, exact BF16 output parity, and CUDA graph/current-stream parity. These are shrink/expand component timings, not model-level or end-to-end inference results.

The subsequent fixed-token stress pilot did not pass the required `1e-3` selected/top-20 logprob tolerance: the high-scale synthetic B adapter reached about 0.498 cross-arm error. A stock-versus-stock repeat on the same stress fixture also differed by 0.0402 and 0.0355 for its two B prompts, while A matched exactly. Both arms produced the same token IDs, but that does not satisfy logprob parity. This exposes strong sensitivity in the stress fixture and does not validate the candidate. Keep the tolerance unchanged; do not claim model-level parity or promote split-K from this result. Fixed-token inference and adapter-reload parity remain unresolved; upstream fusion and deterministic partial-reduction ideas remain unmeasured. Evidence: `outputs/tiny-kernel-20261018/inference-lora-splitk/model-pilot-r2-analysis.md`, `stock-repeat-observed.json`, and the guarded model pilot artifacts, in addition to `result.json`, `result-boundaries.json`, `guard/result.json`, `guard-boundaries/result.json`, and `CONFIG_OVERRIDE_PLAN.md`. Split-K remains a component experiment pending a valid model-level check.

## Experiment 17: Triton backward epilogue component

An isolated Triton kernel replaced only the unmasked LM-head backward pointwise sequence; stock BF16 logits GEMMs, casts before the hidden/weight-gradient GEMMs, vocabulary tiles, and token chunks remained unchanged. The sequence keeps FP32 scaling, exponential, upstream-gradient multiplication, target injection, and final temperature multiplication in the reference order, then stores BF16 dlogits. Masked replay uses the existing Torch backward. Pointwise strict checks were exact after BF16 casting, including tails, extreme logits, ignored labels with nonzero upstream gradient, and signed/zero upstream gradients. Full 298-row real-weight output, entropy, and hidden-gradient parity passed for both BF16 and FP32 inverse temperatures. The V=8209 full weight-gradient case, non-default stream, and masked fallback also passed; masked fallback made zero epilogue calls.

In the same 298-row fixture, four balanced cycles (8 timed samples per arm) measured median synchronized wall times of 24.8010 ms for stock Torch, 20.1484 ms for forward-only Triton v2, and 14.7904 ms for the combined forward/backward Triton component. The combined arm was 40.4% below Torch and 26.6% below forward-only in this local fixture. These are component timings, not a general backend result or RL/training speedup. The guarded run completed in 8.86 seconds without a stop, with 567 MiB sampled GPU peak and 6.10 GiB minimum available RAM. An earlier run reached result writing but failed JSON serialization on tuple metadata keys; its logs and frozen sources are preserved under `backward-epilogue/attempt1/`, and its timing output was not accepted. The serialization fix passed CPU checks before the successful rerun. Evidence: `outputs/tiny-kernel-20261018/backward-epilogue/result.json`, `guard-r2/result.json`, `source-r2/`, and `attempt1/`. The prototype is isolated; kernel work continues and Torch remains the production default.

## Experiment 18: CuTe v2 full-head backward component

The row-by-1024-vocabulary-subtile CuTe backward epilogue passed the same-process strict parity suite on the real 298-row BF16 Qwen head. This included both BF16 and FP32 inverse temperatures, 57 active calls per full-head candidate, masked replay through the stock fallback, a V=8209 full weight-gradient case, and a non-default stream. Pointwise tails and extremes passed. No tolerances changed. An earlier harness run failed its Torch-arm call-counter gate because the displayed module-global counter had not been reset; the frozen attempt is preserved under `backward-epilogue-cute-v2/attempt2/`. The reset was a harness-only fix; the CuTe kernel was unchanged.

In four balanced order cycles (8 timing samples per arm), median synchronized wall times were 24.7514 ms for stock Torch, 14.6921 ms for full Triton v2 forward/backward, 16.5546 ms for the mixed Triton-forward/CuTe-v1-backward arm, and 15.1920 ms for full CuTe v2 forward/backward. CuTe v2 was 38.6% below Torch in this local head fixture; Triton was 0.48 ms faster in the same comparison. CUDA-event medians were 24.7348, 14.6737, 16.5376, and 15.1755 ms, respectively. This is a component comparison on one shape, not an RL or general backend speed claim. The guard completed in 11.07 seconds without a stop (567 MiB GPU peak; minimum available RAM 6.76 GiB). Evidence: `outputs/tiny-kernel-20261018/backward-epilogue-cute-v2/result-r3.json`, `guard-r3/result.json`, `source-r3/manifest.json`, and `attempt2/`.

The latest ten-update Triton integration A-B-B-A completed all four runs with finite losses and gradients. Control/candidate mean elapsed time was 97.07/96.36 seconds (-0.7% candidate), while mean forward/backward time was 7.856/7.898 seconds; batch-wait means were 38.36/33.74 seconds. The small wall-time difference does not establish an attributable whole-RL gain given variable work and wait time. Keep Torch as the default while kernel optimization continues. Evidence: `outputs/tiny-kernel-20261018/triton-integration/abba-summary.json` and its four guarded run artifacts.

A follow-up CuTe fused-head check after removing `KeepPTX` passed all four strict parity cases under the ordinary compile path (8.80 seconds, 3,275 MiB sampled GPU peak, 8.62 GiB minimum available RAM). It produced no PTX dump and adds no timing or production-performance claim. Evidence: `outputs/tiny-kernel-20261018/cute-fused-integration/ordinary-compile-parity.json` and its guard result.

## Experiment 19: 135M full-parameter pilot

The first full-parameter SmolLM2-135M FP32 pilot using optimizer CPU offload stopped safely before its first update when available host RAM crossed the 2 GiB reserve (45.79 seconds; 5,026 MiB GPU peak). Changing only optimizer offload to `none`, while retaining FP32 optimization/reduction, completed 2/2 updates with finite losses and gradients in 55.34 seconds (minimum available RAM 2.26 GiB; 5,000 MiB GPU peak). A later ten-update checkpoint attempt with optimizer offload `none` stopped after one finite update (minimum available RAM 2.09 GiB; GPU peak 4,798 MiB). A no-activation-offload attempt stopped after three finite updates (minimum 2.00 GiB RAM). The cap-4 attempt stopped after seven finite updates (minimum 2.00 GiB RAM). These were guard stops, not numerical failures. A separate four-step profiling run stopped before its first trainer update (minimum 2.12 GiB RAM; GPU peak 4,814 MiB).

A diagnostic ten-update run retained full activation checkpointing and FP32 optimization/reduction and completed all 10 updates with finite losses and gradients in 97.80 seconds. The run recorded a 7.00 GiB cgroup peak, 5,018 MiB sampled GPU peak, and 5.00 GiB minimum available host RAM. Its instrumentation measured trainer-process anonymous memory falling by about 405 MiB after the startup broadcast's combined `gc.collect()` and `malloc_trim(0)`, and about 203 MiB after step 1's broadcast. These are measured post-intervention changes; they do not isolate the effect of GC versus trim, and the host had about 11 GiB available during this later run compared with about 8.5 GiB during earlier attempts. The safe completion therefore cannot be attributed to trimming alone.

The saved `step_10` trainer checkpoint passed the CPU DCP audit: all 272 optimizer parameter entries were present, each FP32 Adam moment family contained 134,515,008 elements, and all recorded optimizer steps equaled 10. The only model FQN without its own optimizer entry was the known tied `lm_head.weight` alias; its FP32 values matched `model.embed_tokens.weight` byte-for-byte using separate bounded reads. Audit output: `outputs/tiny-fulltrain-20261018/dcp_audit/phase-trim10-audit.json` (stdout and source hash beside it). A fresh run resumed from that checkpoint through update 20 and completed updates 11–20 with finite losses and gradients. It took 92.17 seconds under the existing guard (minimum available RAM 4.96 GiB, cgroup peak 7.00 GiB, sampled GPU peak 5,038 MiB); `--no-ckpt` was set, so this resumed run wrote no second trainer checkpoint.

For this resumed run, duration-weighted MFU was 4.18% over updates 11–20 and 6.24% over warm updates 12–20; the first resumed update is included only in the all-ten value. The corrected RTX 4060 Laptop peak reference is 29.12256 TFLOPS; the prior 312 TFLOPS A100 reference does not apply to this device. Warm-window model throughput was 2,034 input tokens/s and 1.817 estimated model TFLOPS/s. Across all ten updates, forward/backward plus optimizer work took 3.201 seconds, trainer step time 46.053 seconds, batch wait 35.351 seconds, and weight broadcast 6.945 seconds. Time-weighted NVML samples measured 14.62% device-wide GPU utilization over the full RL loop and 16.90% over the warm window. The samples include both trainer and vLLM and are not kernel occupancy or process-attributed utilization. MFU is a model-FLOP estimate, not measured hardware FLOPS. Evidence: `outputs/tiny-fulltrain-20261018/mfu-resume20-training-summary.json`, `analysis/mfu-resume20.md`, `mfu-resume20-guard/result.json`, and `runs/smollm2-135m-full-torch-mfu-resume20-20261019/monitors/file/metrics.jsonl`.

To make room while retaining the canonical startup weights, successful pilot step 2, the full `step_10` checkpoint, and all checkpoint files, three verified XOR archives replaced only eight selected broadcast copies. For the first archive, root independently restored and SHA-verified a full 325 MB file, then checked all three original and payload hashes; for the other two, root independently stream-restored and SHA-verified the archived files. Only those exact source files were then removed. The manifests are `outputs/tiny-fulltrain-20261018/archive/failed-older-checkpoints-v1/archive-manifest.json`, `outputs/tiny-fulltrain-20261018/archive/reclaim-for-mfu-v1/archive-manifest.json`, and `outputs/tiny-fulltrain-20261018/archive/reclaim-for-sharded-pilot-v1/archive-manifest.json`; independent restore records and exact removal lists are beside them. The third archive preserves phase-trim10 step 9 and MFU-resume20 step 19. The canonical startup file, pilot step 2, step 10 checkpoint, step 20 broadcast, and all checkpoint files remain untouched. These memory and storage interventions do not alter the kernel comparisons or production defaults.

A separate four-run A-B-B-A CPU writer benchmark compared the default single-shard safetensors serialization with six 64 MB shards. All 273 tensors matched exactly. Mean writer-phase peak RSS fell from 1,191.06 MiB to 941.13 MiB (249.93 MiB, 20.98%); mean serialization time rose from 0.20589 s to 0.21045 s (2.21%). This measures the production writer only in a fresh single-rank Gloo process with tmpfs outputs. It does not measure transfer, job-wide memory, or end-to-end training speed. Evidence: `outputs/tiny-fulltrain-20261018/weight-transfer-benchmark/abba-summary.json`.

The six-shard, 64,000,000-byte setting then completed a guarded two-update SmolLM2-135M full-parameter RL pilot in 54.49 seconds with finite metrics, 5.22 GB minimum host-available RAM, 6.11 GiB cgroup peak, 4,816 MiB sampled GPU peak, and 2.24 GB minimum free disk. The receiver completed all six shards on both reloads, returned HTTP 200 for the startup and two update broadcasts, and continued serving generations. A vLLM rotary-weight load warning appeared during reload; successful acknowledgements and continued serving do not prove exact equality of every inference parameter. This is a functional integration test without a matched single-shard control, not an end-to-end memory or speed result. Evidence: `outputs/tiny-fulltrain-20261018/analysis/shard64-pilot2-review.md`, `shard64-pilot2-guard/result.json`, and the guarded run artifacts.

Evidence: `outputs/tiny-fulltrain-20261018/pilot2-guard/result.json`, `pilot2-offload-none-summary.json`, `baseline10-checkpoint-summary.json`, `baseline10-no-ac-offload-summary.json`, `baseline10-cap4-summary.json`, `memory-profile4-guard/result.json`, `phase-trim10-guard/result.json`, `phase-profile/trim10/`, `dcp_audit/phase-trim10-audit.json`, `archive/failed-older-checkpoints-v1/`, `archive/reclaim-for-mfu-v1/`, and `mfu-resume20-guard/`.
