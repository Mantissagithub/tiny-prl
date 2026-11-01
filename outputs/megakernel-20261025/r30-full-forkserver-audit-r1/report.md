# R30 full forkserver ABBA audit

**Verdict: inconclusive.** This audit compares two forkserver-off runs with two forkserver-on runs, but this is one ABBA block and total sampled work differs. The both-on-faster predicate is recorded separately from work comparability. No profiler or trace decompression was used, so this audit cannot identify a limiting code path.

ABBA completion: **4/4 runs have a successful child completion and two updates with finite gradient norm, loss, and mismatch KL**.
Both on runs beat both off runs by actual child wall time: **true**.
Total work matched across arm totals (rollouts, task groups, tokens): **false**; off `592/296/64553`, on `572/286/62864`.
Descriptive raw child-time means: off 135.950 s, on 72.669 s (46.55% lower for on); this is not an equal-work speedup because total work differs.

## Run evidence

Child seconds come from `completion.json` and exclude preflight/config resolution. Completion counts use orchestrator progress windows; trace completion/error counts are cross-checked against trace-level metrics. `discarded` and `no_signal` remain visible because they affect usable training work.

| ABBA | Arm | State / child RC | Route | Child s | rollouts | completed task groups | tokens | completed traces | admitted / effective train traces | errors / cancelled / stale | discarded / no-signal | trainer index rows | guard (status; min RAM; peak cgroup; swap; GPU MiB) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | off | completed / 0 | pass | 134.073 | 302 | 151 | 32851 | 302 | 302 / 8 | 0 / 0 / 0 | 294 / 294 | 8 | pass; 4.24 GiB; 7.00 GiB; 0 B; 7220 MiB |
| 2 | on | completed / 0 | pass | 72.792 | 290 | 145 | 31986 | 290 | 290 / 8 | 0 / 0 / 0 | 282 / 282 | 8 | pass; 4.94 GiB; 7.00 GiB; 0 B; 7400 MiB |
| 3 | on | completed / 0 | pass | 72.546 | 282 | 141 | 30878 | 282 | 282 / 8 | 0 / 0 / 0 | 274 / 274 | 8 | pass; 4.92 GiB; 7.00 GiB; 0 B; 7218 MiB |
| 4 | off | completed / 0 | pass | 137.828 | 290 | 145 | 31702 | 290 | 290 / 8 | 0 / 0 / 0 | 282 / 282 | 8 | pass; 5.14 GiB; 7.00 GiB; 0 B; 7398 MiB |

## Per-update work and timings

Rows are reported by update, not divided into a synthetic common-work rate. The first off/on pair's step 1 is closely matched (176 rollouts and 88 tasks each; 19,057 vs 19,063 tokens). The second pair's step 1 is not matched (152/76/16,301 off vs 176/88/19,082 on). In the matched first pair, 32.97 s of the 35.67 s step-time gap coincides with lower `wait_for_batch` (50.29 s off vs 17.32 s on); forward/backward was 3.16 vs 2.27 s. This points to the batch-wait path as a place to investigate, but does not isolate a forkserver cause without profiling.

| Run | step | rollouts / tasks / tokens | grad norm | loss | mismatch KL | step s | wait for batch s | forward/backward s | broadcast s | reward mean [min,max] | reward groups none/some/all | admitted/effective (queued) traces |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| fs360-20261101-b-a1 | 1 | 176 / 88 / 19057 | 0.004263 | -0.00000175 | 0.00089545 | 57.234 | 50.289 | 3.161 | 2.324 | 0.00001839 [0.00000000, 0.00226607] | 86/2/0 | 176/4 (4.0) |
| fs360-20261101-b-a1 | 2 | 126 / 63 / 13794 | 0.067887 | 0.00005226 | 0.00095199 | 35.485 | 33.222 | 0.565 | 1.695 | 0.00020356 [0.00000000, 0.02560000] | 61/2/0 | 126/4 (4.0) |
| fs360-20261101-b-b1 | 1 | 176 / 88 / 19063 | 0.007581 | -0.00000199 | 0.00150858 | 21.562 | 17.317 | 2.274 | 1.012 | 0.00003204 [0.00000000, 0.00345386] | 86/2/0 | 176/4 (4.0) |
| fs360-20261101-b-b1 | 2 | 114 / 57 / 12923 | 0.005831 | -0.00000049 | 0.00149479 | 14.717 | 10.952 | 0.977 | 2.784 | 0.00003319 [0.00000000, 0.00277455] | 55/2/0 | 114/4 (4.0) |
| fs360-20261101-b-b2 | 1 | 176 / 88 / 19082 | 0.007558 | -0.00000005 | 0.00100434 | 24.113 | 18.806 | 2.516 | 2.055 | 0.00003250 [0.00000000, 0.00345386] | 86/2/0 | 176/4 (4.0) |
| fs360-20261101-b-b2 | 2 | 106 / 53 / 11796 | 0.015466 | 0.00000193 | 0.00033963 | 11.081 | 8.947 | 0.843 | 1.288 | 0.00006364 [0.00000000, 0.00666389] | 51/2/0 | 106/4 (4.0) |
| fs360-20261101-b-a2 | 1 | 152 / 76 / 16301 | 0.008331 | -0.00000225 | 0.00081853 | 54.979 | 49.631 | 2.226 | 1.390 | 0.00002910 [0.00000000, 0.00345386] | 74/2/0 | 152/4 (4.0) |
| fs360-20261101-b-a2 | 2 | 138 / 69 / 15401 | 0.005807 | 0.00000019 | 0.00097154 | 46.341 | 44.086 | 0.834 | 1.417 | 0.00002742 [0.00000000, 0.00277455] | 67/2/0 | 138/4 (4.0) |

## Integrity and comparability

- Frozen CPU config report: **pass**. Invariants all true: **True** (11 checked). This covers the pinned full-parameter FP32 R30/IPO recipe, two updates, no checkpoint, same task sampling and reward/admission settings, and the only behavioral arm difference being `fork_server`.
- Frozen config SHA256: off `51cec5eeefa3098990947cb79d3f8971bd16b12a3aa37651780cf9de2e1e9b02`; on `2de5b4a08c05266a5bb90182c606f4bce1d8572921b63bf52b3119b24ff33da9`.
- Pinned R30 source: `5e853d1e6b207f199c0cfd27a4d34394f7fa81ab4748ea0e51d3240dd7286638`; include: `60b234e7ab6bbd49ff5a7444fb0ff171a97ee37fe31234a72b8f045b3b495e83`; DSO: `625f39f978e76a3e0ff01ed469bcd0de0cf65e51929a9597668af9cdff6c4db2`. Each run's small worker trace JSONL is checked for a loaded event and a first-decode event on the same PID, with these source/include/library hashes.
- Runtime resolved configs are compared per adjacent off/on pair below. Allowed differences are only forkserver boolean and run/route/address identity paths. This equality check also guards the full-parameter/no-LoRA model and unchanged task/reward/admission settings.

- `fs360-20261101-b-a1` vs `fs360-20261101-b-b1` resolved config: **pass**; changed paths: trainer.json:/output_dir, orchestrator.json:/output_dir, orchestrator.json:/train/source/0/env/agent/runtime/fork_server, inference.json:/env_vars/PRIME_R30_PRELOAD_TRACE_DIR, envs/train/alphabet-sort.json:/address_file, envs/train/alphabet-sort.json:/env/agent/runtime/fork_server.
- `fs360-20261101-b-a2` vs `fs360-20261101-b-b2` resolved config: **pass**; changed paths: trainer.json:/output_dir, orchestrator.json:/output_dir, orchestrator.json:/train/source/0/env/agent/runtime/fork_server, inference.json:/env_vars/PRIME_R30_PRELOAD_TRACE_DIR, envs/train/alphabet-sort.json:/address_file, envs/train/alphabet-sort.json:/env/agent/runtime/fork_server.

## Benchmark limits

- This is 2 runs per arm, one ABBA block, not the five-or-more alternating repeats needed for a repeatability claim. Work is not equal over each complete run: compare raw rollout, task, and token totals above before using the wall-time predicate.
- Only the first pair's step 1 is close in work (176 rollouts and 88 task groups each; 19,057 vs 19,063 tokens). The second pair and the full-run totals differ, so there is no consistent matched-work ABBA comparison.
- Successful child return codes, completed trace rates, rollout errors/cancellations, route hashes, and resource-guard records are reported separately. The prelaunch envelope is not included in child seconds.
- No GPU/model activity was launched by this audit. The source run script was read. At audit preparation, system load average was 3.05 across 24 logical CPUs; this was not captured throughout each ABBA run, so it is not used to explain the gap.
- The earlier guard-only failure `fs360-20261101-a1-off` is excluded from timing and work totals; it produced no RL update and remains preserved in the source experiment directory.
- The underlying cause of any time gap is unprofiled. Step timing shows batch wait and compute durations, but does not isolate forkserver startup from other rollout/queue effects.

This report intentionally makes no verified speedup/regression claim. The report is refreshed from current artifacts each time the script runs.
