# Full RL pair audit

This is a bounded, reproducible CPU audit of the terminal two update stock and private R30 full RL runs for `full360-admission-20261101-a2`. The user-facing result is an honest comparison of completed work and execution evidence, with no speedup or broad numerics claim from this single pair.

Run from any directory with Python 3.10 or newer and no extra packages:

```sh
uv run --offline --no-project --python /home/pradheep/.venv/bin/python python /home/pradheep/tiny-prl/prime-rl/outputs/megakernel-20261025/r30-full-rl-audit-r1/analyze_pair.py
```

The script reads the named `completion.json`, `launch.json`, guard `result.json`, terminal metrics JSONL, actual resolved config JSON, and small trace stream/annotation indexes. It does not read checkpoints, model weights, dump inference records, or decompress trace payloads. It writes `result.json` beside itself. Its input hashes include the metrics, launch/completion/guard records, live resolved configs, run TOML, and the run harness. It records the matched R30 source and DSO hashes from the passing first decode route validation.

The configuration report hash matches between arms. Comparing the live resolved files shows only output path identity and inference decode changes: stock decode is off, R30 decode is on with its private preload environment. Both resolved trainer configs use the same model, full parameters (`lora: null`), FP32 optimization and reduction, and no checkpoint or resume config.

Progress rollout/task/token fields are step-window counts, so the two rows are summed. `rollout/no_signal` is recorded as such, not treated as an admission rejection. Both arms have admission rate 1.0 and four queued/effective trainer samples per update. Trace-index record totals are reported separately from those update counts and include the terminal in-flight spill. The index totals match terminal `request_success_total`: 377 stock records and 308 R30 records. Annotation joins find four effective traces per update.

## Paired summary

| Arm | RL child elapsed | Updates | Window rollouts | Window tasks | Window tokens | Trace-index records | Peak cgroup | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stock | 174.079 s | 2 | 370 | 185 | 40,761 | 377 | 7,515,000,832 B | 7,400 MiB |
| R30 | 145.818 s | 2 | 302 | 151 | 33,285 | 308 | 7,515,977,776 B | 7,222 MiB |

The observed child elapsed difference is 16.23% in favor of R30, but it is not a verified speedup. This is one pair with unequal work, and the decode path differs by design. There are no five alternating repetitions or limiter profile. Seconds per 1,000 tokens are included in `result.json` as descriptive only; token composition also differs.

Each update has finite loss, gradient norm, and mismatch KL. Per-step KL means are retained independently and are not token weighted. R30's run metadata records two private route loads and one verified first decode event. Both child and guard return codes are zero, there is no guard stop reason, and no checkpoint save time occurred. These facts establish successful execution of this case, not numerical equivalence or broad acceptance; the separate HF floors remain unresolved.
