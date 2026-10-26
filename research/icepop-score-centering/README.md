# IcePop score-centering port checkpoint

This directory preserves the isolated bounded-sampling / score-centering port as a patch. It is intentionally not applied to the main source tree and does not change dependency metadata or submodules. The validated R4 run is a three-update integration smoke, not a quality or speed result.

## Contents

- `full-isolated-port.patch`: the complete `git diff HEAD` from base `09ada94d771096ab46ebe0ebad3afc28d7716084`, plus all new source and test files. It does not include dependency source trees, `.venv`, run logs, checkpoints, or generated artifacts.
- `dependency-manifest.json`: exact renderer and verifier source commits and tree hashes from prime-rl #4176 (`735496e36fc9cb82e94b50098841bb80b2a0aaa1`). Renderer: `b6ab8c7d67229be6fb99d57a8395f978d94b0566`; verifiers: `9f0f03ae4733e949e371ec80d3c95d945dfecff6`.
- `runtime-audit-r4.compact.json`: stable R4 guard, metric, bounded-support, trace-join, and policy-freshness evidence without machine-specific absolute paths. Full audit: `../icepop-rl-smoke-r4/runtime-audit.json`.
- `patch-validation.json`: patch digest, included file list, and R4 source-freeze comparison.

## Apply the checkpoint

From the repository root, create a detached validation worktree at the recorded base and apply the patch:

```bash
git worktree add --detach /tmp/icepop-port-checkpoint 09ada94d771096ab46ebe0ebad3afc28d7716084
PATCH="$(pwd)/outputs/megakernel-20261025/icepop-port-r1/checkpoint/full-isolated-port.patch"
git -C /tmp/icepop-port-checkpoint apply --check "$PATCH"
git -C /tmp/icepop-port-checkpoint apply "$PATCH"
```

The patch was checked against that exact base, applied in a temporary worktree, and all 34 production-source files in the R4 frozen-source manifest matched. New unit-test files are included in the patch but were not part of that production-source manifest.

## Reproduce focused CPU checks

After applying the patch, use the source-only dependency overlays from the experiment artifact or construct them from the exact source commits in `dependency-manifest.json`. With the existing Python environment (no package installation), set import roots and run the relevant tests:

```bash
cd /tmp/icepop-port-checkpoint
OVERLAYS="<repo>/outputs/megakernel-20261025/icepop-port-r1/dependency-overlays"
export PYTHONPATH="$PWD/src:$PWD/packages/prime-rl-configs/src:$PWD/deps/prime-kernels:$PWD/deps/verifiers/environments/alphabet_sort:$OVERLAYS/renderers:$OVERLAYS/verifiers"
uv run --offline --no-project --python /home/pradheep/.venv/bin/python -m pytest -q -m 'not gpu' \
  tests/unit/test_configs.py \
  tests/unit/inference/test_serving_tokens.py \
  tests/unit/inference/test_compact_logprobs.py \
  tests/unit/inference/test_sampling_order.py \
  tests/unit/orchestrator/test_algorithms.py \
  tests/unit/orchestrator/test_batch.py \
  tests/unit/orchestrator/test_curriculum.py \
  tests/unit/orchestrator/test_metrics.py \
  tests/unit/orchestrator/test_train_sink.py \
  tests/unit/train/rl/test_fused_lm_head.py
```

The original focused validation recorded 73 passing checks. This command is provided for later reproduction; packaging did not rerun tests. Full repository tests and coherent dependency metadata are still required before any production integration.

## Dependency boundary

The overlay SHAs above expose compact sampled-logprob parsing and `TopLogprobs`/trace-aligned evidence. They were used as source-only API inputs; this checkpoint does not vendor those dependencies. The verifier `SamplingMask` API carries IDs/counts; prime-rl constructs its own optional byte logprob transport after pairing bounded support with the top-logprob head.

The local installed metadata observed during the audit was renderers `0.1.13.dev5`, verifiers `0.4.1.dev12`, and prime-sandboxes `0.4.3`. Prime-RL currently requires verifiers `>=0.4.0`; the relevant verifier package metadata has a prime-sandboxes `>=0.5` floor, above the installed `0.4.3`. Therefore source-overlay compatibility is not a coherent installable dependency declaration yet. Do not merge or enable the port by adding overlay paths to package metadata until that floor/API compatibility is resolved.

## Runtime and validation record

R1, R2, and R3 exposed and fixed local API/transport integration gaps, recorded in the compact audit. R4 completed three updates: finite loss/KL/gradient metrics, all 12 trainer-consumed IDs joined to the stream and orchestrator annotations, bounded `logprobs=8` / `top_k=8`, and a clean resource guard. Mean head mass stayed approximately 1.0, tail scale was zero, and correction L1 was about `6.6e-7` to `7.0e-7`. The observed correction was negligible. The run does not demonstrate reward improvement, learning progress, or wall-time speedup.

The broader CPU sweep records 315 passed, two exact-float assertion failures, and 108 deselected tests. An initial setup attempt had 39 import/config failures; the corrected invocation passed all 191 inference/config tests. See full-tests-results.json and full-tests-commands.txt. A guarded full 10-update run then completed in 65.10 s with 5262 MiB sampled GPU peak and zero job swap; see full-rl-result.json. This is functional validation, not reward or wall-time improvement. The dependency metadata issue still prevents unqualified production activation.
