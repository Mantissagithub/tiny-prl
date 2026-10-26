# tiny-prl research handoff

Checkpoint: 2026-10-26. This note captures the current state of the IcePop port, 360M decode work, and KV-cache work. It does not claim that pending suite or full-run work has passed.

## Repository and environment boundaries

- Prime-RL working repository: `/home/pradheep/tiny-prl/prime-rl`. Local `main` and `origin/main` are `c81caffb2ba8c3492daca8132435e88e082a3f1b`; local `upstream/main` is `40babc0833dda27f6c23ab53547dd527e465ae38`. The origin is the personal fork; upstream is PrimeIntellect.
- Prime-Kernels checkout: `/home/pradheep/tiny-prl/prime-kernels`. Local `main` and `origin/main` are `1ead41b19e71bae6fbaeb9c1c41f05c4ed12491b`, also the Prime-RL `deps/prime-kernels` gitlink. The canonical upstream remote is configured; this checkout had no fetched `upstream/main` ref at checkpoint time.
- The root worktree's tracked production source is not the isolated IcePop port. The local status includes changes to `examples/tiny/EXPERIMENTS.md`, `examples/tiny/hillclimb.png`, and `tools/plot_tiny_hillclimb.py`; preserve those concurrent changes. The primary progress image remains `examples/tiny/hillclimb.png`; this handoff does not change its path or design.
- The score-centering port is a detached, dirty worktree at `outputs/megakernel-20261025/icepop-port-r1/worktree`, based on `09ada94d771096ab46ebe0ebad3afc28d7716084` (parent `8fa401ebc0a5e14495161bf8c8778e93bb28fcc6`). It contains uncommitted port changes and must not be confused with Prime-RL `main`. Reproduction checkpoint/patches are in `icepop-port-r1/checkpoint/`; patch SHA-256 is `6b50a4b496a02c99a92c71362e689d03f34208ab48f4c4a94242edc5c0152aa9`.
- Use the pinned `/home/pradheep/.venv` through offline `uv run`; keep it read-only. No dependency install, root lockfile update, dependency gitlink update, dtype change, or production integration is part of this checkpoint. Dependency overlays are source-only and remain under `outputs/`; they are not symlinked into the checkout or installed environment.

## Score-centering / IcePop

Prime-RL upstream #4012 is `90fbb8db28252f509b9add326aaa37fab4ce2a87`; bounded replay #4176 is `735496e36fc9cb82e94b50098841bb80b2a0aaa1`. The isolated port uses the #4176 source snapshots recorded in `icepop-port-r1/dependency-overlays/dependency-manifest.json`: renderers `b6ab8c7d67229be6fb99d57a8395f978d94b0566`, verifiers `9f0f03ae4733e949e371ec80d3c95d945dfecff6`. The root checkout still pins older submodules (renderers `5d870d8eaa65114f5bef8a4cffc146fbcb418ec4`; verifiers `a25041a12bd2396ecdbbc9e112b0464155ed8de2`). The isolated overlays solve runtime API availability, not minimum dependency metadata; the verifier/sandbox floor and root lock are still unresolved.

**Established:** the upstream loss CPU oracle passes full-head, proportional-tail, complete-support, mask-alignment, and missing-evidence cases (`score-centering-upstream-r1/`). The isolated R4 port completed three guarded updates with bounded sampling (`logprobs=8`, `top_k=8`), finite loss/KL/gradient norm, and all 12 consumed trace IDs joined to stream/orchestrator records. However, head mass was about 1, tail scale was 0, and mean correction L1 only `6.6–7.0e-7`. This is an integration smoke with negligible effective correction, not learning, reward, quality, or speed evidence. See `icepop-rl-smoke-r4/{runtime-audit.md,runtime-audit.json,guard/result.json,frozen-source-hashes.json}` and Experiment 149 in `EXPERIMENTS.md`.

**Current full-suite evidence:** corrected broad CPU sweep: 315 passed, two exact-float assertion failures, 108 deselected. Trainer/orchestrator: 124 passed and two failed in test_advantage.py (0.699999988 versus 0.7). Inference/config: 191 passed after adding the required taskset package roots. Initial 39 setup failures are retained, not the final result. Exact commands/stdout/results: `icepop-full-tests-r1/`; published compact copies: `research/icepop-score-centering/`.

The guarded 10-update IcePop run **completed**: `icepop-rl-full-r1/summary.json`, guard exit 0, 65.10 s, GPU peak 5262 MiB, zero job swap, all ten loss/KL/gradient-norm values finite. All frozen source hashes match. No reward or wall-time improvement claim is established. The loss/std logging metric was dropped as non-finite. Resolve the two CPU precision assertions and coherent dependency metadata before activating the port.

Useful source-only API import command (the dependency overlay manifest has the validated invocation):

```sh
cd /home/pradheep/tiny-prl/prime-rl
ulimit -v 2097152
PYTHONPATH="$PWD/outputs/megakernel-20261025/icepop-port-r1/dependency-overlays/renderers:$PWD/outputs/megakernel-20261025/icepop-port-r1/dependency-overlays/verifiers" \
  uv run --offline --no-project --python /home/pradheep/.venv/bin/python python -c \
  'from renderers.client import _parse_compact_logprobs; from verifiers.v1 import SamplingMask, TopLogprobs; from verifiers.v1.trace import Trace, MessageNode'
```

## SmolLM2-360M decode

The acceptance distinction is important: R5 is a validated **single custom decoder block** at layer 0 with native layers 1–31. R5 passed four initial plus 30 held-out fixtures at the unchanged gates (block output and K/V <=1 BF16 ULP, logits <=2 ULP, exact greedy). On those held-out cases final norm/logits and block output/K/V were bitwise equal. R5 is not a full custom decoder or serving implementation. Its source/library hashes are in `block-360-cuda-r5/`; current hashes: `block360.cu` `a96278ecfa638ea146e1b5e2aeff142dec7a34affb0487dc22a3de72788d412e`, include `b66c7aab82de9a0fb286e43cde141cac586d4745a74e9d9aae17446da103b6d3`, library `c36871803e12a9e7583844e88e217c158b03affe93197e22fb5e7ce792fcf3d4`.

The first full 32-layer prototype is isolated at `decode-360-cuda-r1/`. It uses one cooperative grid, 32 weight/KV pointer entries, layer-0 embedding lookup, R5 block math, final RMSNorm, and optional per-layer post-block capture. It has a capped CPU compile and the full four-case GPU comparison below; its source/library hashes are `block360.cu` `1ca8038b741cc64139622e8e6e062fff7ce3907781cc6730d6953b40e05b6bec`, include `fb397b06d967af530a7e57eaed3e55dc57154494e77bda45e1d19c5c12970d0a`, library `ce7bad11b1877ade275978f1ad599b1ab6d5f785b8c90ac8c13f888bf0f5578a`. Do not infer correctness from the R5 block result or CPU compilation.

Full-model GPU evidence is `decode-360-validation-r3/results-r1/comparison.json`, tied to SmolLM2-360M snapshot `a10cc1512eabd3dde888204e902eca88bddb4951` (config SHA `224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee`; weights SHA `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`). At B1/B2 × context 8/32, three cases pass exactly; B2/context 32 fails strict layer/KV/logit gates although greedy IDs match exactly. The first post-block divergence is layer 15; the first K/V divergence is layer 16. At layer 15, 220/1920 output elements differ, first difference is 1 ULP and max absolute error is 0.0625. Layer-16 K/V has 305 differing elements, first difference 1 ULP and max absolute error 0.015625. Divergence propagates downstream; final hidden max absolute error is 0.15625 and logits max absolute error is 0.1875 across 45,782 differing logits. Maximum BF16 ULP values become large near zero, so use the reported actual values and absolute errors to diagnose; do not loosen the gates. Greedy match alone is not acceptance.

Next action: add or use internal per-stage capture around the failing layer-15 path and identify the first arithmetic stage that changes before layer-16 K/V. The current prototype exposes every post-block output but its internal debug buffer covers only one block, so it cannot identify all 32 layers' internal first divergence. Keep thresholds fixed. Run the corrected diagnostic in a new result directory and preserve the existing R1 results.

Prepared GPU command after root review:

```sh
cd /home/pradheep/tiny-prl/prime-rl
uv run --offline --no-project --python /home/pradheep/.venv/bin/python python \
  outputs/megakernel-20261025/decode-360-validation-r3/validate.py \
  --library outputs/megakernel-20261025/decode-360-cuda-r1/build/libdecode360_r1.so \
  --output-dir outputs/megakernel-20261025/decode-360-validation-r3/results-r1 \
  --device cuda:0
```

The validation harness has CPU syntax, helper, cache clone, and 27-parameter ABI preflights. The actual R3 full-model run above already exists; use a fresh output directory for any rerun. The R1/R2/R3 harness history and results are retained separately.

## KV-cache offload

The short-context R6 smoke is functionally exact for the tested 32 generated IDs and 512-token revisit, but one observation was slightly slower with CPU offload (21.584 ms baseline, 21.699 ms offload). The long-prefix test is a distinct component result: guarded A-B-B-A, three 8,000-token prefixes, 8 generated tokens/request, 192 MiB GPU cache and 768 MiB CPU pool; median 212.174 ms to 42.676 ms (4.9717×), all 96 generated IDs match, and all candidate samples beat controls. It applies to repeated long prefixes within one weight version under R6 eager decode; it does not establish short-context or full-RL gain. The salted cache freshness mutation smoke matches a full reset exactly. Evidence and guard outputs are linked in `ICEPOP_KV_CACHE.md` and Experiments 146–148.

## Continue safely

- Keep `examples/tiny/MEGAKERNEL_PLAN.md` as the megakernel design/acceptance plan, `EXPERIMENTS.md` as the published experiment ledger, and `ICEPOP_KV_CACHE.md` as KV-specific evidence. This handoff records where work stands; it does not replace those documents.
- Preserve the pinned environment and source snapshots. Keep experimental CUDA and port files in the existing ignored `outputs/megakernel-20261025/` paths; do not edit production kernels, dependency pins, locks, or dtype settings as part of diagnosis.
- Use one guarded GPU workload at a time. Record full command, source hashes, guard result, exact output directory, and numerical/workload scope. A partial run or failed guard is not a completed result.
- The user authorizes useful commits and pushes to the personal fork main branches. Stage explicit paths, preserve upstream repositories, and verify remote parity. The opt-in port is checkpointed as a reproducible patch until dependency metadata is coherent.

## Published checkpoint files

`research/icepop-score-centering/` contains the full replayable port patch, exact dependency manifest, patch/source-hash validation, CPU results/commands, and full 10-update RL result. The full 360M decoder prototype is archived in the separate prime-kernels fork under `research/smollm2_360m_decode/`, explicitly marked not accepted (3/4 strict cases). Resume with layer-15 internal-stage capture for B2/context32; do not relax gates.
