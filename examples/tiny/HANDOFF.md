# tiny-prl research handoff

Checkpoint: 2026-10-26. This note captures the current state of the IcePop integration, 360M decode work, and KV-cache work. The main-worktree CPU suite and all three 10-update RL runtime arms completed (bounded IcePop, untruncated IcePop, and default IPO).

## Repository and environment boundaries

- Prime-RL working repository: `/home/pradheep/tiny-prl/prime-rl`. The published IcePop implementation is commit `c4c7722ec`; the later documentation checkpoint is `b0214cf8f`. Local `upstream/main` is `40babc0833dda27f6c23ab53547dd527e465ae38`. The origin is the personal fork; upstream is PrimeIntellect.
- Prime-Kernels checkout: `/home/pradheep/tiny-prl/prime-kernels`. Local `main`, `origin/main`, and the Prime-RL submodule are `8ba238c`. This revision checkpoints the 360M R7 research decoder; GitHub Actions workflows remain removed and the active production kernel remains R6.
- The main worktree now contains the opt-in IcePop integration and related compatibility changes. Concurrent research edits include `examples/tiny/EXPERIMENTS.md`, `examples/tiny/hillclimb.png`, and `tools/plot_tiny_hillclimb.py`; preserve them. The primary progress image remains `examples/tiny/hillclimb.png`.
- Active dependency submodules are verifiers `1e80567bc3e80de16a73f0fa5e0ee93b15d1cdca` and renderers `70c8204bbce9c0afa7de746c1e249a717980f8dc`. The pinned `/home/pradheep/.venv` remains read-only. The offline lock check did not complete because the cached vLLM 0.31 wheel URL was unavailable; no lockfile edit, download, or environment sync was made. Model dtype settings remain unchanged.

## Score-centering / IcePop

Prime-RL upstream #4012 is `90fbb8db28252f509b9add326aaa37fab4ce2a87`; bounded replay #4176 is `735496e36fc9cb82e94b50098841bb80b2a0aaa1`. The main-worktree integration is opt-in through `trainer.loss.type = "icepop"` and `trainer.loss.score_centering = true`. See `ICEPOP_KV_CACHE.md` for bounded-exact and untruncated-proportional-tail invocations based on the complete `smollm2-full-ipc.toml` example.

**CPU validation:** main-checkout tests passed: trainer/orchestrator 126; inference/config 191; 317 total, zero failures and 108 deselected. The run used the pinned submodule source directly, with no overlays. Commands, stdout, import origins and result JSON are in `outputs/icepop-main-tests-r1/`. `test_load_configs` was excluded because it requires external environments; GPU-marked tests were deselected.

**Runtime status:** bounded replay completed ten updates in `outputs/megakernel-20261025/icepop-main-full-r1/`; untruncated sampling completed ten updates in `outputs/megakernel-20261025/icepop-main-untruncated-r2/`. Both guards returned 0 with zero job swap. Both runs have 40 trainer trace IDs joined to orchestrator annotations and stream records. Bounded replay (`top_p=.99`, `top_k=8`, eight logprobs) recorded complete sampler support: head mass was approximately 1 and no tail estimate was used. Untruncated sampling (`top_p=1`, no `top_k`, eight logprobs) recorded head mass between 0.82 and 0.94 with proportional tail scale approximately 1. Both runs had zero clipped tokens, so the correction was at floating-point-noise scale for bounded replay and zero for untruncated sampling. This establishes runtime integration and evidence alignment, not a meaningful centering effect, learning gain, reward gain, or speed gain. A separate default-IPO reference completed ten finite updates with guard code 0, zero swap, and 40/40 trainer trace joins. It is functional context only, not a paired performance or quality control for the IcePop recipes. Overall, the three RL runs provide 30 completed main-worktree updates. Machine-readable audit: `outputs/megakernel-20261025/icepop-main-acceptance-r1/summary.json`; compact public copy: `research/icepop-score-centering/main-integration-result.json`.

The offline lock check could not complete because its cached vLLM 0.31 wheel URL was unavailable. This is a cache limitation, not evidence of a dependency conflict. No dependency download or environment sync was attempted. The lockfile was not changed.

## SmolLM2-360M decode

The portable archive at `prime-kernels/research/smollm2_360m_decode/` now contains the exact compiled R7 source and include. R7 is the full 32-layer cooperative decoder with R6's four-lane QK reduction and an updated RMSNorm reduction sequence. Its archived source/include/library SHA256 values are `0f24bb3a9502310e1e25fcd7c40f0ec299dd2a0a018a5b2cd4850f231e709f66`, `e42fffa718488910810b48ee3c8581d306fbde389b70f980d33005bd8a6e73af`, and `efe0bd039e6aa2146ade424cffb5e5889e80189bc377d593286f922d40b679e1`.

R7 passes the initial four B1/B2 context-8/32 fixtures bitwise. On the unchanged 30-case held-out suite it passes 24 and fails 6 numerical gates; all greedy IDs match. It repairs both R6 norm-stage failures without a pass-to-fail case, but remains unaccepted. R13 completed all 30 cases; the guard returned 1 for numerical failures and reported no resource stop. No speed, serving, reward, or quality gain is established.

Evidence: initial four `outputs/megakernel-20261025/decode-360-validation-r10/results-r2-r7/`; full held-out `block-360-heldout-r13/`; RMS primitive replay `norm360-probe-r3/results-r3/`; build provenance `decode-360-cuda-r7/provenance.json`. The archive `validation-summary.json` retains the compact R4 and R6 history. Six held-out failures remain under the unchanged gates.

## KV-cache offload

The short-context R6 smoke is functionally exact for the tested 32 generated IDs and 512-token revisit, but one observation was slightly slower with CPU offload (21.584 ms baseline, 21.699 ms offload). The long-prefix test is a distinct component result: guarded A-B-B-A, three 8,000-token prefixes, 8 generated tokens/request, 192 MiB GPU cache and 768 MiB CPU pool; median 212.174 ms to 42.676 ms (4.9717×), all 96 generated IDs match, and all candidate samples beat controls. It applies to repeated long prefixes within one weight version under R6 eager decode; it does not establish short-context or full-RL gain. The salted cache freshness mutation smoke matches a full reset exactly. Evidence and guard outputs are linked in `ICEPOP_KV_CACHE.md` and Experiments 146–148.

## Continue safely

- Keep `examples/tiny/MEGAKERNEL_PLAN.md` as the megakernel design/acceptance plan, `EXPERIMENTS.md` as the published experiment ledger, and `ICEPOP_KV_CACHE.md` as KV-specific evidence. This handoff records where work stands; it does not replace those documents.
- Preserve the pinned environment and source snapshots. Keep experimental CUDA and port files in the existing ignored `outputs/megakernel-20261025/` paths; do not edit production kernels, dependency pins, locks, or dtype settings as part of diagnosis.
- Use one guarded GPU workload at a time. Record full command, source hashes, guard result, exact output directory, and numerical/workload scope. A partial run or failed guard is not a completed result.
- The user authorizes useful commits and pushes to the personal fork main branches. Stage explicit paths, preserve upstream repositories, and verify remote parity. The opt-in port is published in the personal Prime-RL fork as implementation commit `c4c7722ec`; the pinned environment remains unchanged. The offline lock check could not complete because the cached vLLM 0.31 wheel URL was unavailable.

## Published checkpoint files

`research/icepop-score-centering/` contains the full replayable port patch, exact dependency manifest, patch/source-hash validation, CPU results/commands, and main integration audit for three 10-update runs. The full 360M decoder archive is in the separate prime-kernels fork under `research/smollm2_360m_decode/`; its four initial fixtures pass but ten of 30 held-out cases fail, so it remains unaccepted. Continue with direct native-versus-candidate QK/PV/reduction diagnosis and validate any changed candidate against the unchanged gates; do not relax them.
