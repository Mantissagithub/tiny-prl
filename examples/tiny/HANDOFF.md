# tiny-prl research handoff

Checkpoint: 2026-10-26. This note captures the current state of the IcePop integration, 360M decode work, and KV-cache work. The main-worktree CPU suite and all three 10-update RL runtime arms completed (bounded IcePop, untruncated IcePop, and default IPO).

## Repository and environment boundaries

- Prime-RL working repository: `/home/pradheep/tiny-prl/prime-rl`. The published IcePop implementation is commit `c4c7722ec`; the later documentation checkpoint is `b0214cf8f`. Local `upstream/main` is `40babc0833dda27f6c23ab53547dd527e465ae38`. The origin is the personal fork; upstream is PrimeIntellect.
- Prime-Kernels checkout: `/home/pradheep/tiny-prl/prime-kernels`. Local `main`, `origin/main`, and the Prime-RL submodule are `8b8f694`. This revision checkpoints the 360M R6 research decoder; GitHub Actions workflows remain removed and the active production kernel remains R6.
- The main worktree now contains the opt-in IcePop integration and related compatibility changes. Concurrent research edits include `examples/tiny/EXPERIMENTS.md`, `examples/tiny/hillclimb.png`, and `tools/plot_tiny_hillclimb.py`; preserve them. The primary progress image remains `examples/tiny/hillclimb.png`.
- Active dependency submodules are verifiers `1e80567bc3e80de16a73f0fa5e0ee93b15d1cdca` and renderers `70c8204bbce9c0afa7de746c1e249a717980f8dc`. The pinned `/home/pradheep/.venv` remains read-only. The offline lock check did not complete because the cached vLLM 0.31 wheel URL was unavailable; no lockfile edit, download, or environment sync was made. Model dtype settings remain unchanged.

## Score-centering / IcePop

Prime-RL upstream #4012 is `90fbb8db28252f509b9add326aaa37fab4ce2a87`; bounded replay #4176 is `735496e36fc9cb82e94b50098841bb80b2a0aaa1`. The main-worktree integration is opt-in through `trainer.loss.type = "icepop"` and `trainer.loss.score_centering = true`. See `ICEPOP_KV_CACHE.md` for bounded-exact and untruncated-proportional-tail invocations based on the complete `smollm2-full-ipc.toml` example.

**CPU validation:** main-checkout tests passed: trainer/orchestrator 126; inference/config 191; 317 total, zero failures and 108 deselected. The run used the pinned submodule source directly, with no overlays. Commands, stdout, import origins and result JSON are in `outputs/icepop-main-tests-r1/`. `test_load_configs` was excluded because it requires external environments; GPU-marked tests were deselected.

**Runtime status:** bounded replay completed ten updates in `outputs/megakernel-20261025/icepop-main-full-r1/`; untruncated sampling completed ten updates in `outputs/megakernel-20261025/icepop-main-untruncated-r2/`. Both guards returned 0 with zero job swap. Both runs have 40 trainer trace IDs joined to orchestrator annotations and stream records. Bounded replay (`top_p=.99`, `top_k=8`, eight logprobs) recorded complete sampler support: head mass was approximately 1 and no tail estimate was used. Untruncated sampling (`top_p=1`, no `top_k`, eight logprobs) recorded head mass between 0.82 and 0.94 with proportional tail scale approximately 1. Both runs had zero clipped tokens, so the correction was at floating-point-noise scale for bounded replay and zero for untruncated sampling. This establishes runtime integration and evidence alignment, not a meaningful centering effect, learning gain, reward gain, or speed gain. A separate default-IPO reference completed ten finite updates with guard code 0, zero swap, and 40/40 trainer trace joins. It is functional context only, not a paired performance or quality control for the IcePop recipes. Overall, the three RL runs provide 30 completed main-worktree updates. Machine-readable audit: `outputs/megakernel-20261025/icepop-main-acceptance-r1/summary.json`; compact public copy: `research/icepop-score-centering/main-integration-result.json`.

The offline lock check could not complete because its cached vLLM 0.31 wheel URL was unavailable. This is a cache limitation, not evidence of a dependency conflict. No dependency download or environment sync was attempted. The lockfile was not changed.

## SmolLM2-360M decode

The portable archive at `prime-kernels/research/smollm2_360m_decode/` now contains the exact compiled R6 source and include. R6 is the R4 full 32-layer cooperative decoder with one compute change: QK uses a generic four-lane FP32 SIMT reduction, while PV and RMSNorm remain unchanged. The archived source/include/library SHA256 values are `4c23ba8bdf28df56c5b4e08420f5fdf93083c07d4d792687534c87164204150f`, `ef5d60f0dd1d31fbebabad60f0f4734cdb18029101aafa2f642259001ead55ef`, and `240caca3a6a284ba4f76b13fe55349b238a0ef05014919cfb648627e26eb563a`.

R6 passes the initial four B1/B2 context-8/32 fixtures bitwise. On the unchanged 30-case held-out suite it passes 22 and fails 8 numerical gates; all greedy IDs match. It fixes two R4 failures without a pass-to-fail case, but remains unaccepted. R11 completed all 30 cases; its guard returned 1 for numerical failures and reported no resource stop. R12 recaptured the eight failed cases at their causal layers: attention Q/K/scores/probabilities match in all eight, six first differ at PV output, and two at norm2 output with exact input and weight. R12 is diagnostic only. No speed, serving, reward, or quality gain is established.

Evidence: initial four `outputs/megakernel-20261025/decode-360-validation-r10/`; full held-out `block-360-heldout-r11/`; targeted diagnosis `block-360-heldout-r12/`; build provenance `decode-360-cuda-r6/provenance.json`. The archive `validation-summary.json` keeps the concise R4 baseline history alongside current R6 results. Continue PV and norm investigation and rerun the unchanged 30-case gates after any arithmetic change. Do not treat the diagnostic subset as a full acceptance test.

## KV-cache offload

The short-context R6 smoke is functionally exact for the tested 32 generated IDs and 512-token revisit, but one observation was slightly slower with CPU offload (21.584 ms baseline, 21.699 ms offload). The long-prefix test is a distinct component result: guarded A-B-B-A, three 8,000-token prefixes, 8 generated tokens/request, 192 MiB GPU cache and 768 MiB CPU pool; median 212.174 ms to 42.676 ms (4.9717×), all 96 generated IDs match, and all candidate samples beat controls. It applies to repeated long prefixes within one weight version under R6 eager decode; it does not establish short-context or full-RL gain. The salted cache freshness mutation smoke matches a full reset exactly. Evidence and guard outputs are linked in `ICEPOP_KV_CACHE.md` and Experiments 146–148.

## Continue safely

- Keep `examples/tiny/MEGAKERNEL_PLAN.md` as the megakernel design/acceptance plan, `EXPERIMENTS.md` as the published experiment ledger, and `ICEPOP_KV_CACHE.md` as KV-specific evidence. This handoff records where work stands; it does not replace those documents.
- Preserve the pinned environment and source snapshots. Keep experimental CUDA and port files in the existing ignored `outputs/megakernel-20261025/` paths; do not edit production kernels, dependency pins, locks, or dtype settings as part of diagnosis.
- Use one guarded GPU workload at a time. Record full command, source hashes, guard result, exact output directory, and numerical/workload scope. A partial run or failed guard is not a completed result.
- The user authorizes useful commits and pushes to the personal fork main branches. Stage explicit paths, preserve upstream repositories, and verify remote parity. The opt-in port is published in the personal Prime-RL fork as implementation commit `c4c7722ec`; the pinned environment remains unchanged. The offline lock check could not complete because the cached vLLM 0.31 wheel URL was unavailable.

## Published checkpoint files

`research/icepop-score-centering/` contains the full replayable port patch, exact dependency manifest, patch/source-hash validation, CPU results/commands, and main integration audit for three 10-update runs. The full 360M decoder archive is in the separate prime-kernels fork under `research/smollm2_360m_decode/`; its four initial fixtures pass but ten of 30 held-out cases fail, so it remains unaccepted. Continue with direct native-versus-candidate QK/PV/reduction diagnosis and validate any changed candidate against the unchanged gates; do not relax them.
