# SM89 model-specific decode megakernels

## Objective

Build and validate shared megakernel machinery with specialized schedules for
SmolLM2-135M-Instruct, SmolLM2-360M-Instruct and Qwen3-0.6B on the RTX 4060 Laptop
GPU. Start with rollout decoding, then measure the effect on complete RL runs.
Luna agents implement scoped components; the root agent reviews and serializes
guarded GPU execution.

## Execution gates

1. Freeze CUDA-graph-enabled vLLM baselines for batch sizes one and two. Record
   model/config hashes, dependency versions, context lengths, generated token
   counts, seeds, numerical outputs, warmup policy and resource measurements.
   Separate startup, prefill, warmed decode and request orchestration time.
2. Profile the current decode path and establish an SM89 implementation plan
   from actual model shapes. Account for synchronization, residency, register
   and shared-memory pressure; do not assume Hopper or Blackwell instructions.
3. Implement and validate one complete SmolLM2-135M decoder block, including
   attention, KV-cache updates and MLP. Label partial fusion as partial fusion.
   Define numerical acceptance before performance measurement, preserving
   reference dtype boundaries and reporting maximum errors and token parity.
4. Extend the execution schedule to the complete decode forward pass. Reuse
   operation implementations across the three model schedules. Add each model
   only after correctness and memory gates pass. Model weights must remain
   replaceable after RL updates, with no stale packed or cached weight values.
5. Integrate an opt-in backend with explicit unsupported-shape fallback. Verify
   weight reloads and complete RL updates before claiming RL wall-time gains.
6. Compare matched warmed trials against the graph-enabled baseline, then
   matched complete RL workloads. Record rejected candidates and measured
   improvements in EXPERIMENTS.md and append to the existing hill-climb figure
   without changing its fonts, colors or layout. Push verified milestones to
   the personal forks.

## Constraints

- Run one GPU job at a time under the established memory, disk and GPU guards.
- Keep the pinned `/home/pradheep/.venv` unchanged; use existing dependencies
  through offline UV execution. Do not change optimization or reduction dtype.
- Keep all existing benchmark evidence and checkpoints. Use new experiment
  directories under `outputs/megakernel-20261025/` for this work.
- Develop kernel code in the sibling `prime-kernels` checkout. The training
  repository continues to pin its tested kernel dependency as a submodule.
- A faster isolated operation does not establish a faster complete block,
  decode pass or RL run. Report each measurement at its actual scope.

## Initial work

- `decode-baseline-r1`: graph-enabled inference baseline harness.
- `design-r1`: model shapes and SM89 scheduling design.
- `block-prototype-r1`: first decoder-block implementation and parity probe.

