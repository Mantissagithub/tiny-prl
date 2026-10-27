# Kernel research direction, 2026-10-27

The 360M decoder currently has a correctness result on the original 34 single-step fixtures, not a serving speedup. R12 passes those 34 fixtures and 6/12 wider seed-777 cases. Keep the remaining numerical failures visible and keep the production 135M R6 path unchanged.

The first completed same-harness R4 serving pilot is a regression. Scheduler-inclusive median intertoken latency is:

| Prompt tokens | Batch | Stock vLLM | R9 experimental decoder |
| --- | --- | --- | --- |
| 32 | 1 | 4.210 ms | 17.412 ms |
| 32 | 2 | 3.952 ms | 9.163 ms |
| 128 | 1 | 4.222 ms | 17.527 ms |
| 128 | 2 | 3.959 ms | 9.373 ms |

These are separate stock-then-candidate pilot sessions with three warmups and seven repeats per shape, not an A-B-B-A acceptance result. Both use the same in-process R4 harness. The measurements include scheduling and the LM head, so they do not establish the decoder kernel's own duration. Evidence: `outputs/megakernel-20261025/decode-baseline-r4/results-r2-stock/stock-r2.json` and `results-r2-candidate/candidate-r2.json`. Token/logprob parity must be audited separately before any serving acceptance.

## Experiments in order

1. Attribute the regression. Profile actual custom-kernel duration, native GPU work, graph replay, and CPU gaps under the same four workloads. Record clocks, power, memory, route proof, and warmup. A faster primitive does not establish a faster engine.
2. Resolve wider arithmetic failures with isolated captured-input QK and PV probes. The independent PV accumulator/y-group tree probe tests a missing reduction-order hypothesis; the QK probe checks scalar, four-lane, and WMMA orders. Re-run all 46 frozen fixtures after any decoder change. Preserve the existing per-layer gates.
3. If profiling identifies weight-load stalls, test a short load lookahead before a larger pipeline. Preserve FP32 accumulation order and BF16 rounding boundaries. Inspect register count and spills as well as latency. The B1 `dots`/`dots16` loops and B2 `wmma_project_tile` are concrete starting points in the experimental package's `csrc/block360.cu`.
4. Test SM89-compatible `cp.async` staging only where the first measurement justifies it. Bound shared memory and register pressure; measure the effect of extra staging and barriers. Never assume a dense GEMM tile is appropriate for batch 1 or 2.
5. Consider finer dependency counters only if profiling shows material grid-barrier idle time. Prefetching immutable weights before waiting for activations is a possible overlap opportunity; policy-weight upload/version boundaries must remain correct.
6. Accept a speed change only after paired engine A-B-B-A runs with token/probability checks. Then run full-parameter RL with joined rollout/trainer evidence and paired wall-time measurements. Update the RL wall-time panel only from that result.

## Primary-source inspiration and limits

[Hazy Research's Llama-1B megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) motivates continuous weight loading and removing synchronization gaps between operations. Its H100 bandwidth and speed figures are not predictions for this RTX 4060 laptop. First measure whether the current decoder is limited by bandwidth, dependent arithmetic, or synchronization.

[Cohere's megakernel account](https://cohere.com/blog/megakernels) describes starting weight loads before activation dependencies resolve and using finer dependency tracking. It also describes the complexity of shared-memory management. Borrow the small overlap mechanism before considering a general task runtime. Its Hopper-oriented mechanisms require hardware-specific adaptation.

[NVIDIA's SM80 CuTe tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu) demonstrates asynchronous global-to-shared copies and staged pipelines. [CuTe DSL's cp.async API](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html) provides a possible implementation route. Verify the installed 4.7.1 API before coding: current online documentation may describe a newer version. SM89 supports the older asynchronous-copy path; Hopper TMA/WGMMA are not this device's implementation path. Keep the pinned environment unchanged.

All pipeline and dependency proposals above are hypotheses until profiled and measured on this laptop. They are not recorded improvements.
