# tiny-prl: next ideas (research shortlist, nothing implemented yet)

This list ranks ideas from public RL systems and kernel work against what this
laptop actually spends time on. Every expected gain is an estimate until a
matched A-B-B-A measures it; experiment numbers refer to `EXPERIMENTS.md`.

## Status

| Item | Outcome | Ledger |
|---|---|---|
| A1 persistent harness worker (warm fork server) | Verified: RL wall −23.2%, enabled in the SmolLM2 profile | Exp 180 |
| A5 more rollouts in flight (after A1) | No gain (wall 0.0%); `max_inflight` stays 4 | Exp 181 |
| A4 overlap trainer init with vLLM startup | Verified: RL wall −13.7%, enabled in the SmolLM2 profile | Exp 182 |
| A3 warm vLLM across runs | Blocked by the shared-GPU / CUDA IPC config contracts; not pursued | Exp 183 |
| A2 skip zero-signal prompts | Not run: changes the training data, outside "task fixed"; needs explicit approval | — |
| B3 360M megakernel in serving | R25 slower than stock in serving; 360M routing is now research-only | Exp 184 |
| B1 speculative decoding (n-gram) | Rejected: decode ~2x slower per token (V1 runner, no async scheduling); wall not improved | Exp 185 |
| B2 partial rollouts | Not run: mixes policy versions inside a sample (algorithm change) and needs long generations | — |
| B4 135M megakernel R7 | Not run: decode is a few seconds per run and earlier megakernel gains did not move wall time (Exp 129, 134) | — |
| B5 FP8 rollouts | Not run: changes rollout numerics, outside "algorithm fixed" | — |
| B6 prefix caching | Already on (vLLM default `enable_prefix_caching`) | — |
| C1 batch-invariant kernels | Not run: numerics project, not a speed change | — |
| C2 FP32 LM head | Already on (`enable_fp32_lm_head`) | — |
| C3 / C4 loss and LoRA changes | Not run: algorithm changes | — |
| D1 FP16 accumulation | Not run: changes reduction precision | — |
| D2 / D3 trainer kernels | Not run: trainer kernels do not move wall time here (Exp 175, 177) | — |
| D4 CUDA Tile IR | Not available: SM89 support starts in CUDA 13.2; the pinned toolkit is 13.0 | — |

## What the measurements say

A ten-update SmolLM2-135M full-parameter RL run takes about 60 s after the
cleanup fix (Exp 137, 177):

- ~34 s startup: vLLM is ready at ~26 s, and the trainer starts only after that (Exp 126).
- ~31 s rollout/training loop. The GPU is busy 18% of it (Exp 126).
- Each rollout is ~1.3 s: 0.86 s environment harness (a fresh Python
  subprocess per rollout), 0.18 s model and 0.29 s setup (Exp 126).
- ~70% of rollouts are discarded as `no_signal`, because both group members score the same
  (Exp 102, 126). About 116 rollouts are generated for 40 used samples.
- Decode is ~6 s of the run, so kernel speedups to decode and to the trainer do
  not move wall time at this size (Exp 129, 134, 175, 177).

So the order is: per-rollout overhead and wasted rollouts, then startup,
then inference for long generations, then trainer kernels.

## A. RL wall time on this profile (biggest expected wins)

| # | Idea | Source | Expected effect here | Cost / risk |
|---|---|---|---|---|
| A1 | **Persistent harness worker.** Keep one warm chat-program process (or an in-process client) per worker and feed it rollouts, instead of spawning a Python subprocess per rollout. | Exp 126 measurement; Prime Sandboxes keep environment latency close to local processes (INTELLECT-3 report). | Removes most of the 0.86 s per rollout, so the loop could drop from ~31 s toward ~10–15 s. Largest single lever. | A change in `verifiers` (submodule); same task and reward. Needs isolation between rollouts. |
| A2 | **Don't generate zero-signal groups.** Pre-filter or "snooze" prompts the policy always gets right or always gets wrong, and size the batch with DAPO-style dynamic sampling. | DAPO; AI21 online difficulty filtering (~3× compute efficiency on GSM8K); DEPO pre-rollout difficulty estimation; prime-rl curriculum gates. | Up to ~3× fewer generated rollouts for the same 40 trained samples. | **Changes the data distribution, so it needs your approval** under "task and algorithm fixed". |
| A3 | **Keep vLLM warm across runs.** A persistent inference server, plus a weight reset at run start, so experiments skip the ~26 s vLLM startup. | TRL server mode; checkpoint-engine-style in-place weight updates. | −20 to −25 s per run for repeated experiments. | Changes the run lifecycle; must prove the weights reset exactly. |
| A4 | **Overlap trainer init with vLLM startup.** | Exp 126 lever 3. | Up to −8 s. | Both share GPU memory during vLLM's memory profiling, so the ordering must be safe. |
| A5 | **Re-test higher `max_inflight` after A1.** | Exp 133 (loop −10%, wall not verified). | Probably larger once per-rollout time is CPU work rather than subprocess startup. | Config only; watch staleness. |

## B. Long-generation inference (for future environments)

| # | Idea | Source | Expected effect | Cost / risk |
|---|---|---|---|---|
| B1 | **Speculative decoding for rollouts.** Use draft-free variants first: n-gram/suffix drafts that reuse earlier rollouts (SPEC-RL, BubbleSpec). Then try a small draft or MTP head trained online during RL. | arXiv 2604.26779 (EAGLE-3: rollout latency 1.5–1.8×, step time up to 1.41× at 8B); Intern-S2 (~2× rollout with an online-trained draft); NeMo RL Eagle3 guide; EfficientRollout (it helps most in the shrinking-batch tail). | Large for long, tail-heavy generations; small for today's 64-token completions. | Draft drift; vLLM spec-decode support with our CUDA graphs; logprobs must stay those of the target model. |
| B2 | **Partial rollouts and in-flight weight updates.** Pause, update weights, resume unfinished generations instead of restarting them. | Kimi k1.5 partial rollouts; verl `partial_rollout`; PipelineRL; vLLM async-RL pause/resume API. | Removes the long-tail stall at each weight update. | Mixed-policy tokens inside a sample need importance correction (IcePop/TIS already exist). |
| B3 | **Finish the 360M decode megakernel in serving.** Run the R25 serving A-B-B-A (interrupted), then fix the batch-two serving gap (R22: B1 ≈ stock, B2 ~2.2× slower). | Exps 158–174; Hazy "No Bubbles" (instruction scheduler, overlapped weight loads, 78% of H100 bandwidth). | Decode-bound long generations only. | R25 package swap is prepared but uncommitted in `prime-kernels`. |
| B4 | **135M megakernel R7.** `cp.async` double-buffered weight tiles; a two-level partial merge (up to 3.5 MB of reads per layer at long context); cap the 255-register spill. | Exp 130–136 notes; ThunderKittens-style tiles on SM89. | R6 is at ~77% of the measured 250 GB/s weight-stream floor, so ≤ ~20% left. | Diminishing returns. |
| B5 | **FP8 (W8A8) rollouts on SM89.** Ada has FP8 tensor cores, and vLLM ships CUTLASS SM89 FP8 kernels. | verl FP8 rollout; vLLM/vime low-precision guide; Unsloth FP8 GRPO on RTX 40 (~1.4× inference, vendor claim). | Up to ~1.4× decode for larger models; little for 135M. | Raises inference/trainer mismatch, so measure `mismatch_kl` with IcePop on. Changes the rollout policy numerics. |
| B6 | **Prefix-cache reuse for group rollouts.** Group members share the prompt. | Exp 148 (long-prefix KV reuse 4.97× on the component). | Prefill savings for long prompts. | Already measured as a component; needs serving integration. |

## C. Numerics and on-policy consistency

| # | Idea | Source | Why |
|---|---|---|---|
| C1 | **Batch-invariant sampler and trainer kernels (true on-policy).** | vLLM + TorchTitan bitwise-consistent RL (Qwen3-1.7B); Thinking Machines "Defeating Nondeterminism"; miles true on-policy; SkyRL IsoExec. | Reports show reward loss without it; our Exps 123–125 already reproduce cuBLAS orders bit for bit, which is the same machinery. |
| C2 | FP32 LM-head logits. | ScaleRL (largest single precision effect). | prime-rl already defaults `enable_fp32_lm_head = true`; keep it. |
| C3 | CISPO loss, prompt-level loss averaging, batch-level advantage normalisation. | ScaleRL recipe. | **Algorithm changes: need your approval.** |
| C4 | Rank-1 to rank-32 LoRA for RL on Qwen3-0.6B. | Thinking Machines "LoRA Without Regret" (a Qwen3-0.6B replication favoured rank 64 over rank 1). | Memory and step-time trade; affects learning. |

## D. Trainer kernels (lowest priority: Exp 175 and 177 show no wall-time effect)

| # | Idea | Source | Notes |
|---|---|---|---|
| D1 | **FP16-accumulate GEMMs.** GeForce tensor cores run FP16 with FP16 accumulate at 2× the FP32-accumulate rate; `torch.backends.cuda.matmul.allow_fp16_accumulation` turns this on without hacks. | PyTorch CUDA notes; RTX 4090 spec (330 vs 165 TFLOPS). | This is the safe "unlock" for the 40 series. It changes reduction precision, which conflicts with "keep reduction dtypes fixed", so it is research only and needs strict parity gates. |
| D2 | Helion (autotuned "PyTorch with tiles" to Triton) for RMSNorm/SwiGLU fwd/bwd variants. | PyTorch Helion blog; LLM-guided autotuning. | Faster kernel iteration; benchmarks are datacentre GPUs. |
| D3 | Liger-style fused linear cross-entropy and chunked GRPO loss. | Liger Kernel; TRL integration. | prime-rl already has a fused, chunked LM head (Exp 7); the Triton backward variant fails strict gradient parity at 135M (Exp 176). |
| D4 | CUDA Tile IR / cuTile Python. | NVIDIA CUDA 13.1+; GPU MODE lecture 89. | Track it; check SM89 support before investing. |
| — | **Firmware "unlock" (gpu-unlock-ada).** | pasqualepillitteri.it coverage; github.com/JustAResearcher/gpu-unlock-ada. | **Rejected.** Laptop GPUs are not supported; READMEs warn of hardware damage; it only lifts the FP32-accumulate cap that D1 avoids in software; decode is memory-bound, so no token-speed gain. |

## Suggested order

1. A1 persistent harness (largest measured cost), then A5 re-test.
2. A3 or A4 to cut startup.
3. A2 if you approve the data-distribution change.
4. B3: finish the R25 serving A-B-B-A and commit or revert the prepared package swap.
5. B1 and B2 once an environment with long generations exists to measure them.

## Sources

- [INTELLECT-3 blog](https://www.primeintellect.ai/blog/intellect-3), [INTELLECT-3 report](https://arxiv.org/abs/2512.16144)
- [Mercor: acquiring Deeptune (RL environments)](https://www.mercor.com/blog/mercor-to-acquire-deeptune/)
- [MoonshotAI checkpoint-engine](https://www.marktechpost.com/2025/09/15/moonshotai-released-checkpoint-engine-a-simple-middleware-to-update-model-weights-in-llm-inference-engines-effective-for-reinforcement-learning/)
- [verl fully async / partial rollout](https://verl.readthedocs.io/en/latest/advance/fully_async.html), [verl FP8 RL](https://verl.readthedocs.io/en/latest/low_precision/fp8.html)
- [miles on the PyTorch blog](https://pytorch.org/blog/miles-a-pytorch-native-stack-for-large-scale-llm-rl-post-training/), [LMSYS miles](https://www.lmsys.org/blog/2025-11-19-miles)
- [TRL co-located vLLM](https://huggingface.co/blog/vllm-colocate), [TRL Liger integration](https://huggingface.co/docs/trl/main/en/liger_kernel_integration)
- [vLLM async RL](https://docs.vllm.ai/en/stable/training/async_rl/), [bitwise-consistent on-policy RL](https://vllm.ai/blog/bitwise-consistent-train-inference), [IsoExec](https://vllm.ai/blog/2026-08-21-isoexec)
- [ScaleRL](https://arxiv.org/abs/2510.13786), [DAPO](https://arxiv.org/abs/2503.14476)
- [System-integrated speculative decoding for RL](https://arxiv.org/abs/2604.26779), [SPEC-RL](https://arxiv.org/abs/2509.23232), [BubbleSpec](https://arxiv.org/abs/2605.08862), [EfficientRollout](https://arxiv.org/abs/2606.18967), [NeMo RL Eagle3](https://docs.nvidia.com/nemo/rl/latest/guides/eagle3-speculative-decoding.html)
- [Hazy "No Bubbles" megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles), [Hazy throughput megakernel](https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main)
- [Helion](https://pytorch.org/blog/helion/), [CUDA Tile IR](https://docs.nvidia.com/cuda/archive/13.2.2/tile-ir/main/sections/prog_model.html), [GPU MODE lectures](https://www.beri.net/learning/gpu-mode-lectures)
- [PyTorch FP16 accumulation](https://docs.pytorch.org/docs/stable/notes/cuda), [RTX 4090 FP16 GEMM discussion](https://discuss.pytorch.org/t/slow-fp16-gemm-on-4090/200232)
- [GPU firmware unlock coverage](https://pasqualepillitteri.it/en/news/27526/rtx-20-30-40-gpus-hacked-firmware-exploit-tensor-cores)
- [LoRA Without Regret (TRL guide)](https://huggingface.co/docs/trl/main/en/lora_without_regret), [Unsloth FP8 RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/fp8-reinforcement-learning)
