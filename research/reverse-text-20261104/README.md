# Reverse Text: Qwen3-0.6B laptop pilot

The pinned Reverse Text taskset completed an end-to-end RL run on the RTX 4060
Laptop. This verifies environment serving, rollouts, real optimizer updates,
LoRA broadcasts, checkpoint saving and evaluation of the final adapter.
It does not establish successful task learning.

| Held-out measurement | Base model | After 10 RL updates |
| --- | ---: | ---: |
| Exact-reversal pass@1 | 0% | 0% |
| Exact-reversal pass@4 | 0% | 0% |
| Mean similarity reward | 0.25970 | 0.27061 |
| Tagged answers | 121/128 | 123/128 |
| Truncated calls | 0/128 | 0/128 |
| Rollout errors / timeouts | 0 / 0 | 0 / 0 |

Each evaluation used the same 32 prompts and seeds 42, 43, 44 and 45, with one
sample per prompt per seed. Pass@1 is the fraction of exact answers across the
128 samples; pass@4 is the fraction of prompts solved in at least one of four
samples. Similarity is the taskset's `SequenceMatcher` ratio, independently
recomputed from the recorded answers. It is not exact accuracy.

The reward delta was +0.01091. A paired bootstrap over tasks, retaining their
four samples together, gave a 95% interval of [-0.00234, +0.02477]. This interval
includes zero; there was only one training run. The small reward increase is
inconclusive.

## Training and resources

- Fresh Qwen/Qwen3-0.6B with LoRA rank 8, alpha 16; no Wordle adapter.
- Ten updates, batch 8, group size 4, learning rate 1e-5, sequence length 512,
  completion limit 192 and thinking disabled. Optimization and reduction stayed
  FP32. vLLM and training shared one physical GPU.
- All ten updates had finite loss and nonzero gradient norms (0.1620–0.5868).
  All 196 final LoRA B tensors contained nonzero values.
- The inference log records adapter loads through step 10 with HTTP 200.
  The final step-10 adapter was loaded again for the held-out evaluation.
- Guarded launch through shutdown took 74.61 seconds in this one pilot.
  Peak sampled GPU memory was 5,878 MiB; minimum available host RAM was
  5.23 GiB. Job swap stayed zero; the guard did not stop the run.
- A final resumable checkpoint exists under
  `outputs/reverse-text-20261104/rl/qwen600m-r1/checkpoints/step_10`.

The data came from `PrimeIntellect/Reverse-Text-RL`: select its 256 shortest
prompts, hold out the shortest 32, and train on the remaining 224. Prompt
lengths in this subset span 20–68 characters. Train and test have no prompt
overlap, but test prompts are shorter than training prompts. This intentionally
easy pilot does not represent the full upstream distribution.

## Evidence and rerunning

`result.json` contains the measurements and resource summaries; `manifest.json`
records split hashes and the starting code revision. Local raw traces, configs,
data, evaluation scripts and guard logs are in
`outputs/reverse-text-20261104/`. The pinned environment was not modified.

From this checkout, the existing local data/config can launch another guarded
pilot with a fresh run name:

```bash
PYTHONPATH="$PWD/deps/verifiers/environments/reverse_text" \
  bin/tiny-prl train @ outputs/reverse-text-20261104/rl.json \
  --guard-timeout 600 --run.name qwen600m-r2 \
  --output-dir "$PWD/outputs/reverse-text-20261104/rl"
```

The first evaluation attempt failed config validation because it reused
Wordle's `player` role. Reverse Text uses `agent`; correcting the experiment
config resolved this. No production pipeline changes were needed. A few
single-microbatch `loss/std` metrics were dropped as undefined; actual losses
and gradients remained finite.

Next: test a shorter-string curriculum or use the upstream SFT warmup before
a longer RL run. Keep exact-reversal accuracy alongside similarity reward.
