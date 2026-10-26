# IcePop score centering and native KV offload

Initial source-audit baseline: upstream `40babc0833dda27f6c23ab53547dd527e465ae38`,
local `8fa401ebc0a5e14495161bf8c8778e93bb28fcc6`, inspected 2026-10-26.
These are separate experimental directions. The score-centering integration
is now in the main worktree behind an opt-in loss flag. CPU regression tests
pass, and bounded, untruncated, and default-IPO runtime runs completed ten
updates each. KV offload results
below are independent of score centering.

## Using score centering

The main-worktree port is opt-in with `trainer.loss.type = "icepop"` and
`trainer.loss.score_centering = true`. The setting is validated for policy
sampling and requests sampler probabilities automatically. Keep the model
dtype settings from the complete base configuration unchanged.

Start from the complete shared-GPU example, which already contains the model,
deployment, renderer, and task configuration:

```sh
uv run --offline rl @examples/tiny/smollm2-full-ipc.toml \
  --trainer.loss.type icepop \
  --trainer.loss.score-centering true \
  --trainer.loss.score-centering-topk 8
```

For **untruncated sampling**, leave `top_p = 1.0` and `top_k` unset in the
training sampling config. The example requests an eight-token probability head;
the loss keeps full-vocabulary normalization and estimates the unobserved tail
proportionally. This is an approximation, not exact full-vocabulary score
centering.

For **bounded exact replay**, use this full invocation instead:

```sh
uv run --offline rl @examples/tiny/smollm2-full-ipc.toml \
  --trainer.loss.type icepop \
  --trainer.loss.score-centering true \
  --trainer.loss.score-centering-topk 8 \
  --orchestrator.train.sampling.top-k 8
```

With replay support complete, centering is exact on that recorded support and
there is no modeled tail. The option still uses eight as the head-size setting
for untruncated sources if the same config contains them. Do not set `top_p < 1`
for this recipe unless the resulting support is also bounded. Config validation
enforces the captured-head/replay requirements.

The compact per-token evidence adds top-token IDs and logprobs plus server-side
extraction and trainer storage. At top-128, the raw IDs and float32 logprobs are
about 1 KiB per scored token before framing. This is a learning-estimator change;
the overhead and throughput effect require measurement.

## Main-worktree integration status

The opt-in port and compatibility changes are present in the main worktree. The
verifier submodule is pinned to `1e80567bc3e80de16a73f0fa5e0ee93b15d1cdca`
and renderers to `70c8204bbce9c0afa7de746c1e249a717980f8dc`. A direct main-tree
CPU regression sweep passed 317 tests (126 trainer/orchestrator and 191
inference/config), with 108 deselected; no GPU test was run. All 32 declared
verifier environment requirements were already satisfied by the pinned local
environment.

Two guarded 10-update main-worktree runs completed with code 0, zero job swap,
and all 40 trainer trace IDs joined to orchestration annotations and stream
records. Bounded replay used `top_p=0.99`, `top_k=8`, eight recorded logprobs,
and sampling-mask capture; sampler-head mass was approximately 1 and no tail
estimate was used. Untruncated sampling used `top_p=1.0`, no `top_k`, and eight
recorded logprobs; the head held 0.82–0.94 of sampler probability mass and the
proportional tail scale was approximately 1. Both runs had zero clipped tokens,
so measured centering corrections were at floating-point-noise scale in the
bounded run and zero in the untruncated run. These runs establish integration
and evidence alignment, not a learning, reward, or speed improvement. Runtime
and trace details are in
`outputs/megakernel-20261025/icepop-main-acceptance-r1/summary.json`.

A separate default-IPO functional reference also completed ten finite updates
with guard code 0, no swap, and 40/40 trainer trace joins. It used untruncated
sampling with no score-centering capture. It is not a matched performance or
quality control for either IcePop recipe. Together, the bounded, untruncated,
and IPO runs provide 30 completed main-worktree updates; they do not establish
a learning, reward, or speed benefit.

The existing lock was not changed. An offline lock check could not complete
because the cached vLLM 0.31 wheel URL was unavailable; no download or
environment sync was attempted. This is a cache limitation, not a dependency
conflict finding. The pinned environment and dtype settings remain unchanged.

## Upstream changes

- [#4012](https://github.com/PrimeIntellect-ai/prime-rl/commit/90fbb8db28252f509b9add326aaa37fab4ce2a87)
  adds opt-in IcePop score centering from captured sampler probabilities.
- [#4176](https://github.com/PrimeIntellect-ai/prime-rl/commit/735496e36fc9cb82e94b50098841bb80b2a0aaa1)
  extends it to bounded sampling replay. Complete recorded support gives exact
  centering on that support; an untruncated top-k head still uses a proportional
  tail approximation. This does not compress or offload KV tensors.
- [#3831](https://github.com/PrimeIntellect-ai/prime-rl/commit/1ed4cf1d8ed42659297380204236b0a73ac1ae9e)
  handles missing MoE routing records when offloaded KV prefixes are restored.
  Our three target models are dense, so this router-replay patch is not needed
  for their native KV-offload experiment.

Upstream currently uses vLLM 0.32; the pinned laptop environment remains 0.31.
The local native offload connector is already present. Score centering needs a
reviewed port across sampler capture, probability transport, batch alignment,
trainer head scoring, configuration and loss. A loss-only change is insufficient.

## Score-centering CPU checks

Luna prepared an isolated mathematical prototype. Review caught and corrected
an arbitrary-head-indexing bug and strengthened the initially weak tail checks.
The final CPU run passes four groups: permuted complete-head gradient against a
dense oracle; proportional tails with explicit mixed accept/reject cases and
nonzero correction gradients; restricted-support replay with zero outside-support
probability and no modeled tail; and malformed/missing replay evidence.

These tests cover the prototype, not the production sampler, transport or RL
integration. Top-128 evidence costs about 1 KiB per scored token for int32 IDs and
float32 logprobs before framing, plus server extraction and trainer storage.
This is a learning-estimator experiment, with throughput costs still to measure.

## Native offload smoke on the pinned stack

RTX 4060 Laptop, SmolLM2-135M BF16, stock eager decode, one request at a time.
Both arms use a fixed 16 MiB GPU KV budget and prefix caching. Only the candidate
adds a 64 MiB native CPU pool. Three disjoint 512-token prompts force eviction;
then the first prompt is repeated. Every request generates exactly eight tokens.

| Observation | GPU cache only | Native CPU offload |
|---|---:|---:|
| Cached tokens on revisit | 0 | 512 |
| Revisit wall time, one observation | 47.48 ms | 114.92 ms |
| Sampled peak GPU memory | 485 MiB | 487 MiB |
| Guard return code | 0 | 0 |
| Job swap | 0 | 0 |

All 32 generated token IDs match between arms. The CPU-cache reload works, but
this small-prefix observation is slower than recomputation. One run per arm,
cold-start effects and stock decode prevent a speedup or RL claim. Offload does
not automatically shrink the configured GPU pool.

Analytical live BF16 KV storage for four unique sequences, excluding block
rounding, metadata, allocation slack and prefix sharing:

| Model | 512 tokens each | 8192 tokens each |
|---|---:|---:|
| SmolLM2-135M | 45 MiB | 720 MiB |
| SmolLM2-360M | 80 MiB | 1280 MiB |
| Qwen3-0.6B | 224 MiB | 3584 MiB |

This favors investigating longer contexts and reused prefixes before enabling
CPU offload in the short-context RL baseline. Weight updates must invalidate
cached prefixes correctly before offload can be accepted in RL.

## R6 integration smoke

The same eviction/revisit workload also completed with production R6 enabled in
both arms. All 32 generated IDs match between those arms, and the revisit restores
512 cached tokens. R6 reports 28 decode calls without offload and 29 with offload:
the fully cached prefix leaves one input token that also takes the decode path.
Revisit time is 21.58 ms without offload and 21.70 ms with it. This single pair is
functional evidence, not a statistical speed comparison. Both guards return 0,
with zero job swap and at most 487 MiB sampled GPU memory.

An earlier R6 harness run generated successfully but failed its final diagnostic
RPC because callable serialization was disabled. Its evidence is preserved in
`kv-offload-r6-smoke-r1/`; revision 2 enables the same local diagnostic RPC setting
used by the existing vLLM harness. No environment package was changed.

## Long-prefix cache-capacity result

A fixed-work A-B-B-A test uses three unique 8000-token prompts, eight generated
tokens per request, a 192 MiB GPU KV pool and R6 eager decode. Each arm runs
three warmup requests followed by nine timed requests. GPU-only storage cannot
retain all three prefixes; the candidate adds a 768 MiB native CPU pool.

Across 18 timed requests per mode, median request time falls from **212.17 ms to
42.68 ms (4.97x)**. Every candidate sample beats every control sample. All 96
generated token IDs per arm match, and all timed candidate requests restore 8000
cached tokens while controls report zero. All four guards return 0 with no job
swap. GPU clocks were not recorded.

This result requires repeated long prefixes within one policy version and adds
host memory. It does not establish an RL wall-time gain, higher reward, less total
memory, or a benefit for the current short-context workload.

## Weight freshness

A controlled in-place mutation of the first MLP down-projection demonstrates
that resetting only GPU prefixes can still reload 512 old tokens from CPU and
change generated tokens/logprobs compared with fresh evaluation. A full connector
reset avoids that. This is a diagnostic, not a production weight-transport test.

The normal rollout path already adds a per-policy-version cache salt. In a second
controlled GPU test, changing that salt after the mutation gives zero cached tokens
and exactly matches tokens/logprobs from a full reset. Source tracing confirms the
salt travels through the environment, verifier, renderer, vLLM request and chained
block hashes into offload keys. This supports the existing namespace mechanism;
it does not prove correctness of every in-flight weight-update scenario.

## Actual upstream loss execution

The actual IcePop loss and centering helper extracted from upstream `40babc083`
pass CPU gradient-oracle checks for full heads, mixed accept/reject proportional
tails, complete bounded supports, loss masks and missing evidence. This advances
beyond the mathematical prototype; full sampler-to-trainer integration is still
being validated in an isolated worktree with source-only dependency overlays.
The global virtual environment and production defaults remain unchanged.

## Local evidence

Under `outputs/megakernel-20261025/`:

- `icepop-kv-audit-r1/`: upstream audit and passing CPU connector/config preflight.
- `score-centering-audit-r1/{validate.py,results.json,README.md}`.
- `icepop-parallel-r1/kv-budget.json`: cached model configs and hashes.
- `kv-offload-smoke-r1/{run.py,summarize.py,comparison.json,guard-*/result.json}`.
- `kv-offload-r6-smoke-r2/{run.py,summarize.py,comparison.json,guard-*/result.json}`.

Additional evidence: `kv-offload-long-r1/{run.py,summarize.py,comparison.json}`,
`kv-freshness-smoke-r1/`, `kv-freshness-salt-r1/`,
`kv-freshness-audit-r1/`, and `score-centering-upstream-r1/`.

The raw outputs stay local under the repository's existing ignore policy.
