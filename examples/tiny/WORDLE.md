# Prime Environments Hub: Wordle

The laptop command supports `primeintellect/wordle@0.1.7` through a taskset API
adapter. The reviewed Hub version is `eletiahb7kg3mkh4qbh6ixlw`; its original
`wordle.py` SHA-256 is
`8567b975122280ef3e29105592dcd0bc0fd11dc4f065be65596c1de074f8d87b`.

```bash
tiny-prl env add primeintellect/wordle

tiny-prl train smollm2-360m --env primeintellect/wordle --dry-run

tiny-prl train smollm2-360m --env primeintellect/wordle \
  --max-steps 1 --run.name wordle-first --guard-timeout 240
```

The setup command downloads the pinned Hub source, verifies its hash, and
prepares TextArena 0.7.4, NLTK 3.9.2 and the required corpora under
`outputs/environments/primeintellect-wordle-0.1.7/`. Optional packages are
installed with `uv pip --target --no-deps`, using the remaining dependencies of
the pinned stack. The global venv is unchanged. Setup requires network access;
TextArena can also check NLTK corpus freshness at startup. This is support for
this reviewed environment, not a generic installer for arbitrary Hub packages.

The Hub package uses the legacy `load_environment` API, which the pinned
Verifiers 0.4 taskset stack does not expose. `src/tiny_prl_wordle.py` adapts it
without substituting the built-in Wordle environment. It preserves:

- The Hub system prompt and `<guess>[word]</guess>` parser behavior.
- TextArena's five-letter, six-attempt game and feedback.
- Seeded train/eval word selection (2000/20 examples by default).
- Exact-answer reward, partial letter credit, shorter-solution bonus and the
  formatting reward with weight 0.2.

The parser reference is Verifiers 0.1.9.post3, the minimum version named by the
Hub package. Its formatting function gives even plain malformed text a raw
score of 0.2 (weighted contribution 0.04). The adapter deliberately retains
that behavior. Missing or malformed guesses are not rewritten into valid moves,
and task answers never enter model prompts.

`examples/tiny/wordle.toml` replaces alphabet-sort with this environment while
retaining the selected model preset. The bounded laptop trial uses at most six
model turns, 32 generated tokens/turn, 192 generated tokens/episode and a
480-token input cap within the 512-token inference context. Input/turn budgets
can stop an episode before six legal guesses. These are trial limits, not a
full-budget benchmark of model capability. The rollout timeout is 60 seconds;
training also retains the launcher resource guards. No hosted sandbox or GPU is
used. The game executes locally in the environment server.

## Verified trial

[Results](wordle-validation.json) record:

- 2,010 parser/reward fixtures match the original Hub functions and legacy XML
  parser exactly. Initial 360M chat-template prompt: 176 tokens.
- Four local vLLM evaluation episodes completed with no infrastructure errors.
  **Zero wins**, all games ended on invalid guesses. Weighted rewards were
  0.04, 0.08, 0.04 and 0.04; variation was formatting, not solving Wordle.
- One full-parameter 360M RL update and final checkpoint completed through
  `tiny-prl train ... --env primeintellect/wordle`. Finalized metrics report
  32 sampled rollouts, 28 discarded for no signal, and 4 admitted traces.
- Training gradient norm: **1.6565e-8**. Loss and mismatch KL are finite, but
  this near-zero gradient does not demonstrate useful learning. Serialized
  per-group rewards are tied at stored precision; do not interpret admission
  alone as proof of meaningful reward contrast.
- Training guard: GPU peak 7,170 MiB, zero job swap, no resource stop. Evaluation
  guard: GPU peak 1,833 MiB and zero job swap.

The integration is functional. Longer training should wait for evidence of
valid guesses and meaningful reward contrast; a small format-focused SFT warmup
or a more capable locally fitting policy can be tested separately. Changing
rewards or task difficulty would be a new environment variant.

Raw logs, traces, guards and the saved checkpoint from this trial remain in
`outputs/wordle-hub-20261103/`. No weights or downloaded third-party packages
are committed to Git.

## Format warmup and longer-run investigation

The [Wordle experiment record](../../research/wordle-20261103/README.md) contains
the 50-update format warmup, matched pass@1/pass@4 evaluation, the stopped
131-update RL run and its tied-reward admission diagnosis. Formatting improved,
but every evaluated checkpoint still scored zero wins. GRPO now centers shaped
rewards before its weighted baseline so tied rewards receive exactly zero
advantage. Hub rewards and dtype settings remain unchanged.
