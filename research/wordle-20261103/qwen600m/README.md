# Qwen3-600M Wordle baseline

I tested the cached Qwen3-0.6B checkpoint without additional training on the same 20 evaluation
tasks and seeds 42–45 used for 360M. The Hub prompt and rewards were unchanged.
Thinking was disabled, with stock vLLM, temperature 1, 32 tokens per turn,
six turns and the same 480-token input budget. No SFT or RL updates were run.

| Metric | Result |
|---|---:|
| Games | 80 |
| Wins | 0 |
| pass@1 / pass@4 | 0% / 0% |
| Opening moves followed by game feedback | 60/80 |
| Inference errors / timeouts / token-length truncations | 0 / 0 / 0 |
| GPU peak | 2,057 MiB |
| Minimum available host RAM | 6.77 GiB |
| Job swap | 0 |

Opening replies were `apple` in 40 games, `guess` in 20, and malformed in 20.
All games ended in invalid moves, often repeated guesses. TextArena's original
action-formatting wrapper supplies missing brackets, so legal moves and strict
XML-plus-bracket formatting are different metrics. This baseline does not
establish that Qwen cannot learn Wordle or how thinking-enabled play performs.

Full results: [pass-at.json](pass-at.json), [diagnostics.json](diagnostics.json)
and [guard-result.json](guard-result.json).

## Thinking comparison

I repeated the same tasks and seeds with thinking on and off, both at 256 tokens
per turn and a 2,048-token context. All other game and sampling settings matched.

| Mode | Games | pass@1 / pass@4 | Truncated calls | Generated tokens |
|---|---:|---:|---:|---:|
| Thinking off | 80 | 0% / 0% | 0/220 | 1,862 |
| Thinking on | 80 | 0% / 0% | 145/206 | 48,837 |

Every thinking-enabled opening reply contained reasoning without a final answer.
The control had 60 accepted opening moves. Both runs completed without evaluation
errors, timeouts or job swap; sampled GPU peak was 2,047 MiB in each run.
Thinking did not help within these limits. These results do not establish how
the model would perform with a larger reasoning allowance. No training was run.

[Matched results](thinking-comparison.json). The archived `evaluate.py` accepts
`pass4 qwen roomy thinking` after the run tag and model path; omit `thinking`
for the matched control. `qwen_thinking_report.py` checks the recorded task keys,
seeds and budgets before producing this comparison. The scorer treats assistant
messages containing only reasoning as empty final answers, preserving those
failed attempts in the denominator.

## Larger reasoning budgets

Four-game pilots used eval tasks 2000–2003, one attempt each, seed 42,
temperature 1 and unchanged Hub instructions and rewards. These are not pass@4
estimates or matched samples of the full 20-word evaluation.

| Tokens/turn | Context | Completed opening answers | Accepted opening moves | Wins |
|---|---:|---:|---:|---:|
| 1,024 | 4,096 | 0/4 | 0/4 | 0/4 |
| 2,048 | 8,192 | 0/4 | 0/4 | 0/4 |
| 4,096 | 8,192 | 4/4 | 3/4 | 0/4 |

At 4,096 tokens no calls were token-length truncated, but the model still
repeated guesses, generated nonwords or malformed answers, or reached the turn
limit. More budget let it finish reasoning without establishing successful play.
All pilots completed without infrastructure errors, timeouts or job swap;
sampled GPU peak stayed below 2.4 GiB. No training was run.

[Pilot results and budgets](large-budget-pilots.json). Pass `pilot qwen large thinking`
after the run tag and model path; replace `large` with `xlarge` or `xxlarge`
for the other profiles. `qwen_reasoning_pilot.py`
scores completed pilots separately from the full pass@1/pass@4 scorer.
