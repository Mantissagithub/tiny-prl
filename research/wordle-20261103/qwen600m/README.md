# Qwen3-600M Wordle baseline

I tested the cached, untrained Qwen3-0.6B checkpoint on the same 20 evaluation
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
