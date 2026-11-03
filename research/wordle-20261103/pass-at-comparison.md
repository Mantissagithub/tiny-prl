# Wordle baseline and checkpoint evaluation

20 eval-split tasks, four different seeds per word, 80 game attempts per checkpoint.
Same Hub prompt and rewards, temperature 1, six-turn limit, 32 tokens per turn, 480 input tokens.

| Checkpoint | pass@1 | pass@4 | Wins / games | Words solved / 20 | Formatted first guess | Mean reward |
|---|---:|---:|---:|---:|---:|---:|
| Original 360M | 0.0% | 0.0% | 0/80 | 0/20 | 0/80 | 0.04000 |
| SFT 50 steps | 0.0% | 0.0% | 0/80 | 0/20 | 80/80 | 0.41375 |
| SFT 50 + RL 2 steps | 0.0% | 0.0% | 0/80 | 0/20 | 80/80 | 0.41250 |

pass@1 estimates the single-attempt win probability using all four samples per word. pass@4 counts a word as solved if any of its four attempts wins.
All 240 episodes are retained, including invalid moves and generation or episode truncations. No evaluation errors or timeouts were accepted.
These measurements establish the baseline and current checkpoint scores. Twenty words are a small evaluation; these scores do not predict eventual convergence.

The original Hub task sampler uses replacement. RL rollouts exposed these eval answer words: ['stamp']. SFT source games with any eval answer were excluded before training.
The strict subset excludes those answer words from every checkpoint comparison.
- Original 360M: 19 unseen-answer words, 76 attempts, pass@1 0.0%, pass@4 0.0%.
- SFT 50 steps: 19 unseen-answer words, 76 attempts, pass@1 0.0%, pass@4 0.0%.
- SFT 50 + RL 2 steps: 19 unseen-answer words, 76 attempts, pass@1 0.0%, pass@4 0.0%.
