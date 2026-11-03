# Wordle on my laptop

This is still in progress. I have verified the training pipeline, but I have not demonstrated improved Wordle solving.

I used the cached SmolLM2-360M-Instruct model with the original `primeintellect/wordle@0.1.7` prompt, parser and rewards. The format warmup uses concise legal guesses extracted from `willcb/V3-wordle` at revision `b853a767e6e13ac7db769b72d7f8fdb3acf83278`. I excluded source games whose answers match the 20 evaluation words. The resulting 3,067 training rows fit within 512 tokens. All rows passed exact chat-template and final-assistant loss-mask checks.

Most of this work was built with Codex coding agents. These records describe measured behavior on my RTX 4060 laptop, not general training guarantees.

## Measurements

I evaluated 20 taskset evaluation tasks with four different seeds per word, temperature 1, six turns, 32 tokens per turn and a 480-token input budget. A pass means winning the game. pass@1 is the average single-attempt success estimate across the four samples. pass@4 is the fraction of words solved by at least one of those samples.

| Checkpoint | Games | pass@1 | pass@4 | Formatted first guesses |
|---|---:|---:|---:|---:|
| Original 360M | 80 | 0% | 0% | 0/80 |
| 50 SFT updates | 80 | 0% | 0% | 80/80 |
| SFT plus 2 RL updates | 80 | 0% | 0% | 80/80 |

The initial two-update RL run completed, saved a checkpoint and produced gradient norms 0.175 and 0.105. That establishes functional updates, not task improvement. The task sampler uses replacement; one evaluation answer occurred in those RL rollouts. The matched 19-word unseen-answer subset also scored zero for all three checkpoints. Details are in [pass-at-comparison.md](pass-at-comparison.md).

The longer run was stopped at update 131. Its 1,763 recorded episodes contained zero wins. After update 100, all 561 recorded episodes ended in invalid moves, and 275 of 280 completed rollout pairs tied in reward. The model commonly opened with `[crane]` and then produced invalid guesses such as `[slatey']`. Existing Hub rewards retain earlier partial credit after a later invalid move. I have preserved those rewards rather than silently changing the environment.

## Tied-reward fix

The FP32 length-weighted baseline could give equal rewards tiny nonzero advantages. With rewards 0.4 and lengths 33 and 34, the old code produced `2.9802322387695312e-08` for both episodes. The exact-nonzero admission check then accepted them as trainable. In the stopped run, 360 of 524 admitted episodes had advantage magnitude below `1e-6`.

GRPO now subtracts one shaped reward before calculating its weighted baseline. This is algebraically the same estimator, but tied shaped rewards become exact zeros. It keeps FP32 optimization and reduction, the Hub rewards, length weighting and the existing admission rule. Genuine unequal rewards and length-penalty differences still produce signed advantages. Tests cover the original failure, small reward differences, length penalties and empty groups.

This fix removes spurious training signal. It does not repair invalid gameplay, create reward contrast or establish that longer training will converge. Before another long run, I need stronger valid multi-turn behavior and a short RL run with sustained real reward contrast. The step-100 policy is unverified and should not replace the SFT baseline as an accepted policy.

## Local scripts and artifacts

This directory preserves source and small result files. It excludes weights, the venv, datasets, downloaded environment packages and bulk logs. The scripts target my existing runtime directory `/home/pradheep/tiny-prl/prime-rl/outputs/wordle-learning-20261103`; they are laptop experiment snapshots, not a standalone downloadable training setup.

- `prepare.py`, `source.json` and `data-manifest.json` record dataset preparation and provenance.
- `smol_renderer.py` and `prove_renderer.py` provide the explicit SFT renderer and parity check.
- `sft.toml`, `train.py`, `export.py` and `rl.py` preserve the initial training/export commands.
- `evaluate.py`, `pass_at.py` and `compare_pass_at.py` preserve the matched four-seed evaluation.
- `diagnose_stall.py` and `stall-diagnosis.json` record the stopped run investigation.
- `run-200.sh` preserves the attempted long-run recipe. It starts from SFT, excludes all 20 eval answers from RL training, saves every 50 steps while retaining the latest checkpoint, and evaluates afterward. It is not evidence that a 200-step run completed.

The first SFT attempt hit the RAM floor during checkpoint saving. Per-tensor checkpoint writes fixed that resource failure. A subsequent RL attempt also hit the host RAM floor; the later recipe reduced concurrent rollouts to two. Resource guards remain enabled. All successful checkpoints are local.
