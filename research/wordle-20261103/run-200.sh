#!/usr/bin/env bash
set -euo pipefail
cd /home/pradheep/tiny-prl/prime-rl
wordle_work="$PWD/outputs/wordle-learning-20261103"

bin/tiny-prl train smollm2-360m \
  --env primeintellect/wordle \
  @ "$wordle_work/wordle-answer-disjoint.json" \
  --model.name "$wordle_work/model-step50" \
  --max-steps 200 \
  --orchestrator.concurrency.max-inflight 2 \
  --ckpt.interval 50 \
  --ckpt.keep-last 1 \
  --run.name wordle-200-r1 \
  --output-dir "$wordle_work/rl" \
  --guard-timeout 7200

test -f "$wordle_work/rl/wordle-200-r1/checkpoints/step_200/trainer/.metadata"

bash outputs/megakernel-20261025/runtime/run_gpu.sh \
  tiny-wordle-export-200 \
  "$wordle_work/guard-export-200" \
  "$wordle_work/export.py" \
  "$wordle_work/rl/wordle-200-r1/checkpoints/step_200" \
  "$wordle_work/model-rl-step200"

bash outputs/megakernel-20261025/runtime/run_micro.sh \
  tiny-wordle-eval-200 \
  "$wordle_work" pass4-rl200-r1 \
  "$wordle_work/evaluate.py" \
  pass4-rl200-r1 "$wordle_work/model-rl-step200" pass4

uv run --offline --no-project --python /usr/bin/python3 \
  python "$wordle_work/pass_at.py" pass4-rl200-r1

echo "Results: $wordle_work/pass4-rl200-r1/pass-at.json"
