# Fixed-work fork-server microbenchmark

From `prime-rl/`, run:

```sh
CUDA_VISIBLE_DEVICES="" uv run --offline --no-project --python "$HOME/.venv/bin/python" python outputs/rollout-overhead-20261101/forkserver-fixed-work-r1/bench.py
```

This drives the real `launch_chat_program` and `SubprocessRuntime` path for eight unique frozen alphabet-sort prompts at concurrency two. It compares twelve alternating placements (`off, on, on, off, on, off` repeated), giving six blocks per arm. A first invocation of each arm is recorded separately as setup and excluded from timed blocks. Each block records wall time, launch-to-first-mock-request, per-episode duration, time after the mock response, every actual OpenAI-compatible request body and deterministic mock answer, and each episode's exit code/stdout/stderr. It checks prompt uniqueness, exact request/answer and episode-output equivalence, and fails on any nonzero exit.

The local mock returns deterministic SSE responses immediately. This isolates subprocess/runtime overhead; it does not measure live policy generation, verifier reward computation, GPU work, optimizer work, or end-to-end RL throughput. Setup includes first-call effects (including fork-server startup for `on`); timed blocks reuse the warm server. The report is written to `results.json` beside the script and printed as JSON. Existing reports are never overwritten. The `finally` cleanup stops fork servers and the local HTTP server.
