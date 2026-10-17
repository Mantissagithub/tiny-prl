#!/usr/bin/env python3
"""Summarize completed tiny-rollout trials from guard and file-monitor artifacts."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

TRAINER_KEYS = (
    "time/step",
    "time/wait_for_batch",
    "time/forward_backward",
    "time/broadcast_weights",
    "perf/throughput",
    "perf/peak_memory",
    "loss/mean",
    "optim/grad_norm",
)
ORCH_KEYS = (
    "time/step",
    "time/wait_for_policy",
    "time/pack",
    "time/send_batch",
    "progress/input_tokens",
    "progress/output_tokens",
    "progress/tokens",
    "progress/rollouts",
    "rollout/attempts",
    "rollout/discarded",
    "rollout/stale",
    "rollout/errored",
    "rollout/cancelled",
    "concurrency/max_inflight",
)
ENGINE_PREFIX = "inference/server0.0/"


def read_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def read_metrics(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def metric_value(row: dict[str, Any], name: str) -> float | None:
    value = row.get(name)
    return float(value) if finite_number(value) else None


def maximum(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [value for row in rows if (value := metric_value(row, name)) is not None]
    return max(values) if values else None


def sum_step_metric(rows_by_step: dict[int, dict[str, Any]], steps: list[int], name: str) -> float | None:
    values = [metric_value(rows_by_step[step], name) for step in steps]
    return sum(values) if values and all(value is not None for value in values) else None


def counter_delta(rows: list[dict[str, Any]], name: str) -> tuple[float | None, bool]:
    values = [value for row in rows if (value := metric_value(row, name)) is not None]
    if len(values) < 2:
        return None, False
    reset = any(current < previous for previous, current in zip(values, values[1:]))
    return (None if reset else values[-1] - values[0]), reset


def token_window(rows: list[dict[str, Any]]) -> dict[str, Any]:
    points = [
        (float(row["time"]), value)
        for row in rows
        if finite_number(row.get("time"))
        and (value := metric_value(row, ENGINE_PREFIX + "generation_tokens_total")) is not None
    ]
    points.sort()
    increments = [
        (start_time, end_time, current - previous)
        for (start_time, previous), (end_time, current) in zip(points, points[1:])
        if current > previous
    ]
    if not increments:
        return {
            "generation_tokens_active_seconds_approx": None,
            "generation_tokens_per_second_approx": None,
            "generation_tokens_sampling_coarse": True,
        }
    duration = increments[-1][1] - increments[0][0]
    token_count = sum(delta for _, _, delta in increments)
    rate = token_count / duration if duration > 0 else None
    return {
        "generation_tokens_active_seconds_approx": duration,
        "generation_tokens_per_second_approx": rate,
        "generation_tokens_sampling_coarse": True,
    }


def summarize_inference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: row.get("time", 0))
    running_key = ENGINE_PREFIX + "num_requests_running"
    waiting_key = ENGINE_PREFIX + "num_requests_waiting"
    waiting_capacity_key = ENGINE_PREFIX + "num_requests_waiting_reason_capacity"
    kv_key = ENGINE_PREFIX + "kv_cache_usage_perc"
    cap_key = "concurrency/max_inflight"
    token_key = ENGINE_PREFIX + "generation_tokens_total"

    token_delta, token_reset = counter_delta(rows, token_key)
    preemption_delta, preemption_reset = counter_delta(rows, ENGINE_PREFIX + "num_preemptions_total")
    if preemption_delta is None:
        preemption_delta, preemption_reset = counter_delta(rows, ENGINE_PREFIX + "num_preemptions")
    histogram_sum_delta, histogram_sum_reset = counter_delta(rows, ENGINE_PREFIX + "request_num_preemptions_sum")
    histogram_count_delta, histogram_count_reset = counter_delta(rows, ENGINE_PREFIX + "request_num_preemptions_count")

    max_running = maximum(rows, running_key)
    max_cap = maximum(rows, cap_key)
    return {
        "max_running": max_running,
        "max_waiting": maximum(rows, waiting_key),
        "max_capacity_waiting": maximum(rows, waiting_capacity_key),
        "max_kv_cache_usage_perc": maximum(rows, kv_key),
        "max_controller_inflight": max_cap,
        "generation_tokens_delta": token_delta,
        "generation_tokens_counter_reset": token_reset,
        **token_window(rows),
        "preemptions_counter_delta": preemption_delta,
        "preemptions_counter_reset": preemption_reset,
        "request_num_preemptions_sum_delta": histogram_sum_delta,
        "request_num_preemptions_sum_reset": histogram_sum_reset,
        "request_num_preemptions_count_delta": histogram_count_delta,
        "request_num_preemptions_count_reset": histogram_count_reset,
        "preemption_histogram_sampling_coarse": True,
        "four_request_capacity_exercised": None
        if max_running is None or max_cap is None
        else max_running > 2 and max_cap >= 4,
    }


def summarize_trial(trial: dict[str, Any]) -> dict[str, Any] | None:
    guard_path = Path(trial["guard_dir"]) / "result.json"
    if not guard_path.is_file():
        return None

    guard = read_json(guard_path)
    expected = int(trial["expected_updates"])
    metrics_path = Path(trial["run_dir"]) / "monitors" / "file" / "metrics.jsonl"
    metric_rows = read_metrics(metrics_path) if metrics_path.is_file() else []

    by_producer_step: dict[str, dict[int, dict[str, Any]]] = {"trainer": {}, "orch": {}}
    orch_rows = []
    for row in metric_rows:
        producer = row.get("producer")
        step = row.get("step")
        if producer == "orch":
            orch_rows.append(row)
        if producer == "orch" and step is None:
            continue
        if producer not in by_producer_step or not isinstance(step, int):
            continue
        by_producer_step[producer].setdefault(step, {}).update(row)

    per_step = []
    trainer_steps: dict[int, dict[str, Any]] = by_producer_step["trainer"]
    orch_steps: dict[int, dict[str, Any]] = by_producer_step["orch"]
    for step in range(1, expected + 1):
        trainer = {key: trainer_steps.get(step, {})[key] for key in TRAINER_KEYS if key in trainer_steps.get(step, {})}
        orch = {key: orch_steps.get(step, {})[key] for key in ORCH_KEYS if key in orch_steps.get(step, {})}
        per_step.append({"step": step, "trainer": trainer, "orch": orch})

    completed = [step for step in range(1, expected + 1) if "time/step" in trainer_steps.get(step, {})]
    finite_rows = []
    for step in completed:
        row = trainer_steps[step]
        finite_rows.append(finite_number(row.get("loss/mean")) and finite_number(row.get("optim/grad_norm")))

    orch_steps_done = [step for step in range(1, expected + 1) if "time/step" in orch_steps.get(step, {})]

    return {
        "arm": trial["arm"],
        "source": trial["source"],
        "expected_updates": expected,
        **guard,
        "completed_updates": len(completed),
        "finite_loss_grad": all(finite_rows) if finite_rows else None,
        "finite_loss_grad_updates": sum(finite_rows) if finite_rows else None,
        "trainer_seconds": sum_step_metric(trainer_steps, completed, "time/step"),
        "trainer_wait_seconds": sum_step_metric(trainer_steps, completed, "time/wait_for_batch"),
        "forward_backward_seconds": sum_step_metric(trainer_steps, completed, "time/forward_backward"),
        "orch_seconds": sum_step_metric(orch_steps, orch_steps_done, "time/step"),
        "steps": per_step,
        "inference": summarize_inference(orch_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    spec = read_json(args.spec)
    records = [record for trial in spec["trials"] if (record := summarize_trial(trial)) is not None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
