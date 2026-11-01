#!/usr/bin/env python3
"""Rebuild a bounded offline comparison of the two terminal full-RL runs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "r30-full-rl-r1"
BATCH = "full360-admission-20261101-a2"
ARMS = {
    "stock": ("live-full360-admission-20261101-a2-stock", "r30-full-rl-stock-2update"),
    "r30": ("live-full360-admission-20261101-a2-r30", "r30-full-rl-fast-r30-2update"),
}
CONFIGS = ("trainer.json", "orchestrator.json", "inference.json")


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metrics_rows(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def trace_index_stats(path: Path, annotation_paths: list[Path]):
    total = 0
    by_kind = {}
    rewards = []
    trace_ids = set()
    errored = 0
    for row in metrics_rows(path):
        total += 1
        by_kind[row.get("kind", "<missing>")] = by_kind.get(row.get("kind", "<missing>"), 0) + 1
        reward = row.get("reward")
        if isinstance(reward, (int, float)):
            rewards.append(float(reward))
        errored += int(row.get("num_errors", 0) or 0)
        for trace_id in row.get("trace_ids", []):
            trace_ids.add(trace_id)
    annotation_ids = set()
    effective_by_step = {}
    for p in annotation_paths:
        for row in metrics_rows(p):
            annotation_ids.add(row["trace_id"])
            info = row.get("info", {})
            ship = info.get("ship", {})
            step = ship.get("step")
            if step is not None and info.get("effective"):
                effective_by_step[str(step)] = effective_by_step.get(str(step), 0) + 1
    matched = trace_ids & annotation_ids
    return {
        "records_in_stream_index_including_terminal_spill": total,
        "records_by_kind": by_kind,
        "reward": {
            "n": len(rewards),
            "mean": sum(rewards) / len(rewards) if rewards else None,
            "min": min(rewards) if rewards else None,
            "max": max(rewards) if rewards else None,
            "nonzero_n": sum(value != 0 for value in rewards),
        },
        "num_record_errors": errored,
        "unique_trace_ids_referenced_by_stream_index": len(trace_ids),
        "unique_annotation_trace_ids": len(annotation_ids),
        "stream_annotation_trace_id_intersection": len(matched),
        "stream_annotation_trace_id_unmatched_stream": len(trace_ids - annotation_ids),
        "stream_annotation_trace_id_unmatched_annotations": len(annotation_ids - trace_ids),
        "effective_annotations_by_step": effective_by_step,
    }


def changes(a, b, pointer=""):
    out = []
    if type(a) is not type(b):
        return [(pointer or "/", a, b)]
    if isinstance(a, dict):
        for key in sorted(a.keys() | b.keys()):
            child = f"{pointer}/{key}"
            if key not in a or key not in b:
                out.append((child, a.get(key, "<absent>"), b.get(key, "<absent>")))
            else:
                out.extend(changes(a[key], b[key], child))
    elif isinstance(a, list):
        if a != b:
            out.append((pointer or "/", a, b))
    elif a != b:
        out.append((pointer or "/", a, b))
    return out


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def analyze_arm(arm: str):
    live_dir, run_name = ARMS[arm]
    run = BASE / "runs" / live_dir / run_name
    meta = BASE / "results" / f"rl-case-{BATCH}-{arm}"
    guard = read_json(BASE / "results" / f"rl-guard-{BATCH}-{arm}" / "result.json")
    completion = read_json(meta / "completion.json")
    launch = read_json(meta / "launch.json")
    rows = metrics_rows(run / "monitors/file/metrics.jsonl")
    steps = []
    for number in (1, 2):
        progress = [r for r in rows if r.get("step") == number and "progress/rollouts" in r]
        update = [r for r in rows if r.get("step") == number and "optim/grad_norm" in r]
        loss = [r for r in rows if r.get("step") == number and "loss/mean" in r]
        timing = [r for r in rows if r.get("step") == number and r.get("producer") == "trainer" and "time/step" in r]
        if not all((len(progress) == len(update) == len(loss) == len(timing) == 1,)):
            raise ValueError(f"{arm} step {number} metric row cardinality is not exactly one per field")
        p, u, l, t = progress[0], update[0], loss[0], timing[0]
        steps.append({
            "step": number,
            "window_rollouts": int(p["progress/rollouts"]),
            "window_tasks": int(p["progress/tasks"]),
            "window_tokens": int(p["progress/tokens"]),
            "window_no_signal": int(p.get("rollout/no_signal", 0)),
            "window_discarded": int(p.get("rollout/discarded", 0)),
            "window_errored": int(p.get("rollout/errored", 0)),
            "queued_trace_samples": int(p.get("rollout/traces/queued", 0)),
            "effective_reward_mean": p.get("train/agg/effective/agent/reward/mean"),
            "effective_reward_min": p.get("train/agg/effective/agent/reward/min"),
            "effective_reward_max": p.get("train/agg/effective/agent/reward/max"),
            "effective_admitted_mean": p.get("train/agg/effective/agent/is_admitted/mean"),
            "loss_mean": l.get("loss/mean"),
            "gradient_norm": u.get("optim/grad_norm"),
            "mismatch_kl_all_mean": l.get("mismatch_kl/all/mean"),
            "time_step_seconds": t.get("time/step"),
            "time_wait_for_batch_seconds": t.get("time/wait_for_batch"),
            "time_forward_backward_seconds": t.get("time/forward_backward"),
            "time_broadcast_weights_seconds": t.get("time/broadcast_weights"),
            "time_save_checkpoint_seconds": t.get("time/save_ckpt"),
        })
    total = {key: sum(step[key] for step in steps) for key in
             ("window_rollouts", "window_tasks", "window_tokens", "window_no_signal", "window_discarded", "window_errored", "queued_trace_samples")}
    resolved_dir = run / "configs/attempt_1/resolved"
    resolved = {name: read_json(resolved_dir / name) for name in CONFIGS}
    live_hashes = {name: sha256(resolved_dir / name) for name in CONFIGS}
    model = resolved["trainer.json"]["model"]
    training_shape = {
        "model_name": model["name"],
        "lora": model["lora"],
        "optimization_dtype": model["optimization_dtype"],
        "reduce_dtype": model["reduce_dtype"],
        "checkpoint_config": resolved["trainer.json"].get("ckpt"),
        "resume_config": resolved["trainer.json"].get("resume"),
        "max_steps": resolved["trainer.json"].get("max_steps"),
    }
    audit_ref = launch["dry_resolved"]["resolved"]
    trace_root = run / "monitors/file/traces"
    trace = trace_index_stats(
        trace_root / "stream.index.jsonl",
        [trace_root / "annotations/trainer.index.jsonl", trace_root / "annotations/orch.index.jsonl"],
    )
    hashes = {
        "metrics_jsonl": sha256(run / "monitors/file/metrics.jsonl"),
        "rl_toml": sha256(run / "configs/attempt_1/rl.toml"),
        "completion_json": sha256(meta / "completion.json"),
        "launch_json": sha256(meta / "launch.json"),
        "guard_result_json": sha256(BASE / "results" / f"rl-guard-{BATCH}-{arm}" / "result.json"),
        **{f"live_resolved_{name}": value for name, value in live_hashes.items()},
        "run_rl_case_py": sha256(BASE / "probe/run_rl_case.py"),
    }
    return {
        "arm": arm,
        "run_name": run_name,
        "returncode": completion.get("returncode"),
        "rl_child_returncode": completion.get("rl_child_returncode"),
        "elapsed_seconds": completion.get("rl_elapsed_seconds"),
        "elapsed_scope": completion.get("elapsed_scope"),
        "steps": steps,
        "two_update_totals": total,
        "admission_rate": 1.0 if all(s["effective_admitted_mean"] == 1.0 for s in steps) else None,
        "metrics_request_success_total": next((r.get("inference/server0.0/request_success_total") for r in reversed(rows) if r.get("step") is None and "inference/server0.0/request_success_total" in r), None),
        "trace_index": trace,
        "guard": {key: guard.get(key) for key in ("returncode", "stop_reason", "elapsed_seconds", "minimum_available_bytes", "peak_cgroup_bytes", "peak_swap_bytes", "peak_gpu_mib", "minimum_disk_free_bytes")},
        "configuration": {
            "dry_resolved_report_sha256": launch["dry_resolved"]["report_sha256"],
            "same_passing_report": launch["dry_resolved"]["report_sha256"] == read_json(BASE / "results" / f"rl-case-{BATCH}-stock/launch.json")["dry_resolved"]["report_sha256"],
            "live_resolved_files_present": all((resolved_dir / name).is_file() for name in CONFIGS),
            "passing_report_resolved_paths": {name: audit_ref[name]["path"] for name in CONFIGS},
            "live_resolved_paths": {name: str(resolved_dir / name) for name in CONFIGS},
            "runtime_config_check": launch.get("runtime_config_check"),
            "live_resolved_hashes": live_hashes,
            "live_resolved": resolved,
            "training_shape": training_shape,
        },
        "route_validation": completion.get("route_validation"),
        "input_sha256": hashes,
    }


def main():
    arms = {arm: analyze_arm(arm) for arm in ARMS}
    left, right = arms["stock"]["configuration"]["live_resolved"], arms["r30"]["configuration"]["live_resolved"]
    differences = []
    for name in CONFIGS:
        for pointer, a, b in changes(left[name], right[name]):
            differences.append({"file": name, "pointer": pointer, "stock": a, "r30": b})
    if arms["stock"]["configuration"]["training_shape"] != arms["r30"]["configuration"]["training_shape"]:
        raise ValueError("full-RL optimizer/model settings differ")
    shape = arms["stock"]["configuration"]["training_shape"]
    if shape["lora"] is not None or shape["optimization_dtype"] != "float32" or shape["reduce_dtype"] != "float32" or shape["checkpoint_config"] is not None or shape["resume_config"] is not None:
        raise ValueError("run does not match full-parameter FP32 optimizer/reduce without checkpointing")
    for arm in arms.values():
        del arm["configuration"]["live_resolved"]
    script_hash = sha256(Path(__file__))
    payload = {
        "schema": "tiny-prl.full-rl-pair-audit.v1",
        "claim": {
            "classification": "inconclusive as a speedup claim",
            "reason": "one terminal pair with unequal rollout/task/token work; no repeated alternating runs; no limiter profile",
            "observed_elapsed_change_percent": (arms["stock"]["elapsed_seconds"] - arms["r30"]["elapsed_seconds"]) / arms["stock"]["elapsed_seconds"] * 100,
            "work_normalized_seconds_per_1000_tokens": {
                arm: arms[arm]["elapsed_seconds"] / arms[arm]["two_update_totals"]["window_tokens"] * 1000 for arm in ARMS
            },
            "normalization_warning": "descriptive only; token composition and model decode configuration differ",
            "numerics_claim": "not established; no broad acceptance claim because HF floors remain unresolved",
        },
        "configuration_diff": differences,
        "configuration_diff_classification": "run/output identity and decode/R30 preload environment only" if all(
            item["pointer"] in {"/output_dir", "/decode_megakernel", "/env_vars/PRIME_DECODE_MEGAKERNEL_360M", "/env_vars/PRIME_R30_PRELOAD_TRACE_DIR", "/env_vars/PRIME_R30_PRIVATE_PRELOAD"}
            for item in differences
        ) else "unexpected differences require review",
        "arms": arms,
        "source_and_harness": {
            "audit_script_sha256": script_hash,
            "matched_r30_source_sha256": arms["r30"]["route_validation"]["matched_source_sha256"],
            "matched_r30_dso_sha256": arms["r30"]["route_validation"]["matched_dso_sha256"],
        },
        "scope": "CPU-only metadata/metrics/config audit; no checkpoints, model weights, or compressed trace payloads were opened",
    }
    if not all(arms[a]["configuration"]["same_passing_report"] and arms[a]["configuration"]["live_resolved_files_present"] for a in ARMS):
        raise ValueError("passing configuration report or live resolved files are missing")
    if any(not finite(s["gradient_norm"]) or not finite(s["loss_mean"]) or not finite(s["mismatch_kl_all_mean"]) for a in ARMS for s in arms[a]["steps"]):
        raise ValueError("nonfinite training metric")
    out = HERE / "result.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(out)
    print(json.dumps({"claim": payload["claim"], "totals": {a: arms[a]["two_update_totals"] for a in ARMS}}, indent=2))


if __name__ == "__main__":
    main()
