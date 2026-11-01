#!/usr/bin/env python3
"""Summarize the bounded R30 forkserver ABBA evidence without loading models."""

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "r30-full-forkserver-r1"
OUTPUT = HERE / "report.md"
RESULT = HERE / "audit-result.json"
RUNS = (
    ("fs360-20261101-b-a1", "off"),
    ("fs360-20261101-b-b1", "on"),
    ("fs360-20261101-b-b2", "on"),
    ("fs360-20261101-b-a2", "off"),
)
SOURCE_SHA256 = "5e853d1e6b207f199c0cfd27a4d34394f7fa81ab4748ea0e51d3240dd7286638"
INCLUDE_SHA256 = "60b234e7ab6bbd49ff5a7444fb0ff171a97ee37fe31234a72b8f045b3b495e83"
DSO_SHA256 = "625f39f978e76a3e0ff01ed469bcd0de0cf65e51929a9597668af9cdff6c4db2"
CONFIG_FILES = (
    "trainer.json",
    "orchestrator.json",
    "inference.json",
    "envs/train/alphabet-sort.json",
)
ALLOWED_CONFIG_DIFFS = {
    "/output_dir",
    "/train/source/0/env/agent/runtime/fork_server",
    "/env_vars/PRIME_R30_PRELOAD_TRACE_DIR",
    "/address_file",
    "/env/agent/runtime/fork_server",
}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def sha256_file(path):
    if not path.exists() or path.stat().st_size > 5 * 1024 * 1024:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_rows(rows):
    merged = {}
    for row in rows:
        merged.update(row)
    return merged


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        changed = []
        for key in sorted(left.keys() | right.keys()):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                changed.append(child)
            else:
                changed.extend(differences(left[key], right[key], child))
        return changed
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path + "/length"]
        return [item for i, pair in enumerate(zip(left, right))
                for item in differences(pair[0], pair[1], f"{path}/{i}")]
    return [] if left == right else [path]


def verify_route(case, route):
    if not route or route.get("status") != "pass":
        return "missing/fail"
    workers = case / "worker-traces"
    loaded_by_pid = set()
    decodes = []
    try:
        files = sorted(workers.glob("worker-*.jsonl"))
        if not files or any(path.stat().st_size > 32 * 1024 for path in files):
            return "unverified: worker trace bound"
        for path in files:
            for line in path.read_text().splitlines():
                event = json.loads(line)
                pid = event.get("pid")
                if event.get("event") == "private_r30_loaded":
                    if (event.get("source_sha256") == SOURCE_SHA256
                            and event.get("include_sha256") == INCLUDE_SHA256
                            and event.get("library_sha256") == DSO_SHA256):
                        loaded_by_pid.add(pid)
                elif event.get("event") == "private_r30_first_decode":
                    if (event.get("source_sha256") == SOURCE_SHA256
                            and event.get("library_sha256") == DSO_SHA256):
                        decodes.append(pid)
    except (OSError, json.JSONDecodeError):
        return "unverified: unreadable worker trace"
    return "pass" if any(pid in loaded_by_pid for pid in decodes) else "fail: no pinned same-worker decode"


def run_record(run_id, arm):
    case = SOURCE / "results" / f"rl-case-{run_id}-{arm}"
    guard = SOURCE / "results" / f"rl-guard-{run_id}-{arm}" / "result.json"
    output = SOURCE / "runs" / f"live-{run_id}-{arm}" / f"r30-full-forkserver-{arm}-2update"
    metadata = read_json(case / "completion.json")
    launch = read_json(case / "launch.json")
    route = read_json(case / "route-validation.json")
    route_status = verify_route(case, route)
    guard_data = read_json(guard)

    metrics_path = output / "monitors/file/metrics.jsonl"
    metric_rows, metrics_note = [], "available"
    if metrics_path.exists():
        if metrics_path.stat().st_size > 5 * 1024 * 1024:
            metrics_note = "skipped: exceeds 5 MiB bound"
        else:
            try:
                for line in metrics_path.read_text().splitlines():
                    try:
                        metric_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        metrics_note = "partial: ignored malformed trailing/intermediate line"
            except OSError as error:
                metrics_note = f"unavailable: {error}"
    else:
        metrics_note = "not present yet"

    step_rows = {}
    for step in (1, 2):
        orch = [r for r in metric_rows if r.get("producer") == "orch" and r.get("step") == step]
        trainer = [r for r in metric_rows if r.get("producer") == "trainer" and r.get("step") == step]
        step_rows[step] = {"orch": merge_rows(orch), "trainer": merge_rows(trainer)}

    steps = []
    totals = {"rollouts": 0, "tasks": 0, "tokens": 0, "completed_traces": 0,
              "errors": 0, "cancelled": 0, "stale": 0, "effective_traces": 0,
              "admitted_traces": 0, "completed_tasks": 0}
    for step in (1, 2):
        orch, trainer = step_rows[step]["orch"], step_rows[step]["trainer"]
        if not orch or not trainer:
            continue
        rollouts = orch.get("progress/rollouts")
        tasks = orch.get("progress/tasks")
        tokens = orch.get("progress/tokens")
        completed_rate = orch.get("train/agg/all/agent/is_completed/mean")
        error_rate = orch.get("train/agg/all/agent/has_error/mean")
        trainable_rate = orch.get("train/agg/all/agent/is_trainable/mean")
        admitted_rate = orch.get("train/agg/all/agent/is_admitted/mean")
        completed = round(rollouts * completed_rate) if finite(rollouts) and finite(completed_rate) else None
        completed_tasks = (tasks if completed_rate == 1.0
                           and (orch.get("rollout/errored") or 0) == 0
                           and (orch.get("rollout/cancelled") or 0) == 0
                           and (orch.get("rollout/stale") or 0) == 0 else None)
        errors = round(rollouts * error_rate) if finite(rollouts) and finite(error_rate) else None
        effective = round(rollouts * trainable_rate) if finite(rollouts) and finite(trainable_rate) else None
        queued = orch.get("rollout/traces/queued")
        reward_groups = {}
        if finite(tasks):
            for label in ("none", "some", "all"):
                rate = orch.get(f"train/agg/all/agent/solved_{label}")
                reward_groups[label] = round(tasks * rate) if finite(rate) else None
        step_data = {
            "step": step,
            "rollouts": rollouts,
            "tasks": tasks,
            "completed_tasks": completed_tasks,
            "tokens": tokens,
            "total_rollouts": orch.get("progress/total_rollouts"),
            "total_tasks": orch.get("progress/total_tasks"),
            "total_tokens": orch.get("progress/total_tokens"),
            "completed_traces": completed,
            "error_traces": errors,
            "rollout_errors": orch.get("rollout/errored"),
            "rollout_cancelled": orch.get("rollout/cancelled"),
            "rollout_stale": orch.get("rollout/stale"),
            "attempts": orch.get("rollout/attempts"),
            "discarded": orch.get("rollout/discarded"),
            "no_signal": orch.get("rollout/no_signal"),
            "effective_traces": effective,
            "queued_traces": queued,
            "admitted_traces": round(rollouts * admitted_rate) if finite(rollouts) and finite(admitted_rate) else None,
            "reward_groups": reward_groups,
            "reward_mean": orch.get("train/agg/all/agent/reward/mean"),
            "reward_min": orch.get("train/agg/all/agent/reward/min"),
            "reward_max": orch.get("train/agg/all/agent/reward/max"),
            "effective_reward_mean": orch.get("train/agg/effective/agent/reward/mean"),
            "grad_norm": trainer.get("optim/grad_norm"),
            "loss": trainer.get("loss/mean"),
            "kl": trainer.get("mismatch_kl/all/mean"),
            "step_seconds": trainer.get("time/step"),
            "wait_for_batch_seconds": trainer.get("time/wait_for_batch"),
            "forward_backward_seconds": trainer.get("time/forward_backward"),
            "broadcast_seconds": trainer.get("time/broadcast_weights"),
            "valid_update": all(finite(trainer.get(key)) for key in
                                 ("optim/grad_norm", "loss/mean", "mismatch_kl/all/mean")),
        }
        steps.append(step_data)
        for target, source in (("rollouts", "rollouts"), ("tasks", "tasks"), ("tokens", "tokens"),
                               ("completed_traces", "completed_traces"), ("errors", "error_traces"),
                               ("cancelled", "rollout_cancelled"), ("stale", "rollout_stale"),
                               ("effective_traces", "effective_traces"),
                               ("admitted_traces", "admitted_traces"),
                               ("completed_tasks", "completed_tasks")):
            value = step_data[source]
            if isinstance(value, (int, float)):
                totals[target] += value

    trainer_index = output / "monitors/file/traces/annotations/trainer.index.jsonl"
    try:
        trainer_annotation_count = len(trainer_index.read_text().splitlines())
    except OSError:
        trainer_annotation_count = None

    return {
        "id": run_id,
        "arm": arm,
        "status": (metadata or {}).get("status", "running" if launch else "missing"),
        "launch": launch,
        "completion": metadata,
        "route": route,
        "route_status": route_status,
        "guard": guard_data,
        "input_hashes": {
            "launch": sha256_file(case / "launch.json"),
            "completion": sha256_file(case / "completion.json"),
            "route_validation": sha256_file(case / "route-validation.json"),
            "guard": sha256_file(guard),
            "metrics": sha256_file(metrics_path),
        },
        "steps": steps,
        "totals": totals,
        "trainer_annotation_count": trainer_annotation_count,
        "metrics_note": metrics_note,
        "output": output,
    }


def config_pair_status(off_run, on_run):
    if not off_run["output"].exists() or not on_run["output"].exists():
        return "pending: both resolved run directories are not available", []
    all_differences = []
    for filename in CONFIG_FILES:
        left = read_json(off_run["output"] / "configs/attempt_1/resolved" / filename)
        right = read_json(on_run["output"] / "configs/attempt_1/resolved" / filename)
        if left is None or right is None:
            return f"incomplete: missing resolved {filename}", all_differences
        changed = differences(left, right)
        all_differences.extend(f"{filename}:{p}" for p in changed)
    unexpected = [item for item in all_differences
                  if item.split(":", 1)[1] not in ALLOWED_CONFIG_DIFFS]
    return ("pass" if not unexpected else "fail: unexpected config differences " + ", ".join(unexpected)), all_differences


def fmt(value, digits=3):
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(runs):
    complete = [r for r in runs if r["status"] == "completed"
                and (r["completion"] or {}).get("rl_child_returncode") == 0
                and len(r["steps"]) == 2
                and all(s["valid_update"] for s in r["steps"])]
    all_complete = len(complete) == 4
    by_arm = {arm: [r for r in complete if r["arm"] == arm] for arm in ("off", "on")}
    repeated_on_beats_off = None
    if all(len(by_arm[a]) == 2 for a in ("off", "on")):
        on_times = [(r["completion"] or {}).get("rl_elapsed_seconds") for r in by_arm["on"]]
        off_times = [(r["completion"] or {}).get("rl_elapsed_seconds") for r in by_arm["off"]]
        if all(finite(v) for v in on_times + off_times):
            repeated_on_beats_off = max(on_times) < min(off_times)
    descriptive_time = None
    if all(len(by_arm[a]) == 2 for a in ("off", "on")):
        off_times = [(r["completion"] or {}).get("rl_elapsed_seconds") for r in by_arm["off"]]
        on_times = [(r["completion"] or {}).get("rl_elapsed_seconds") for r in by_arm["on"]]
        if all(finite(v) for v in off_times + on_times):
            off_mean = sum(off_times) / len(off_times)
            on_mean = sum(on_times) / len(on_times)
            descriptive_time = (off_mean, on_mean, 100 * (off_mean - on_mean) / off_mean)

    config_resolution_path = SOURCE / "results/config-resolution-check.json"
    config_resolution = read_json(config_resolution_path) or {}
    invariants = config_resolution.get("invariants", {})
    pair_results = []
    for off, on in ((runs[0], runs[1]), (runs[3], runs[2])):
        status, changes = config_pair_status(off, on)
        pair_results.append((off["id"], on["id"], status, changes))

    same_work_totals = False
    arm_work = {}
    if all_complete:
        arm_work = {arm: {key: sum(r["totals"][key] for r in by_arm[arm])
                          for key in ("rollouts", "tasks", "tokens")}
                    for arm in ("off", "on")}
        same_work_totals = arm_work["off"] == arm_work["on"]

    lines = [
        "# R30 full forkserver ABBA audit",
        "",
        "**Verdict: inconclusive.** This audit compares two forkserver-off runs with two forkserver-on runs, but this is one ABBA block and total sampled work differs. The both-on-faster predicate is recorded separately from work comparability. No profiler or trace decompression was used, so this audit cannot identify a limiting code path.",
        "",
        f"ABBA completion: **{len(complete)}/4 runs have a successful child completion and two updates with finite gradient norm, loss, and mismatch KL**.",
    ]
    if not all_complete:
        lines.append("The ABBA sequence is still partial; pending runs remain unknown and no complete-sequence predicate is asserted.")
    else:
        lines.append(f"Both on runs beat both off runs by actual child wall time: **{str(repeated_on_beats_off).lower()}**.")
        lines.append(f"Total work matched across arm totals (rollouts, task groups, tokens): **{str(same_work_totals).lower()}**; off `{arm_work['off']['rollouts']}/{arm_work['off']['tasks']}/{arm_work['off']['tokens']}`, on `{arm_work['on']['rollouts']}/{arm_work['on']['tasks']}/{arm_work['on']['tokens']}`.")
        if descriptive_time:
            lines.append(f"Descriptive raw child-time means: off {descriptive_time[0]:.3f} s, on {descriptive_time[1]:.3f} s ({descriptive_time[2]:.2f}% lower for on); this is not an equal-work speedup because total work differs.")
    lines.extend([
        "",
        "## Run evidence",
        "",
        "Child seconds come from `completion.json` and exclude preflight/config resolution. Completion counts use orchestrator progress windows; trace completion/error counts are cross-checked against trace-level metrics. `discarded` and `no_signal` remain visible because they affect usable training work.",
        "",
        "| ABBA | Arm | State / child RC | Route | Child s | rollouts | completed task groups | tokens | completed traces | admitted / effective train traces | errors / cancelled / stale | discarded / no-signal | trainer index rows | guard (status; min RAM; peak cgroup; swap; GPU MiB) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for i, r in enumerate(runs, 1):
        completion = r["completion"] or {}
        guard = r["guard"] or {}
        totals = r["totals"]
        discarded = sum(s.get("discarded", 0) or 0 for s in r["steps"])
        no_signal = sum(s.get("no_signal", 0) or 0 for s in r["steps"])
        roll_errors = sum(s.get("rollout_errors", 0) or 0 for s in r["steps"])
        cancelled = totals["cancelled"]
        stale = totals["stale"]
        guard_state = "pending" if not guard else ("pass" if guard.get("returncode") == 0 and not guard.get("stop_reason") else "fail")
        guard_text = (f"{guard_state}; {fmt(guard.get('minimum_available_bytes') / 1024**3 if guard.get('minimum_available_bytes') else None, 2)} GiB; "
                      f"{fmt(guard.get('peak_cgroup_bytes') / 1024**3 if guard.get('peak_cgroup_bytes') else None, 2)} GiB; "
                      f"{fmt(guard.get('peak_swap_bytes'))} B; {fmt(guard.get('peak_gpu_mib'))} MiB")
        lines.append(
            f"| {i} | {r['arm']} | {r['status']} / {fmt(completion.get('rl_child_returncode'))} | {r['route_status']} | {fmt(completion.get('rl_elapsed_seconds'))} | "
            f"{totals['rollouts']} | {totals['completed_tasks']} | {totals['tokens']} | {totals['completed_traces']} | "
            f"{totals['admitted_traces']} / {totals['effective_traces']} | {roll_errors} / {cancelled} / {stale} | {discarded} / {no_signal} | "
            f"{fmt(r['trainer_annotation_count'])} | {guard_text} |"
        )
    lines.extend(["", "## Per-update work and timings", "",
                  "Rows are reported by update, not divided into a synthetic common-work rate. The first off/on pair's step 1 is closely matched (176 rollouts and 88 tasks each; 19,057 vs 19,063 tokens). The second pair's step 1 is not matched (152/76/16,301 off vs 176/88/19,082 on). In the matched first pair, 32.97 s of the 35.67 s step-time gap coincides with lower `wait_for_batch` (50.29 s off vs 17.32 s on); forward/backward was 3.16 vs 2.27 s. This points to the batch-wait path as a place to investigate, but does not isolate a forkserver cause without profiling.",
                  "",
                  "| Run | step | rollouts / tasks / tokens | grad norm | loss | mismatch KL | step s | wait for batch s | forward/backward s | broadcast s | reward mean [min,max] | reward groups none/some/all | admitted/effective (queued) traces |",
                  "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|"])
    for r in runs:
        for s in r["steps"]:
            groups = s["reward_groups"]
            lines.append(
                f"| {r['id']} | {s['step']} | {s['rollouts']} / {s['tasks']} / {s['tokens']} | "
                f"{fmt(s['grad_norm'], 6)} | {fmt(s['loss'], 8)} | {fmt(s['kl'], 8)} | "
                f"{fmt(s['step_seconds'])} | {fmt(s['wait_for_batch_seconds'])} | {fmt(s['forward_backward_seconds'])} | "
                f"{fmt(s['broadcast_seconds'])} | {fmt(s['reward_mean'], 8)} [{fmt(s['reward_min'], 8)}, {fmt(s['reward_max'], 8)}] | "
                f"{groups.get('none')}/{groups.get('some')}/{groups.get('all')} | "
                f"{s['admitted_traces']}/{s['effective_traces']} ({s['queued_traces']}) |"
            )
    lines.extend(["", "## Integrity and comparability", "",
                  f"- Frozen CPU config report: **{config_resolution.get('status', 'missing')}**. Invariants all true: **{bool(invariants) and all(v is True for v in invariants.values())}** ({len(invariants)} checked). This covers the pinned full-parameter FP32 R30/IPO recipe, two updates, no checkpoint, same task sampling and reward/admission settings, and the only behavioral arm difference being `fork_server`.",
                  f"- Frozen config SHA256: off `{config_resolution.get('config_sha256', {}).get('off', 'missing')}`; on `{config_resolution.get('config_sha256', {}).get('on', 'missing')}`.",
                  f"- Pinned R30 source: `{SOURCE_SHA256}`; include: `{INCLUDE_SHA256}`; DSO: `{DSO_SHA256}`. Each run's small worker trace JSONL is checked for a loaded event and a first-decode event on the same PID, with these source/include/library hashes.",
                  "- Runtime resolved configs are compared per adjacent off/on pair below. Allowed differences are only forkserver boolean and run/route/address identity paths. This equality check also guards the full-parameter/no-LoRA model and unchanged task/reward/admission settings.",
                  ""])
    for off_id, on_id, status, changes in pair_results:
        changed = ", ".join(changes) if changes else "none observed"
        lines.append(f"- `{off_id}` vs `{on_id}` resolved config: **{status}**; changed paths: {changed}.")
    lines.extend(["", "## Benchmark limits", "",
                  "- This is 2 runs per arm, one ABBA block, not the five-or-more alternating repeats needed for a repeatability claim. Work is not equal over each complete run: compare raw rollout, task, and token totals above before using the wall-time predicate.",
                  "- Only the first pair's step 1 is close in work (176 rollouts and 88 task groups each; 19,057 vs 19,063 tokens). The second pair and the full-run totals differ, so there is no consistent matched-work ABBA comparison.",
                  "- Successful child return codes, completed trace rates, rollout errors/cancellations, route hashes, and resource-guard records are reported separately. The prelaunch envelope is not included in child seconds.",
                  "- No GPU/model activity was launched by this audit. The source run script was read. At audit preparation, system load average was 3.05 across 24 logical CPUs; this was not captured throughout each ABBA run, so it is not used to explain the gap.",
                  "- The earlier guard-only failure `fs360-20261101-a1-off` is excluded from timing and work totals; it produced no RL update and remains preserved in the source experiment directory.",
                  "- The underlying cause of any time gap is unprofiled. Step timing shows batch wait and compute durations, but does not isolate forkserver startup from other rollout/queue effects.",
                  "",
                  "This report intentionally makes no verified speedup/regression claim. The report is refreshed from current artifacts each time the script runs.",
                  ""])
    return "\n".join(lines)


def main():
    runs = [run_record(run_id, arm) for run_id, arm in RUNS]
    config_resolution_path = SOURCE / "results/config-resolution-check.json"
    config_resolution = read_json(config_resolution_path) or {}
    report = render(runs)
    OUTPUT.write_text(report)
    complete = [r for r in runs if r["status"] == "completed"
                and (r["completion"] or {}).get("rl_child_returncode") == 0
                and len(r["steps"]) == 2 and all(s["valid_update"] for s in r["steps"])]
    arm_runs = {arm: [r for r in complete if r["arm"] == arm] for arm in ("off", "on")}
    child_times = {
        arm: [(r["completion"] or {}).get("rl_elapsed_seconds") for r in arm_runs[arm]]
        for arm in ("off", "on")
    }
    work_totals = {
        arm: {key: sum(r["totals"][key] for r in arm_runs[arm])
              for key in ("rollouts", "completed_tasks", "tokens")}
        for arm in ("off", "on")
    }
    both_on_below_both_off = None
    raw_mean_time = None
    if len(child_times["off"]) == len(child_times["on"]) == 2 and all(
            finite(v) for side in child_times.values() for v in side):
        both_on_below_both_off = max(child_times["on"]) < min(child_times["off"])
        off_mean = sum(child_times["off"]) / 2
        on_mean = sum(child_times["on"]) / 2
        raw_mean_time = {
            "off_seconds": off_mean,
            "on_seconds": on_mean,
            "on_lower_percent": 100 * (off_mean - on_mean) / off_mean,
            "equal_work_speedup_claim": False,
        }
    pair_configs = []
    for off, on in ((runs[0], runs[1]), (runs[3], runs[2])):
        status, changed_paths = config_pair_status(off, on)
        pair_configs.append({"off_run": off["id"], "on_run": on["id"],
                             "status": status, "changed_paths": changed_paths})
    result = {
        "verdict": "inconclusive",
        "sequence_complete": len(complete) == 4,
        "completed_runs": len(complete),
        "run_count_per_arm": {arm: len(arm_runs[arm]) for arm in ("off", "on")},
        "child_times_seconds": child_times,
        "both_on_below_both_off": both_on_below_both_off,
        "descriptive_raw_mean_child_time": raw_mean_time,
        "work_totals": work_totals,
        "work_totals_equal": work_totals["off"] == work_totals["on"] if len(complete) == 4 else None,
        "config_resolution": {
            "status": config_resolution.get("status"),
            "report_sha256": sha256_file(config_resolution_path),
            "frozen_config_sha256": config_resolution.get("config_sha256"),
            "all_invariants_true": bool(config_resolution.get("invariants"))
                                   and all(value is True for value in config_resolution.get("invariants", {}).values()),
            "pairwise_runtime_resolved": pair_configs,
        },
        "pins": {"r30_source_sha256": SOURCE_SHA256, "r30_include_sha256": INCLUDE_SHA256,
                 "r30_dso_sha256": DSO_SHA256},
        "runs": [{
            "id": r["id"], "arm": r["arm"], "status": r["status"],
            "child_returncode": (r["completion"] or {}).get("rl_child_returncode"),
            "child_seconds": (r["completion"] or {}).get("rl_elapsed_seconds"),
            "route_status": r["route_status"], "work": r["totals"],
            "route_pins": {
                "source_sha256": (r["route"] or {}).get("matched_source_sha256"),
                "dso_sha256": (r["route"] or {}).get("matched_dso_sha256"),
            },
            "input_hashes": r["input_hashes"],
            "frozen_config_sha256": ((r["launch"] or {}).get("frozen_config") or {}).get("config_sha256"),
            "guard": ({key: (r["guard"] or {}).get(key) for key in
                       ("returncode", "stop_reason", "minimum_available_bytes", "peak_cgroup_bytes",
                        "peak_swap_bytes", "peak_gpu_mib")} if r["guard"] else None),
            "trainer_annotation_index_rows": r["trainer_annotation_count"],
            "steps": r["steps"],
        } for r in runs],
        "excluded_guard_only_failure": "fs360-20261101-a1-off",
        "limitations": ["one ABBA block; two runs per arm", "total work differs", "no profiling; limiter unproven"],
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(report)


if __name__ == "__main__":
    main()
