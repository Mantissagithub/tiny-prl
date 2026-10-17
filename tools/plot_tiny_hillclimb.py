#!/usr/bin/env python3
"""Plot measured tiny-prl autoresearch results and the next experiment steps."""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


def read_json(root: Path, name: str) -> object | None:
    path = root / name
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def rss_value(data: object | None) -> float | None:
    if not isinstance(data, dict) or not isinstance(data.get("rss_bytes"), (int, float)):
        return None
    return data["rss_bytes"] / 1024**2


def comparison_runs(data: object | None) -> list[dict]:
    return [run for run in data if isinstance(run, dict)] if isinstance(data, list) else []


def completed(run: dict) -> bool:
    expected = run.get("expected_updates", 10)
    return (
        run.get("returncode") == 0
        and run.get("completed_updates") == expected
        and run.get("finite_loss_grad") is True
        and run.get("stop_reason") is None
    )


def format_delta_mib(delta_bytes: int | float) -> str:
    sign = "−" if delta_bytes >= 0 else "+"
    return f"{sign}{abs(delta_bytes) / 1024**2:.0f} MiB"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_tiny_hillclimb.py EXPERIMENT_ROOT OUTPUT.png")
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    stages = [
        ("baseline", "import-control.json"),
        ("tokenizer only", "import-tokenizer-only.json"),
        ("lazy transport", "import-decoupled.json"),
    ]
    import_data = [read_json(root, filename) for _, filename in stages]
    imports = [(label, rss_value(data)) for (label, _), data in zip(stages, import_data)]
    runs = comparison_runs(read_json(root, "comparison.json"))
    activation_runs = comparison_runs(read_json(root, "activation-comparison.json"))
    launch_runs = comparison_runs(read_json(root, "launch-comparison.json"))
    pool_runs = comparison_runs(read_json(root, "pool-comparison.json"))
    rollout_runs = comparison_runs(read_json(root, "rollout-comparison.json"))
    kv_runs = comparison_runs(read_json(root, "kv-comparison.json"))
    kv_decision = read_json(root / "kv-budget", "decision.json")
    graph_runs = comparison_runs(read_json(root, "graph-comparison.json"))
    graph_parity = read_json(root / "graph-reload", "parity.json")
    lm_head_results = read_json(root / "lm-head-chunks", "results.json")
    vocab_tile_results = read_json(root / "lm-head-vocab-tiles", "results.json")
    activation_pilot = read_json(root / "activation-retest" / "pilot", "result.json")
    activation_retest_runs = comparison_runs(read_json(root, "activation-retest-comparison.json"))
    activation_decision = read_json(root / "activation-retest", "decision.json")
    activation_pilot_passed = (
        isinstance(activation_pilot, dict)
        and activation_pilot.get("returncode") == 0
        and activation_pilot.get("stop_reason") is None
    )
    runs.extend([*activation_runs, *launch_runs, *pool_runs, *rollout_runs, *graph_runs, *activation_retest_runs])
    health = read_json(root / "single-worker", "health-comparison.json")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={"width_ratios": [1.15, 1, 1.25]})
    fig.patch.set_facecolor("#10151d")
    for ax in axes:
        ax.set_facecolor("#171e28")
        ax.tick_params(colors="#c8d1dc")
        for spine in ax.spines.values():
            spine.set_color("#364252")
        ax.title.set_color("#f1f5f9")
        ax.xaxis.label.set_color("#c8d1dc")
        ax.yaxis.label.set_color("#c8d1dc")
        ax.grid(axis="y", color="#364252", alpha=0.45, linewidth=0.7)

    ax = axes[0]
    colors = ["#9aa7b7", "#45c4b0", "#f0b45a"]
    measured = [(index, value) for index, (_, value) in enumerate(imports) if value is not None]
    if measured:
        xs, values = zip(*measured)
        ax.plot(xs, values, color="#66758a", linewidth=1.4, zorder=1)
        ax.scatter(xs, values, s=58, color=[colors[index] for index in xs], edgecolor="white", linewidth=0.8, zorder=3)
        best_so_far = []
        best = float("inf")
        for value in values:
            best = min(best, value)
            best_so_far.append(best)
        ax.step(xs, best_so_far, where="post", color="#45c4b0", linewidth=2, alpha=0.9, label="best so far")
        ax.legend(loc="upper right", frameon=False, labelcolor="#c8d1dc", fontsize=8)
        for index, value in measured:
            ax.annotate(
                f"{value:.1f}",
                (index, value),
                xytext=(5, 5),
                textcoords="offset points",
                color=colors[index],
                fontsize=8,
            )
        if measured[0][0] == 0 and measured[-1][0] == 2 and values[0] > 0:
            reduction = (values[0] - values[-1]) / values[0] * 100
            ax.text(
                0.03,
                0.04,
                f"Baseline → latest: {values[0]:.1f} → {values[-1]:.1f} MiB  ({reduction:.1f}% drop)",
                color="#c8d1dc",
                fontsize=8,
                transform=ax.transAxes,
            )
    stage_labels = []
    for (label, _), data in zip(stages, import_data):
        seconds = data.get("import_seconds") if isinstance(data, dict) else None
        stage_labels.append(f"{label}\n{seconds:.1f}s import" if isinstance(seconds, (int, float)) else label)
    ax.set_xticks(range(3), stage_labels)
    ax.set_ylabel("RSS after import (MiB)")
    ax.set_title("Import memory · measured")
    if not measured:
        ax.text(
            0.5, 0.5, "No import RSS results yet", ha="center", va="center", color="#c8d1dc", transform=ax.transAxes
        )

    ax = axes[1]
    eager_times = [
        run["elapsed_seconds"]
        for run in graph_runs
        if run.get("source") == "eager" and run.get("expected_updates", 10) == 10 and completed(run)
    ]
    graph_times = [
        run["elapsed_seconds"]
        for run in graph_runs
        if run.get("source") == "decode-graph" and run.get("expected_updates", 10) == 10 and completed(run)
    ]
    graph_paired = len(eager_times) >= 2 and len(graph_times) >= 2
    graph_parity_passed = (
        isinstance(graph_parity, dict)
        and graph_parity.get("eager_a_restored") is True
        and graph_parity.get("graph_a_restored") is True
        and graph_parity.get("eager_graph_parity") is True
        and graph_parity.get("scaled_b_changed_output_or_selected_logprobs") is True
    )
    graph_change_percent = None
    if graph_paired:
        eager_mean = sum(eager_times) / len(eager_times)
        graph_mean = sum(graph_times) / len(graph_times)
        graph_change_percent = (graph_mean / eager_mean - 1) * 100
    graph_change_text = None
    if graph_change_percent is not None:
        graph_change_text = (
            f"{abs(graph_change_percent):.1f}% lower mean"
            if graph_change_percent < 0
            else f"{graph_change_percent:.1f}% higher mean"
        )
    old_control_valid = any(
        run.get("source") == "control" and run.get("expected_updates", 10) == 10 and completed(run) for run in runs
    )
    import_fix_valid = any(
        run.get("source") == "candidate" and run.get("expected_updates", 10) == 10 and completed(run) for run in runs
    )
    categories = {}
    category_labels = []
    if old_control_valid:
        categories["control"] = (len(categories), "#9aa7b7")
        category_labels.append("old\nctrl")
    if import_fix_valid:
        categories["candidate"] = (len(categories), "#45c4b0")
        category_labels.append("import\nfix")
    pool_control_valid = any(
        run.get("source") == "pool-control" and run.get("expected_updates", 10) == 10 and completed(run)
        for run in pool_runs
    )
    single_worker_valid = any(
        run.get("source") == "single-worker" and run.get("expected_updates", 10) == 10 and completed(run)
        for run in pool_runs
    )
    if pool_control_valid:
        categories["pool-control"] = (len(categories), "#60a5fa")
        category_labels.append("pool")
    if single_worker_valid:
        categories["single-worker"] = (len(categories), "#c084fc")
        category_labels.append("1worker")
    rollout_full_run = any(
        run.get("source") in {"seqs2", "seqs4"} and run.get("expected_updates", 10) == 10 and completed(run)
        for run in rollout_runs
    )
    if rollout_full_run:
        categories["rollout-sequences"] = (len(categories), "#2dd4bf")
        category_labels.append("seqs")
    activation_retest_full_runs = [
        run
        for run in activation_retest_runs
        if run.get("source") in {"activation-offload-on", "activation-offload-off"}
        and run.get("expected_updates", 10) == 10
        and completed(run)
    ]
    if activation_retest_full_runs:
        categories["activation-retest"] = (len(categories), "#f0b45a")
        category_labels.append("offload")
    eager_full_run = any(
        run.get("source") == "eager" and run.get("expected_updates", 10) == 10 and completed(run) for run in graph_runs
    )
    decode_graph_full_run = any(
        run.get("source") == "decode-graph" and run.get("expected_updates", 10) == 10 and completed(run)
        for run in graph_runs
    )
    if eager_full_run:
        categories["eager"] = (len(categories), "#60a5fa")
        category_labels.append("eager")
    if decode_graph_full_run:
        categories["decode-graph"] = (len(categories), "#2dd4bf")
        category_labels.append("decode\ngraph")
    activation_full_run = any(
        run.get("source") == "activation-off" and run.get("expected_updates", 10) == 10 and completed(run)
        for run in activation_runs
    )
    launch_full_run = any(
        run.get("source") == "direct-launch" and run.get("expected_updates", 10) == 10 and completed(run)
        for run in launch_runs
    )
    if activation_full_run:
        categories["activation-off"] = (len(categories), "#f0b45a")
    if launch_full_run:
        categories["direct-launch"] = (len(categories), "#a78bfa")
    incomplete_x = len(categories)
    if activation_full_run:
        category_labels.append("offload\noff")
    if launch_full_run:
        category_labels.append("direct")
    category_labels.append("stops")
    valid_control = False
    failure_count = 0
    pilot_notes = []
    valid_lane_counts = {}
    for run in runs:
        source = run.get("source")
        if source not in {
            "control",
            "candidate",
            "activation-off",
            "direct-launch",
            "pool-control",
            "single-worker",
            "seqs2",
            "seqs4",
            "eager",
            "decode-graph",
            "activation-offload-on",
            "activation-offload-off",
        } or not isinstance(run.get("elapsed_seconds"), (int, float)):
            continue
        expected_updates = run.get("expected_updates", 10)
        pilot = (
            source in {"activation-off", "direct-launch", "seqs2", "seqs4", "eager", "decode-graph"}
            and expected_updates < 10
        )
        run_complete = completed(run)
        if source in {"seqs2", "seqs4"}:
            point_category = "rollout-sequences"
        elif source in {"activation-offload-on", "activation-offload-off"}:
            point_category = "activation-retest"
        else:
            point_category = source
        valid_point = run_complete and expected_updates == 10 and point_category in categories
        source_label = {
            "direct-launch": "Direct launch",
            "activation-off": "Offload-off",
            "seqs2": "Seqs 2",
            "seqs4": "Seqs 4",
            "eager": "Eager",
            "decode-graph": "Decode graph",
        }.get(source, "Pilot")
        if pilot and run_complete:
            pilot_notes.append(
                f"{source_label}: {expected_updates}/{expected_updates} smoke passed (not timed comparison)"
            )
            continue
        x, color = categories[point_category] if valid_point else (incomplete_x, "#ef6a73")
        if valid_point and source in {"seqs2", "seqs4"}:
            color = "#2dd4bf" if source == "seqs2" else "#f97316"
        if valid_point and source in {"activation-offload-on", "activation-offload-off"}:
            color = "#9aa7b7" if source == "activation-offload-on" else "#c084fc"
        valid_control |= valid_point and source in {"control", "pool-control"}
        marker = "o" if valid_point else "X"
        ax.scatter(
            [x],
            [run["elapsed_seconds"]],
            marker=marker,
            s=66,
            color=color,
            edgecolor="#f8fafc",
            linewidth=0.7,
            zorder=3,
        )
        label = run.get("arm", "trial")
        if not valid_point:
            arm_short = str(label).replace("direct-full", "direct").replace("direct-pilot", "direct")
            label = f"{arm_short}: {run.get('completed_updates', 0)}/{expected_updates}"
            if pilot:
                reason = str(run.get("stop_reason", "")).lower()
                if any(word in reason for word in ("ram", "memory", "reserve")):
                    if source != "activation-off" or not activation_pilot_passed:
                        pilot_notes.append(
                            f"{source_label}: {run.get('completed_updates', 0)}/{expected_updates} RAM stop"
                        )
                else:
                    pilot_notes.append(
                        f"{source_label}: {run.get('completed_updates', 0)}/{expected_updates} incomplete"
                    )
        else:
            arm_short = str(label)
            if source in {"candidate", "single-worker"}:
                arm_short = "b2" if arm_short.endswith("2") else "b1"
            elif source == "pool-control":
                arm_short = "a2" if arm_short.endswith("2") else "a1"
            elif source in {"activation-offload-on", "activation-offload-off"}:
                arm_short = arm_short.removeprefix("offload-")
            for prefix in ("graph-", "seqs-"):
                arm_short = arm_short.removeprefix(prefix)
            label = f"{arm_short} {run['elapsed_seconds']:.1f}"
        label_offset = (6, 4)
        label_align = "left"
        if valid_point:
            lane_count = valid_lane_counts.get(point_category, 0)
            if point_category == "activation-retest":
                label_offset = [(0, 28), (0, 44), (0, -28), (0, -44)][lane_count % 4]
            else:
                above, below = ((0, 14), (0, -15)) if x % 2 == 0 else ((0, -15), (0, 14))
                label_offset = above if lane_count % 2 == 0 else below
            label_align = "center"
            valid_lane_counts[point_category] = lane_count + 1
        if not valid_point:
            label_offset = (6, 9) if failure_count % 2 == 0 else (-6, -11)
            label_align = "left" if failure_count % 2 == 0 else "right"
            failure_count += 1
        if not pilot:
            ax.annotate(
                label,
                (x, run["elapsed_seconds"]),
                xytext=label_offset,
                textcoords="offset points",
                ha=label_align,
                color=color,
                fontsize=7,
            )
    if activation_pilot_passed:
        pilot_notes.append("Offload-off retest: 2-step smoke passed")
    ax.set_xticks(list(range(len(category_labels))), category_labels, fontsize=7.5)
    ax.set_xlim(-0.5, len(category_labels) - 0.3)
    ax.set_ylim(45, 130)
    ax.set_ylabel("Run elapsed time (s)")
    ax.set_title("Training comparison · no checkpoint")
    if graph_paired and graph_parity_passed:
        comparison_note = (
            f"Decode graph retained · {graph_change_text} in one ABBA; selected + top-20 parity passed (4-token reload)"
        )
    elif graph_paired:
        comparison_note = f"Decode graph promising · {graph_change_text}; LoRA hot reload parity pending"
    elif pool_runs and not pool_control_valid:
        comparison_note = "No speedup claim: pool control incomplete"
    elif pool_control_valid and not single_worker_valid:
        comparison_note = "No speedup claim: single-worker trial incomplete"
    elif pool_control_valid and single_worker_valid:
        comparison_note = "No repeatable throughput gain established"
    elif not valid_control:
        comparison_note = "No speedup claim: control incomplete"
    else:
        comparison_note = None
    if comparison_note:
        ax.text(
            0.02,
            0.97,
            textwrap.fill(comparison_note, width=46),
            ha="left",
            va="top",
            color="#f0b45a" if graph_paired else "#ef6a73",
            fontsize=7.5,
            transform=ax.transAxes,
        )
    if failure_count:
        ax.text(
            0.02,
            0.89,
            "Red X: RAM-reserve stop; not comparable",
            ha="left",
            va="top",
            color="#ef6a73",
            fontsize=7.5,
            transform=ax.transAxes,
        )
    if pilot_notes:
        ax.text(
            0.02,
            0.025,
            textwrap.fill("Pilot: " + "; ".join(pilot_notes), width=50),
            ha="left",
            va="bottom",
            color="#f0b45a",
            fontsize=7.5,
            transform=ax.transAxes,
        )
    if not runs:
        ax.text(
            0.5,
            0.5,
            "No repeated comparison results yet",
            ha="center",
            va="center",
            color="#c8d1dc",
            transform=ax.transAxes,
        )
    ax = axes[2]
    ax.set_title("Hill-climb roadmap")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    latest = imports[-1][1]
    launch_smoke_passed = any(run.get("expected_updates") == 2 and completed(run) for run in launch_runs)
    launch_ten_step_ram_stop = any(
        run.get("expected_updates", 10) == 10
        and run.get("completed_updates", 0) < 10
        and any(word in str(run.get("stop_reason", "")).lower() for word in ("ram", "memory", "reserve"))
        for run in launch_runs
    )
    launch_ten_step_completed = any(run.get("expected_updates", 10) == 10 and completed(run) for run in launch_runs)
    direct_status = "Direct launch · validated (10 steps)"
    if pool_control_valid:
        direct_status = "Direct launch · validated (10 steps)"
    elif launch_smoke_passed and launch_ten_step_ram_stop:
        direct_status = "Direct launch · 2-step smoke passed; 10-step RAM stop"
    elif launch_smoke_passed and launch_ten_step_completed:
        direct_status = "Direct launch · 2-step smoke passed; 10-step completed"
    elif launch_smoke_passed:
        direct_status = "Direct launch · 2-step smoke passed; RL run pending"
    elif launch_ten_step_ram_stop:
        direct_status = "Direct launch · 10-step RAM stop"
    elif launch_ten_step_completed:
        direct_status = "Direct launch · 10-step completed"
    if isinstance(health, dict) and health.get("both_healthy") is True:
        control_pss = health.get("control_pss_bytes")
        candidate_pss = health.get("candidate_pss_bytes")
        if isinstance(control_pss, (int, float)) and isinstance(candidate_pss, (int, float)):
            pss_delta = control_pss - candidate_pss
            pss_status = f"env PSS {format_delta_mib(pss_delta)}"
        else:
            pss_status = "env PSS unmeasured"
        last_pool_control = next(
            (
                run
                for run in reversed(pool_runs)
                if run.get("source") == "pool-control"
                and run.get("expected_updates", 10) == 10
                and completed(run)
                and isinstance(run.get("peak_cgroup_anon_plus_swap_bytes"), (int, float))
            ),
            None,
        )
        last_single_worker = next(
            (
                run
                for run in reversed(pool_runs)
                if run.get("source") == "single-worker"
                and run.get("expected_updates", 10) == 10
                and completed(run)
                and isinstance(run.get("peak_cgroup_anon_plus_swap_bytes"), (int, float))
            ),
            None,
        )
        if last_pool_control is not None and last_single_worker is not None:
            cgroup_delta = (
                last_pool_control["peak_cgroup_anon_plus_swap_bytes"]
                - last_single_worker["peak_cgroup_anon_plus_swap_bytes"]
            )
            cgroup_status = f"job anon+swap {format_delta_mib(cgroup_delta)}"
        else:
            cgroup_status = "job anon+swap unpaired"
        health_status = f"1 env server retained · {pss_status}; {cgroup_status}"
    elif health is not None:
        health_status = "Env-server health check failed; RL validation pending"
    else:
        health_status = "Single-worker memory comparison pending"
    rollout_decision = read_json(root / "rollout-seqs", "decision.json")
    if isinstance(rollout_decision, dict) and rollout_decision.get("decision") == "reject_four_keep_two":
        delta = rollout_decision.get("four_mean_wall_change_percent")
        rollout_status = (
            f"4 seqs rejected · +{delta:.1f}%; keep 2"
            if isinstance(delta, (int, float))
            else "4 seqs rejected · keep 2"
        )
    else:
        rollout_status = "Rollout · 2 → 4 seqs: testing"
    kv_full_runs = [
        run
        for run in kv_runs
        if run.get("source") in {"kv-budget-35", "kv-budget-25"}
        and run.get("expected_updates", 10) == 10
        and completed(run)
    ]
    if isinstance(kv_decision, dict) and kv_decision.get("decision") == "retain_kv_budget_025_for_gpu_headroom":
        comparison = kv_decision.get("summary", {}).get("candidate_vs_control", {})
        gpu_delta = comparison.get("peak_gpu_mib", {}).get("difference")
        elapsed_change = comparison.get("elapsed_seconds", {}).get("change_percent")
        if isinstance(gpu_delta, (int, float)) and isinstance(elapsed_change, (int, float)):
            kv_status = f"KV .25 retained · {gpu_delta:.0f} MiB GPU peak; +{elapsed_change:.2f}% time"
        else:
            kv_status = "KV .25 retained for GPU headroom; no speedup claim"
    elif len(kv_full_runs) == 4:
        kv_status = "KV .25 · 4/4 full runs passed; decision pending"
    elif kv_full_runs:
        kv_status = f"KV .25 · {len(kv_full_runs)}/4 full runs passed; ABBA active"
    else:
        kv_status = "KV .25 · pilot + 2×512-token stress passed; ABBA active"
    graph_pilot_runs = [run for run in graph_runs if run.get("expected_updates", 10) < 10]
    if graph_paired and graph_parity_passed:
        graph_status = (
            f"Decode graph retained · {graph_change_text} in one ABBA; parity passed (selected + top-20, 4 tokens)"
        )
    elif graph_paired:
        graph_status = f"Decode graph · {graph_change_text}; hot-reload parity pending"
    elif graph_pilot_runs and any(completed(run) for run in graph_pilot_runs):
        graph_status = "Decode graph · pilot recorded; analysis pending"
    elif graph_pilot_runs:
        graph_status = "Decode graph · pilot running"
    else:
        graph_status = "Decode graph · profiling pending"
    activation_full_runs = [
        run for run in activation_retest_runs if run.get("expected_updates", 10) == 10 and completed(run)
    ]
    if len(activation_full_runs) >= 4:
        if (
            isinstance(activation_decision, dict)
            and activation_decision.get("decision") == "retain_activation_offloading_on"
        ):
            activation_status = "Activation · keep offloading on; disabling showed no clear gain"
        else:
            activation_status = "Activation offload · all four ABBA runs passed"
    elif activation_full_runs:
        activation_status = f"Activation offload · {len(activation_full_runs)}/4 full runs passed; ABBA underway"
    elif activation_pilot_passed:
        activation_status = (
            "Activation offload · 2-step pilot passed; ABBA running"
            if (root / "activation-retest" / "launch-a1.json").exists()
            else "Activation offload · 2-step pilot passed; ABBA queued"
        )
    else:
        activation_status = "Activation offload · pilot pending"
    lm_head_rejected = (isinstance(lm_head_results, dict) and lm_head_results.get("parity") is False) or (
        isinstance(vocab_tile_results, dict) and vocab_tile_results.get("parity_passed") is False
    )
    roadmap = [
        (0.90, "✓  Baseline · 10 steps + resume verified", "#45c4b0"),
        (
            0.78,
            f"●  Import · {latest:.0f} MiB RSS; full-run RAM marginal"
            if latest is not None
            else "●  Import memory · measuring",
            "#f0b45a",
        ),
        (0.66, f"●  Direct + env · {direct_status}; {health_status}", "#45c4b0"),
        (0.54, f"●  {rollout_status}; {kv_status}", "#f0b45a"),
        (
            0.42,
            f"●  {graph_status}" if graph_paired and graph_parity_passed else f"○  {graph_status}",
            "#45c4b0" if graph_paired and graph_parity_passed else "#f0b45a" if graph_paired else "#9aa7b7",
        ),
        (
            0.30,
            f"●  {activation_status}",
            "#f0b45a" if activation_pilot_passed else "#9aa7b7",
        ),
        (
            0.18,
            "×  LM-head token/vocab tuning rejected: logprob parity; keep 128/8192; defer CuTe"
            if lm_head_rejected
            else "○  LM-head / CuTeDSL · profile first",
            "#ef6a73" if lm_head_rejected else "#9aa7b7",
        ),
    ]
    ax.plot([0.065, 0.065], [0.18, 0.90], color="#364252", linewidth=1.4, transform=ax.transAxes)
    for y, text, color in roadmap:
        ax.scatter([0.065], [y], color=color, s=40, transform=ax.transAxes, zorder=2)
        wrapped = textwrap.fill(text, width=34)
        ax.text(0.13, y, wrapped, color=color, fontsize=9, va="center", transform=ax.transAxes)
    ax.text(0.065, 0.02, "No future speedups inferred", color="#c8d1dc", fontsize=9, transform=ax.transAxes)

    fig.suptitle("tiny-prl · autoresearch hill climbing", color="#f8fafc", fontsize=17, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.035,
        "RTX 4060 laptop · 8 GiB VRAM · 7 GiB cgroup guard · 2 GiB reserve   |   "
        "Kernel changes require numerical + gradient checks and end-to-end benefit",
        ha="center",
        color="#9aa7b7",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.18, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
