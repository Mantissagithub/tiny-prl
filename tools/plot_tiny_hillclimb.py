#!/usr/bin/env python3
"""Plot the tiny-prl hill climb: every measured run, the retained frontier, and what never reached wall time."""

import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
FRONTIER = "#2a78d6"
VERDICT = {
    "kept": ("#0ca30c", "faster, kept"),
    "slower": ("#d03b3b", "slower, rejected"),
    "neutral": ("#8a8984", "no measurable change"),
    "invalid": ("#ec835a", "fails strict parity (invalid)"),
}

# (label, artifact, control arm, candidate arm, verdict, note) for Qwen3-0.6B LoRA 10-update RL A-B-B-As.
QWEN_ABBA = [
    (
        "Exp 4\ninline env",
        "tiny-opt-tokenizer-20261017/pool-comparison.json",
        "pool-control",
        "single-worker",
        "neutral",
        "",
    ),
    ("Exp 5\n4 inference seqs", "tiny-opt-tokenizer-20261017/rollout-comparison.json", "seqs2", "seqs4", "slower", ""),
    (
        "Exp 6\ndecode CUDA graphs",
        "tiny-opt-tokenizer-20261017/graph-comparison.json",
        "eager",
        "decode-graph",
        "kept",
        "",
    ),
    (
        "Exp 9\nactivation offload off",
        "tiny-opt-tokenizer-20261017/activation-retest-comparison.json",
        "activation-offload-on",
        "activation-offload-off",
        "neutral",
        "",
    ),
    (
        "Exp 10\nKV budget .25",
        "tiny-opt-tokenizer-20261017/kv-comparison.json",
        "kv-budget-35",
        "kv-budget-25",
        "neutral",
        "kept for\n-762 MiB GPU",
    ),
    (
        "Exp 11\nAC off",
        "tiny-opt-tokenizer-20261017/ac-comparison.json",
        "checkpoint-full",
        "checkpoint-off",
        "neutral",
        "",
    ),
    (
        "Exp 12\nrollout cap 4",
        "tiny-opt-tokenizer-20261017/queue-comparison.json",
        "queue-adaptive",
        "queue-cap4",
        "neutral",
        "",
    ),
    ("Exp 14\nCuTe head", "tiny-kernel-20261018/comparison/summary.json", "a", "b", "neutral", ""),
    ("Exp 18\nTriton head", "tiny-kernel-20261018/triton-integration/abba-summary.json", "a", "b", "neutral", ""),
    (
        "Exp 20\nCuTe fused head",
        "tiny-kernel-20261018/cute-fused-integration/real-abba-20261019/analysis.json",
        "A",
        "B",
        "neutral",
        "",
    ),
]

# (label, trainer-abba tag, verdict) for SmolLM2-135M full-parameter trainer-only A-B-B-As on fixed fake data.
TRAINER_ABBA = [
    ("Triton\nRoPE", "rope", "neutral"),
    ("Triton\nSwiGLU", "swiglu", "slower"),
    ("torch.compile\n(Inductor)", "compile", "invalid"),
    ("Triton\nRMSNorm", "rmsnorm", "kept"),
    ("Triton\nRMSNorm+RoPE", "rope_rmsnorm", "kept"),
    ("residual RMSNorm\npre-exact-DW", "residual_fusion", "slower"),
    ("residual RMSNorm\nexact DW", "residual_fusion_dw_exact", "slower"),
]

# (label, rl-abba tag, verdict) for SmolLM2-135M full-parameter full-RL A-B-B-As with Triton RMSNorm+RoPE.
RL_ABBA = [
    ("autotuned\n(recompiles)", "triton", "slower"),
    ("shape-stable\n(start-up tune)", "tritonfix", "slower"),
    ("pinned\nconfigs", "pinned", "kept"),
]

MILESTONES = [
    ("135M full-param", "10 updates", True),
    ("135M full-param", "checkpoint + fresh resume", True),
    ("360M full-param", "10 updates", True),
    ("360M full-param", "checkpoint save: host-RAM floor", False),
    ("Qwen3-0.6B full-param", "FP32 state exceeds 8 GiB", False),
]


def load(path: Path) -> object:
    return json.loads(path.read_text())


def qwen_runs(root: Path, artifact: str, control: str, candidate: str) -> tuple[list[float], list[float]]:
    data = load(root / artifact)
    if isinstance(data, dict) and "arms" in data:
        rows = [(name, arm["guard"]["elapsed_seconds"]) for name, arm in data["arms"].items()]
        return [t for n, t in rows if n.startswith(control)], [t for n, t in rows if n.startswith(candidate)]
    rows = [row for row in data if isinstance(row, dict) and row.get("completed_updates", 10) == 10]
    rows = [r for r in rows if r.get("returncode", 0) == 0]
    if any(r.get("source") == control for r in rows):
        key = lambda r: r["source"]  # noqa: E731
    else:
        key = lambda r: r["arm"].split("-")[-1][0].lower()  # noqa: E731
    return (
        [r["elapsed_seconds"] for r in rows if key(r) == control],
        [r["elapsed_seconds"] for r in rows if key(r) == candidate],
    )


def trainer_runs(root: Path, tag: str) -> tuple[list[float], list[float]]:
    if tag == "residual_fusion":
        summary = root / "tiny-kernel-20261020/fusion-profile/abba/summary.json"
    elif tag == "residual_fusion_dw_exact":
        summary = root / "tiny-kernel-20261020/fusion-profile/abba-dw-exact-r1/summary.json"
    else:
        summary = root / "tiny-kernel-20261020/triton/trainer-abba" / tag / "summary.json"
    timing = load(summary)["timing"]
    run = lambda name: 1000 * timing[f"{name} time/forward_backward median_warm"]  # noqa: E731
    return [run("A1"), run("A2")], [run("B1"), run("B2")]


def rl_runs(root: Path, tag: str) -> dict[str, dict[str, float]]:
    base = root / "tiny-kernel-20261020/triton/rl-abba"
    runs = {}
    for name in ("A1", "B1", "B2", "A2"):
        rows = {}
        for line in (base / "runs" / f"rl-abba-{tag}-{name}" / "monitors/file/metrics.jsonl").read_text().splitlines():
            record = json.loads(line)
            if record.get("producer") == "trainer":
                rows.setdefault(record["step"], {}).update(record)
        fb = [rows[s]["time/forward_backward"] for s in sorted(rows)]
        wall = load(base / tag / name / "guard/result.json")["elapsed_seconds"]
        runs[name] = {"fb_ms": 1000 * statistics.median(fb[2:]), "wall_s": wall}
    return runs


def renderer_runs(root: Path) -> dict[str, tuple[float, float]]:
    data = load(root / "tiny-fulltrain-20261018/environment-memory-20261020/train-tokenizer-parity-r2-comparison.json")
    return {
        model: (
            data[model]["baseline_renderer_memory"]["pss_kib"] / 1024,
            data[model]["candidate_renderer_memory"]["pss_kib"] / 1024,
        )
        for model in ("135", "360")
    }


def style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)


def pairs(ax, x: float, control: list[float], candidate: list[float], verdict: str) -> None:
    color = VERDICT[verdict][0]
    ax.scatter([x - 0.14] * len(control), control, s=46, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.6, zorder=3)
    marker = "X" if verdict == "invalid" else "o"
    ax.scatter(
        [x + 0.14] * len(candidate),
        candidate,
        s=58,
        color=color,
        marker=marker,
        edgecolors=SURFACE,
        linewidths=1.2,
        zorder=4,
    )


def pct(control: list[float], candidate: list[float]) -> float:
    return 100 * (statistics.mean(candidate) / statistics.mean(control) - 1)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_tiny_hillclimb.py OUTPUTS_DIR OUTPUT.png")
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})

    fig = plt.figure(figsize=(17, 16.5), facecolor=SURFACE)
    grid = fig.add_gridspec(
        5,
        2,
        height_ratios=[1.25, 1.0, 1.0, 0.32, 0.72],
        hspace=0.62,
        wspace=0.16,
        left=0.055,
        right=0.985,
        top=0.855,
        bottom=0.04,
    )

    # Header: the honest bottom line as hero numbers.
    fig.text(
        0.055, 0.965, "tiny-prl hill climb on an RTX 4060 Laptop (8 GiB)", fontsize=20, fontweight="bold", color=INK
    )
    fig.text(
        0.055,
        0.945,
        "Every point is one guarded run. Hollow gray = control arm, filled = candidate arm "
        "(A-B-B-A order). A candidate is kept only if both its runs beat both controls and it passes strict parity.",
        fontsize=10,
        color=INK_2,
    )
    heroes = [
        ("-12.6%", "Qwen3-0.6B LoRA RL wall time", "decode CUDA graphs (Exp 6): the only end-to-end gain"),
        ("-15.6%", "135M full-RL trainer fwd/bwd", "Triton RMSNorm+RoPE (Exp 55); wall gain not attributable"),
        ("10 / 10", "135M full-parameter RL updates", "with checkpoint and fresh resume (Exp 48)"),
    ]
    for i, (value, label, note) in enumerate(heroes):
        x = 0.055 + i * 0.315
        fig.text(x, 0.9, value, fontsize=26, fontweight="bold", color=INK)
        fig.text(x + 0.095, 0.912, label, fontsize=10.5, color=INK, fontweight="bold")
        fig.text(x + 0.095, 0.895, note, fontsize=9, color=INK_2)

    # Panel A: Qwen LoRA end-to-end wall time, every run plus the measured retained-recipe frontier.
    ax = fig.add_subplot(grid[0, :])
    style(ax, "A  Qwen3-0.6B LoRA - 10-update RL wall time per run (lower is better)", "wall time (s)")
    record = None
    for x, (label, artifact, control, candidate, verdict, note) in enumerate(QWEN_ABBA):
        a, b = qwen_runs(root, artifact, control, candidate)
        pairs(ax, x, a, b, verdict)
        if verdict == "kept":
            record = (x, statistics.mean(a), statistics.mean(b))
        change = pct(a, b)
        ax.annotate(
            f"{change:+.1f}%",
            (x + 0.14, max(b)),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK,
            fontweight="bold" if verdict == "kept" else "normal",
            zorder=5,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": SURFACE, "edgecolor": "none"},
        )
        if note:
            ax.annotate(
                note,
                (x, min(a + b)),
                xytext=(0, -26),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=INK_2,
            )
    x0, before, after = record
    end = len(QWEN_ABBA) - 0.5
    ax.plot([-0.5, x0 + 0.14, x0 + 0.14], [before, before, after], color=FRONTIER, linewidth=2, zorder=2)
    ax.plot([x0 + 0.14, end], [after, after], color=FRONTIER, linewidth=2, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"record {before:.1f} s -> {after:.1f} s\ndecode CUDA graphs: the only\nattributable end-to-end win",
        (x0 + 0.14, after),
        xytext=(-150, -30),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1},
    )
    ax.text(
        end - 0.05,
        after - 1.2,
        "record holds; later controls drift with host load",
        ha="right",
        va="top",
        fontsize=8,
        color=FRONTIER,
    )
    ax.axvspan(6.5, len(QWEN_ABBA) - 0.5, color="#f1f0ec", zorder=0)
    ax.text(
        8,
        ax.get_ylim()[1],
        "LM-head kernels: component wins (17-40%)\ndid not reach wall time",
        ha="center",
        va="top",
        fontsize=8.5,
        color=INK_2,
    )
    ax.set_xticks(range(len(QWEN_ABBA)), [q[0] for q in QWEN_ABBA])
    ax.set_xlim(-0.6, len(QWEN_ABBA) - 0.4)

    # Panel B: 135M trainer-only, fixed fake data.
    ax = fig.add_subplot(grid[1, :])
    style(
        ax,
        "B  SmolLM2-135M full-parameter trainer, fixed-shape fake data - warm fwd/bwd per run "
        "(launch-bound: GPU 30% busy, 13k kernels/step)",
        "fwd/bwd (ms)",
    )
    for x, (label, tag, verdict) in enumerate(TRAINER_ABBA):
        a, b = trainer_runs(root, tag)
        pairs(ax, x, a, b, verdict)
        text = f"{pct(a, b):+.1f}%" + ("\nfails parity:\nnot a real gain" if verdict == "invalid" else "")
        ax.annotate(
            text,
            (x + 0.14, max(b)),
            xytext=(10, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8.5,
            color=INK,
            fontweight="bold" if verdict == "kept" else "normal",
        )
        if tag == "residual_fusion":
            ax.annotate(
                "Pre-exact-DW round: step +1.1%\n"
                "Torch all-RMSNorm gradients were 210/212;\n"
                "2 weight-gradient elements were pending",
                xy=(x + 0.14, max(b)),
                xytext=(3.05, 82),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=7.5,
                color=INK_2,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.9},
                bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": GRID},
            )
        elif tag == "residual_fusion_dw_exact":
            ax.annotate(
                "Exact-DW round: step +3.1%\n"
                "Torch all-RMSNorm gradients: 212/212 bitwise",
                xy=(x + 0.14, max(b)),
                xytext=(4.05, 125),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=7.5,
                color=INK_2,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.9},
                bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": GRID},
            )
    ax.set_xticks(range(len(TRAINER_ABBA)), [t[0] for t in TRAINER_ABBA])
    ax.set_xlim(-0.6, len(TRAINER_ABBA) - 0.4)
    ax.set_ylim(0, None)

    # Panel C: 135M full RL, small multiples: the trainer gain vs the wall time it fails to move.
    for col, (key, title, unit) in enumerate(
        [
            ("fb_ms", "C1  135M full-RL - trainer warm fwd/bwd (Triton RMSNorm+RoPE)", "fwd/bwd (ms)"),
            ("wall_s", "C2  135M full-RL - 10-update wall time (same runs)", "wall time (s)"),
        ]
    ):
        ax = fig.add_subplot(grid[2, col])
        style(ax, title, unit)
        for x, (label, tag, verdict) in enumerate(RL_ABBA):
            runs = rl_runs(root, tag)
            a = [runs["A1"][key], runs["A2"][key]]
            b = [runs["B1"][key], runs["B2"][key]]
            shown = verdict if key == "fb_ms" else ("neutral" if verdict == "kept" else verdict)
            pairs(ax, x, a, b, shown)
            ax.annotate(
                f"{pct(a, b):+.1f}%",
                (x + 0.14, max(b)),
                xytext=(10, -4),
                textcoords="offset points",
                va="top",
                fontsize=8.5,
                color=INK,
            )
        ax.set_xticks(range(len(RL_ABBA)), [r[0] for r in RL_ABBA])
        ax.set_xlim(-0.6, len(RL_ABBA) - 0.4)
        if key == "fb_ms":
            ax.set_yscale("log")
            ax.set_yticks([150, 200, 300, 500, 1000, 2000, 3000], ["150", "200", "300", "500", "1000", "2000", "3000"])
            ax.minorticks_off()
        if key == "wall_s":
            ax.text(
                0.99,
                0.96,
                "pinned: both candidates faster, but about 3.3 s of wall\ndifference vs at most "
                "about 0.35 s of trainer saving:\ninference-wait variance, not attributable",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=INK_2,
            )

    # Panel D: capability milestones (status, always icon + label).
    ax = fig.add_subplot(grid[3, :])
    ax.axis("off")
    ax.set_title(
        "D  Full-parameter training capability (beyond LoRA)", loc="left", fontsize=11, color=INK, fontweight="bold"
    )
    for i, (model, step, ok) in enumerate(MILESTONES):
        x = i / len(MILESTONES)
        color = VERDICT["kept"][0] if ok else VERDICT["slower"][0]
        ax.text(x, 0.55, "✓" if ok else "✗", fontsize=18, color=color, fontweight="bold", va="center")
        ax.text(x + 0.03, 0.62, model, fontsize=9.5, color=INK, fontweight="bold", va="center")
        ax.text(x + 0.03, 0.3, step, fontsize=8.5, color=INK_2, va="center")

    # Panel E: isolated CPU renderer setup PSS; this does not measure a full RL process tree.
    ax = fig.add_subplot(grid[4, :])
    style(ax, "E  Train-tokenizer renderer setup PSS, isolated CPU processes (lower is better)", "PSS (MiB)")
    renderer = renderer_runs(root)
    positions = [0, 1]
    for x, model in zip(positions, ("135", "360")):
        baseline, candidate = renderer[model]
        ax.bar(x - 0.16, baseline, width=0.28, color=MUTED, label="baseline" if x == 0 else None, zorder=3)
        ax.bar(x + 0.16, candidate, width=0.28, color=FRONTIER, label="candidate" if x == 0 else None, zorder=3)
        ax.annotate(f"{baseline:.0f}", (x - 0.16, baseline), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
        ax.annotate(f"{candidate:.0f}", (x + 0.16, candidate), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(positions, ["SmolLM2-135M\n−594 MiB", "SmolLM2-360M\n−595 MiB"])
    ax.set_xlim(-0.7, 2.8)
    ax.set_ylim(0, 940)
    ax.text(
        0.54,
        0.78,
        "4/4 semantic parity; 64 tasks, one-turn config; no Torch.\n"
        "Exp 56 full-RL: 0/2 updates; stopped at 116 s on\n"
        "the 2 GiB RAM floor. Isolated setup only; whole-run\n"
        "memory and speed impact remain unmeasured.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=INK_2,
    )
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=8, labelcolor=INK_2)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor=SURFACE,
            markeredgecolor=MUTED,
            markersize=8,
            markeredgewidth=1.6,
            label="control run",
        )
    ]
    for verdict, (color, label) in VERDICT.items():
        handles.append(
            Line2D(
                [],
                [],
                marker="X" if verdict == "invalid" else "o",
                linestyle="",
                color=color,
                markersize=8,
                label=label,
            )
        )
    handles.append(Line2D([], [], color=FRONTIER, linewidth=2, label="end-to-end record"))
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_2,
    )
    fig.savefig(output, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
