#!/usr/bin/env python3
"""Plot the tiny-prl hill climb as one slide: every measured run, the end-to-end record, and what never reached wall time."""

import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Rectangle

BG, INK, INK_2, DIM, FAINT, RULE = "#ffffff", "#141414", "#4a4844", "#9b988f", "#d6d4cd", "#ecebe6"
YELLOW, MINT, CORAL, SKY, LILAC = "#2a78d6", "#178a52", "#d64545", "#0e7c86", "#6250d6"
VERDICT = {"kept": MINT, "slower": CORAL, "neutral": DIM, "invalid": LILAC}

# (label, artifact, control arm, candidate arm, verdict, note) for Qwen3-0.6B LoRA 10-update RL A-B-B-As.
QWEN_ABBA = [
    ("inline\nenv", "tiny-opt-tokenizer-20261017/pool-comparison.json", "pool-control", "single-worker", "neutral", ""),
    ("4 infer\nseqs", "tiny-opt-tokenizer-20261017/rollout-comparison.json", "seqs2", "seqs4", "slower", ""),
    ("decode\nCUDA graphs", "tiny-opt-tokenizer-20261017/graph-comparison.json", "eager", "decode-graph", "kept", ""),
    (
        "act. offload\noff",
        "tiny-opt-tokenizer-20261017/activation-retest-comparison.json",
        "activation-offload-on",
        "activation-offload-off",
        "neutral",
        "",
    ),
    (
        "KV budget\n.25",
        "tiny-opt-tokenizer-20261017/kv-comparison.json",
        "kv-budget-35",
        "kv-budget-25",
        "neutral",
        "kept: -762 MiB GPU",
    ),
    ("AC\noff", "tiny-opt-tokenizer-20261017/ac-comparison.json", "checkpoint-full", "checkpoint-off", "neutral", ""),
    (
        "rollout\ncap 4",
        "tiny-opt-tokenizer-20261017/queue-comparison.json",
        "queue-adaptive",
        "queue-cap4",
        "neutral",
        "",
    ),
    ("CuTe\nhead", "tiny-kernel-20261018/comparison/summary.json", "a", "b", "neutral", ""),
    ("Triton\nhead", "tiny-kernel-20261018/triton-integration/abba-summary.json", "a", "b", "neutral", ""),
    (
        "CuTe fused\nhead",
        "tiny-kernel-20261018/cute-fused-integration/real-abba-20261019/analysis.json",
        "A",
        "B",
        "neutral",
        "",
    ),
]
QWEN_EXP = [4, 5, 6, 9, 10, 11, 12, 14, 18, 20]

# (label, source, tag, verdict): SmolLM2-135M full-parameter trainer fwd/bwd A-B-B-As.
TRAINER_ABBA_DIR = "tiny-kernel-20261020/triton/trainer-abba"
TRITON_BOARD = [
    ("RoPE · exp 50", "fake", f"{TRAINER_ABBA_DIR}/rope", "neutral"),
    ("SwiGLU · exp 50", "fake", f"{TRAINER_ABBA_DIR}/swiglu", "slower"),
    ("torch.compile · exp 53", "fake", f"{TRAINER_ABBA_DIR}/compile", "invalid"),
    ("RMSNorm · exp 52", "fake", f"{TRAINER_ABBA_DIR}/rmsnorm", "kept"),
    ("RMSNorm + RoPE · exp 52", "fake", f"{TRAINER_ABBA_DIR}/rope_rmsnorm", "kept"),
    ("+ residual fusion · exp 60", "fake", "tiny-kernel-20261020/fusion-profile/abba-dw-exact-r1", "slower"),
    ("exact-dW tile 32 · exp 61", "fake", "tiny-kernel-20261020/dw-tune/trainer-abba-r1", "neutral"),
    ("corrected RMSNorm + RoPE · exp 63", "fake", "tiny-kernel-20261020/dw-tune/corrected-vs-torch", "kept"),
    ("autotuned · exp 52", "rl", "triton", "slower"),
    ("shape-stable · exp 54", "rl", "tritonfix", "neutral"),
    ("pinned · exp 55", "rl", "pinned", "kept"),
]
FIXED_SHAPE_ROWS = 8
BOARD_NOTE = {
    "invalid": "fails strict parity: not a gain",
    "triton": "re-autotunes every RL batch shape",
    "tiny-kernel-20261020/fusion-profile/abba-dw-exact-r1": "vs pinned Triton, not Torch",
    "tiny-kernel-20261020/dw-tune/trainer-abba-r1": "vs tile 128, not Torch",
    "tritonfix": "start-up autotune; runs disagree",
}

# Every ledger experiment: (lane, outcome). Outcomes summarize each EXPERIMENTS.md entry's own verdict.
LANES = ["end-to-end RL", "trainer + kernels", "memory + capacity", "numerics + checkpoints"]
E2E, TRAIN, MEM, NUM = range(4)
OUTCOMES = {
    "kept": ("o", MINT, "kept: measured gain"),
    "component": ("o", YELLOW, "component-only win"),
    "rejected": ("o", CORAL, "rejected / regression"),
    "parity": ("X", LILAC, "fails strict parity"),
    "stop": ("^", "#c98500", "safe stop / blocked"),
    "validated": ("s", SKY, "validated pass"),
    "none": ("o", DIM, "diagnostic / no claim"),
}
EXPERIMENTS = {
    1: (MEM, "kept"),
    2: (MEM, "stop"),
    3: (MEM, "none"),
    4: (E2E, "none"),
    5: (E2E, "rejected"),
    6: (E2E, "kept"),
    7: (NUM, "parity"),
    8: (NUM, "parity"),
    9: (E2E, "none"),
    10: (E2E, "none"),
    11: (E2E, "none"),
    12: (E2E, "none"),
    13: (TRAIN, "none"),
    14: (E2E, "none"),
    15: (TRAIN, "component"),
    16: (NUM, "parity"),
    17: (TRAIN, "component"),
    18: (TRAIN, "component"),
    19: (MEM, "validated"),
    20: (E2E, "none"),
    21: (MEM, "kept"),
    22: (MEM, "rejected"),
    23: (TRAIN, "component"),
    24: (MEM, "none"),
    25: (MEM, "none"),
    26: (NUM, "validated"),
    27: (MEM, "stop"),
    28: (MEM, "rejected"),
    29: (MEM, "stop"),
    30: (MEM, "stop"),
    31: (MEM, "none"),
    32: (MEM, "none"),
    33: (MEM, "none"),
    34: (MEM, "validated"),
    35: (NUM, "parity"),
    36: (MEM, "stop"),
    37: (NUM, "validated"),
    38: (NUM, "validated"),
    39: (NUM, "validated"),
    40: (TRAIN, "component"),
    41: (MEM, "stop"),
    42: (MEM, "stop"),
    43: (NUM, "parity"),
    44: (TRAIN, "none"),
    45: (MEM, "stop"),
    46: (MEM, "stop"),
    47: (NUM, "validated"),
    48: (MEM, "validated"),
    49: (TRAIN, "none"),
    50: (TRAIN, "rejected"),
    51: (TRAIN, "none"),
    52: (TRAIN, "rejected"),
    53: (NUM, "parity"),
    54: (TRAIN, "rejected"),
    55: (TRAIN, "kept"),
    56: (MEM, "stop"),
    57: (MEM, "component"),
    58: (NUM, "parity"),
    59: (TRAIN, "rejected"),
    60: (NUM, "validated"),
    61: (TRAIN, "none"),
    62: (NUM, "validated"),
    63: (TRAIN, "kept"),
    64: (TRAIN, "component"),
    65: (NUM, "validated"),
    66: (E2E, "rejected"),
    67: (NUM, "none"),
    68: (NUM, "validated"),
    69: (TRAIN, "none"),
    70: (TRAIN, "component"),
    71: (MEM, "validated"),
    72: (MEM, "component"),
    73: (TRAIN, "component"),
    74: (TRAIN, "kept"),
    75: (TRAIN, "none"),
    76: (NUM, "validated"),
    77: (TRAIN, "component"),
    78: (TRAIN, "none"),
    79: (NUM, "validated"),
    80: (NUM, "validated"),
    81: (TRAIN, "component"),
    82: (TRAIN, "component"),
    83: (E2E, "component"),
    84: (TRAIN, "component"),
    85: (MEM, "validated"),
    86: (NUM, "validated"),
    87: (TRAIN, "none"),
    88: (MEM, "stop"),
    89: (TRAIN, "none"),
    90: (NUM, "validated"),
    91: (TRAIN, "stop"),
    92: (MEM, "validated"),
    93: (TRAIN, "rejected"),
    94: (MEM, "validated"),
    95: (TRAIN, "none"),
    96: (NUM, "validated"),
    97: (TRAIN, "kept"),
    98: (NUM, "validated"),
    99: (NUM, "validated"),
    100: (TRAIN, "component"),
    101: (TRAIN, "component"),
    102: (E2E, "none"),
    103: (MEM, "rejected"),
    104: (MEM, "rejected"),
    105: (NUM, "validated"),
    106: (TRAIN, "component"),
    107: (MEM, "rejected"),
    108: (NUM, "validated"),
    109: (MEM, "component"),
    110: (MEM, "component"),
    111: (TRAIN, "component"),
    112: (TRAIN, "component"),
    113: (NUM, "validated"),
    114: (NUM, "validated"),
    115: (TRAIN, "none"),
    116: (NUM, "parity"),
    117: (TRAIN, "none"),
    118: (NUM, "validated"),
}
# Callout text with (dx, dy) offsets in points, staggered so neighbours do not collide.
CALLOUTS = {
    1: ("import RSS -29%", (18, -16)),
    6: ("decode graphs -12.6% wall", (0, 15)),
    21: ("stream writer -192 MiB", (0, -16)),
    48: ("135M full-param ckpt + resume", (-40, -16)),
    55: ("RL fwd/bwd -15.6%", (-70, 15)),
    63: ("corrected kernels -9.4%", (-110, 45)),
    64: ("clip+AdamW -9.8% (component)", (44, 25)),
    79: ("vLLM CUDA IPC: exact tiny reload", (-77, 25)),
    83: ("IPC reload -43.8%\n(receiver component)", (-44, 35)),
    86: ("CuTe: 120 updates exact", (10, 18)),
    97: ("Qwen steps -9.5%\n(1 GiB GPU state)", (-35, 12)),
    100: ("unfused CuTe -64%\n(component)", (-80, 100)),
    101: ("Qwen CuTe -2.8%\n(no-swap trainer prototype)", (-40, 55)),
}

MILESTONES = [
    ("135M", "10 full-param updates", True),
    ("135M", "checkpoint + fresh resume", True),
    ("360M", "10 full-param updates", True),
    ("360M", "checkpoint save (host RAM)", False),
    ("Qwen3-0.6B", "5 full updates · seq 64", True),
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
    return [r["elapsed_seconds"] for r in rows if key(r) == control], [
        r["elapsed_seconds"] for r in rows if key(r) == candidate
    ]


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
        runs[name] = {
            "fb": statistics.median(fb[2:]),
            "wall": load(base / tag / name / "guard/result.json")["elapsed_seconds"],
        }
    return runs


def board_effect(root: Path, source: str, tag: str) -> tuple[float, bool]:
    if source == "fake":
        summary = load(root / tag / "summary.json")
        if "comparisons" in summary:
            fb = summary["comparisons"]["time/forward_backward"]
            return fb["B_vs_A_percent"], fb["both_B_runs_faster_than_both_A_runs"]
        timing = summary["timing"]
        return timing["time/forward_backward B_vs_A_pct"], timing["time/forward_backward both_B_beat_both_A"]
    runs = rl_runs(root, tag)
    a, b = [runs["A1"]["fb"], runs["A2"]["fb"]], [runs["B1"]["fb"], runs["B2"]["fb"]]
    return 100 * (statistics.mean(b) / statistics.mean(a) - 1), max(b) < min(a)


def pct(control: list[float], candidate: list[float]) -> float:
    return 100 * (statistics.mean(candidate) / statistics.mean(control) - 1)


def register_fonts() -> tuple[str, str]:
    font_dir = Path(os.environ.get("TINY_PRL_FONT_DIR", ""))
    if font_dir.is_dir():
        for path in list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf")):
            font_manager.fontManager.addfont(str(path))
    names = {font.name for font in font_manager.fontManager.ttflist}
    sans = "Lexend" if "Lexend" in names else "DejaVu Sans"
    mono = "IBM Plex Mono" if "IBM Plex Mono" in names else "DejaVu Sans Mono"
    return sans, mono


class Sketch:
    """Thin card and box strokes; optional wobble for a hand-drawn look, deterministic per call order."""

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)

    def line(self, ax, x0, y0, x1, y1, color, width=1.0, wobble=0.0, transform=None, passes=1, zorder=5):
        t = np.linspace(0, 1, 24)
        for _ in range(passes):
            jx, jy = self.rng.normal(0, wobble, 2)
            bow = self.rng.normal(0, wobble * 1.5) * np.sin(np.pi * t)
            xs = x0 + (x1 - x0) * t + jx + bow * (y1 - y0 != 0)
            ys = y0 + (y1 - y0) * t + jy + bow * (y1 - y0 == 0)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=width,
                solid_capstyle="round",
                transform=transform or ax.transAxes,
                zorder=zorder,
                alpha=0.95,
            )

    def box(self, ax, x, y, w, h, color, width=1.4, transform=None, zorder=5):
        for (a, b), (c, d) in [
            ((x, y), (x + w, y)),
            ((x + w, y), (x + w, y + h)),
            ((x + w, y + h), (x, y + h)),
            ((x, y + h), (x, y)),
        ]:
            self.line(ax, a, b, c, d, color, width, transform=transform, zorder=zorder)


def text_width(fig, text, **kw) -> float:
    artist = fig.text(0, 0, text, **kw)
    fig.canvas.draw()
    width = artist.get_window_extent().width / fig.bbox.width
    artist.remove()
    return width


def style_axes(ax, mono: str) -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
    ax.tick_params(colors=DIM, labelsize=10, length=0, pad=8)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(mono)
    ax.grid(axis="y", color=RULE, linewidth=0.9)
    ax.set_axisbelow(True)


def mark(fig, x, y, ok, size, color, **kw):
    fig.text(
        x,
        y,
        "\u2713" if ok else "\u2717",
        fontsize=size,
        color=color,
        fontfamily="DejaVu Sans",
        fontweight="bold",
        **kw,
    )


def eyebrow(fig, x, y, text, mono, color=YELLOW, size=11):
    fig.text(
        x,
        y,
        "  ".join(text.upper().split(" ")),
        fontsize=size,
        color=color,
        fontfamily=mono,
        fontweight="medium",
    )


def draw_index(fig, sans: str, mono: str) -> None:
    eyebrow(fig, 0.06, 0.92, f"e · every experiment · 1-{max(EXPERIMENTS)}", mono)
    fig.text(0.06, 0.82, "The whole climb, one marker per ledger entry", fontsize=22, color=INK, fontweight="semibold")
    ax = fig.add_axes([0.2, 0.3, 0.76, 0.44])
    style_axes(ax, mono)
    ax.grid(axis="x", color=RULE, linewidth=0.9)
    for number, (lane, outcome) in EXPERIMENTS.items():
        marker, color, _ = OUTCOMES[outcome]
        size = 90 if outcome in ("kept", "component") else 55
        ax.scatter(number, lane, marker=marker, s=size, color=color, edgecolors=BG, linewidths=0.8, zorder=3)
    for number, (text, offset) in CALLOUTS.items():
        lane = EXPERIMENTS[number][0]
        ax.annotate(
            text,
            (number, lane),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontfamily=mono,
            color=INK,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": BG, "edgecolor": "none"},
            zorder=5,
        )
    ax.set_yticks(range(len(LANES)), LANES)
    ax.set_ylim(len(LANES) - 0.45, -0.6)
    ax.set_xticks(range(0, max(EXPERIMENTS) + 1, 5))
    ax.set_xlim(0, max(EXPERIMENTS) + 1)
    ax.set_xlabel("experiment number (chronological)", color=DIM, fontsize=10, fontfamily=mono)
    for label in ax.get_yticklabels():
        label.set_fontfamily(sans)
        label.set_color(INK)
        label.set_fontsize(11)
    for i, (marker, color, label) in enumerate(OUTCOMES.values()):
        x = 0.05 + i * 0.137
        fig.add_artist(
            plt.Line2D(
                [x], [0.12], marker=marker, markersize=8, color=color, linestyle="", transform=fig.transSubfigure
            )
        )
        fig.text(x + 0.01, 0.12, label, fontsize=9.5, color=INK_2, fontfamily=mono, va="center")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_tiny_hillclimb.py OUTPUTS_DIR OUTPUT.png")
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    sans, mono = register_fonts()
    plt.rcParams.update({"font.family": sans, "axes.unicode_minus": False})
    sketch = Sketch()

    outer = plt.figure(figsize=(16, 27.5), facecolor=BG)
    fig, index_fig = outer.subfigures(2, 1, height_ratios=[22, 5.5], facecolor=BG)

    # Header.
    eyebrow(fig, 0.06, 0.957, "tiny-prl · hill climb · rtx 4060 laptop", mono)
    title_kw = {"fontsize": 44, "fontweight": "semibold", "fontfamily": sans}
    lead = "One RL win: "
    accent = "12.6% off RL wall time"
    fig.text(0.06, 0.918, lead, color=INK, **title_kw)
    lead_w = text_width(fig, lead, **title_kw)
    fig.text(0.06 + lead_w, 0.918, accent, color=YELLOW, **title_kw)
    fig.text(
        0.06,
        0.889,
        "Every dot is one guarded run under a 2 GiB RAM floor. A change is kept only if both of its "
        "runs beat both controls (A-B-B-A) and it passes strict parity.",
        fontsize=13,
        color=INK_2,
        fontfamily=sans,
    )

    # Hero cards.
    cards = [
        ("-12.6%", MINT, "Qwen3-0.6B LoRA · RL wall time", "decode CUDA graphs", "exp 6 · only e2e win"),
        ("-15.6%", SKY, "SmolLM2-135M · RL trainer fwd/bwd", "Triton RMSNorm+RoPE", "exp 55 · wall: no claim"),
        ("10/10", YELLOW, "SmolLM2-135M · full-parameter RL", "ckpt + fresh resume", "exp 48 · beyond LoRA"),
    ]
    hero = fig.add_axes([0.06, 0.79, 0.88, 0.085], zorder=2)
    hero.axis("off")
    for i, (value, color, label, source, verdict) in enumerate(cards):
        x = i / 3 + 0.005
        sketch.box(hero, x, 0.04, 1 / 3 - 0.025, 0.92, FAINT if i else color, width=1.3, transform=hero.transAxes)
        hero.text(
            x + 0.016,
            0.38,
            value,
            fontsize=34,
            fontweight="semibold",
            color=color,
            va="center",
            transform=hero.transAxes,
        )
        hero.text(
            x + 0.016, 0.8, label, fontsize=10.5, color=INK, fontweight="medium", va="center", transform=hero.transAxes
        )
        hero.text(
            x + 0.148, 0.5, source, fontsize=9, color=INK_2, fontfamily=mono, va="center", transform=hero.transAxes
        )
        hero.text(
            x + 0.148, 0.26, verdict, fontsize=9, color=color, fontfamily=mono, va="center", transform=hero.transAxes
        )

    # Panel A: Qwen LoRA end-to-end record.
    eyebrow(fig, 0.06, 0.757, "a · qwen3-0.6b lora · 10-update rl wall time", mono)
    fig.text(0.06, 0.735, "Verified RL wall-time gain: 12.6%", fontsize=22, color=INK, fontweight="semibold")
    ax = fig.add_axes([0.085, 0.555, 0.875, 0.165])
    style_axes(ax, mono)
    record = None
    for x, (label, artifact, control, candidate, verdict, note) in enumerate(QWEN_ABBA):
        a, b = qwen_runs(root, artifact, control, candidate)
        ax.scatter([x - 0.13] * len(a), a, s=60, facecolors=BG, edgecolors=DIM, linewidths=1.6, zorder=3)
        ax.scatter([x + 0.13] * len(b), b, s=70, color=VERDICT[verdict], edgecolors=BG, linewidths=1.2, zorder=4)
        if verdict == "kept":
            record = (x, statistics.mean(a), statistics.mean(b))
        ax.annotate(
            f"{pct(a, b):+.1f}%",
            (x + 0.13, max(b)),
            xytext=(-30 if verdict == "kept" else 0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontfamily=mono,
            color=VERDICT[verdict] if verdict != "neutral" else INK_2,
            fontweight="medium",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": BG, "edgecolor": "none"},
        )
        if note:
            ax.annotate(
                note,
                (x, min(a + b)),
                xytext=(0, -22),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                fontfamily=mono,
                color=DIM,
            )
    x0, before, after = record
    end = len(QWEN_ABBA) - 0.5
    ax.plot([-0.5, x0 + 0.13, x0 + 0.13], [before, before, after], color=YELLOW, linewidth=2.4, zorder=2)
    ax.plot([x0 + 0.13, end], [after, after], color=YELLOW, linewidth=1.6, linestyle=(0, (5, 4)), zorder=2, alpha=0.8)
    ax.annotate(
        f"record  {before:.1f} s  ->  {after:.1f} s",
        (x0 + 0.13, (before + after) / 2),
        xytext=(14, 0),
        textcoords="offset points",
        fontsize=11,
        fontfamily=mono,
        color=YELLOW,
        va="center",
    )
    ax.axvspan(6.5, end, color="#f6f5f1", zorder=0)
    ax.text(
        8,
        113.5,
        "LM-head kernels: 17-40% faster in isolation,\nnone moved wall time",
        ha="center",
        va="top",
        fontsize=9.5,
        fontfamily=mono,
        color=INK_2,
    )
    ax.set_xticks(range(len(QWEN_ABBA)), [f"exp {e}\n{q[0]}" for e, q in zip(QWEN_EXP, QWEN_ABBA)], linespacing=1.4)
    ax.set_xlim(-0.6, len(QWEN_ABBA) - 0.4)
    ax.set_ylim(83, 117)
    ax.set_ylabel("seconds", color=DIM, fontsize=10, fontfamily=mono)

    # Panel B: the Triton push as a diverging verdict board.
    eyebrow(fig, 0.06, 0.49, "b · triton push · smollm2-135m full-parameter trainer", mono)
    fig.text(0.06, 0.468, "Kernels made the trainer faster ...", fontsize=22, color=INK, fontweight="semibold")
    fig.text(
        0.06,
        0.451,
        "fwd/bwd change vs Torch, A-B-B-A. Fixed-shape fake data first, then real RL batches. "
        "The step is launch-bound: GPU 30% busy, 13k kernels per step.",
        fontsize=11.5,
        color=INK_2,
    )
    ax = fig.add_axes([0.31, 0.255, 0.65, 0.18])
    ax.set_facecolor(BG)
    ax.axis("off")
    clip = 22
    rows = len(TRITON_BOARD)
    for i, (label, source, tag, verdict) in enumerate(TRITON_BOARD):
        y = rows - 1 - i
        value, both = board_effect(root, source, tag)
        color = VERDICT[verdict]
        shown = max(-clip * 2.4, min(value, clip))
        ax.add_patch(Rectangle((0, y - 0.3), shown, 0.6, facecolor=color, edgecolor="none", alpha=0.85, zorder=3))
        side = 1 if value > 0 else -1
        tail = f"{value:+.1f}%" + (" ->" if value > clip else "")
        ax.text(
            shown + side * 1.2,
            y,
            tail,
            ha="left" if side > 0 else "right",
            va="center",
            fontsize=11,
            fontfamily=mono,
            color=color if verdict != "neutral" else INK_2,
            fontweight="medium",
        )
        note = BOARD_NOTE.get(tag) or BOARD_NOTE.get(verdict, "")
        if note:
            ax.text(
                -1.5 if value > 0 else 1.5,
                y,
                note,
                ha="right" if value > 0 else "left",
                va="center",
                fontsize=9,
                fontfamily=mono,
                color=DIM,
            )
        fy = 0.255 + 0.18 * (y + 0.6) / (rows + 0.4)
        word = {"kept": "kept", "slower": "rejected", "neutral": "no claim", "invalid": "invalid"}[verdict]
        if verdict == "neutral":
            fig.text(0.06, fy, "\u2013", fontsize=11, color=color, fontfamily=mono, va="center")
        else:
            mark(fig, 0.06, fy, verdict == "kept", 11, color, va="center")
        fig.text(0.078, fy, word, fontsize=10, color=color, fontfamily=mono, va="center")
        fig.text(0.295, fy, label, ha="right", va="center", fontsize=11.5, color=INK)
        if i == FIXED_SHAPE_ROWS - 1:
            ax.plot([-clip * 2.6, clip + 14], [y - 0.62, y - 0.62], color=FAINT, linewidth=1, linestyle=(0, (2, 3)))
    ax.text(-clip * 2.55, rows - 0.35, "fixed-shape fake data", fontsize=9, fontfamily=mono, color=DIM, va="bottom")
    ax.text(
        -clip * 2.55,
        rows - FIXED_SHAPE_ROWS - 0.62,
        "real RL batches (RMSNorm + RoPE)",
        fontsize=9,
        fontfamily=mono,
        color=DIM,
        va="bottom",
    )
    ax.axvline(0, color=INK_2, linewidth=1.2, zorder=5)
    ax.set_xlim(-clip * 2.6, clip + 14)
    ax.set_ylim(-0.6, rows - 0.2)

    # Panel C: the same RL runs, wall time.
    eyebrow(fig, 0.06, 0.212, "c · same rl runs · 10-update wall time", mono)
    fig.text(0.06, 0.19, "... but RL wall time is inference-bound", fontsize=22, color=INK, fontweight="semibold")
    ax = fig.add_axes([0.085, 0.09, 0.5, 0.085])
    style_axes(ax, mono)
    for x, (label, tag) in enumerate([("autotuned", "triton"), ("shape-stable", "tritonfix"), ("pinned", "pinned")]):
        runs = rl_runs(root, tag)
        a, b = [runs["A1"]["wall"], runs["A2"]["wall"]], [runs["B1"]["wall"], runs["B2"]["wall"]]
        color = CORAL if pct(a, b) > 5 else DIM
        ax.scatter([x - 0.13] * 2, a, s=60, facecolors=BG, edgecolors=DIM, linewidths=1.6, zorder=3)
        ax.scatter([x + 0.13] * 2, b, s=70, color=color, edgecolors=BG, linewidths=1.2, zorder=4)
        ax.annotate(
            f"{pct(a, b):+.1f}%",
            (x + 0.13, max(b)),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontfamily=mono,
            color=color if color != DIM else INK_2,
        )
    ax.set_xticks(range(3), ["autotuned", "shape-stable", "pinned"])
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylabel("seconds", color=DIM, fontsize=10, fontfamily=mono)
    fig.text(
        0.62,
        0.17,
        "Pinned kernels saved about 0.35 s of trainer time\nper run. Both candidate runs were "
        "3.3 s faster,\nbut that tracks inference-wait variance\n(33-41 s per run), so no end-to-end claim.",
        fontsize=11,
        color=INK_2,
        va="top",
        linespacing=1.5,
    )

    # Panel D: capability chips.
    eyebrow(fig, 0.06, 0.05, "d · full-parameter training beyond lora", mono)
    for i, (model, step, ok) in enumerate(MILESTONES):
        x, y = 0.06 + i * 0.178, 0.03
        color = MINT if ok else CORAL
        mark(fig, x, y, ok, 12, color)
        fig.text(x + 0.018, y, model, fontsize=11, color=color, fontfamily=mono, fontweight="medium")
        fig.text(x + 0.018, y - 0.01, step, fontsize=9, color=INK_2, fontfamily=mono)

    # Legend and footer.
    legend = [
        (BG, DIM, "control run"),
        (MINT, BG, "kept"),
        (CORAL, BG, "rejected"),
        (DIM, BG, "no claim"),
        (LILAC, BG, "fails parity"),
    ]
    for i, (face, edge, label) in enumerate(legend):
        lx = 0.42 + i * 0.09
        fig.add_artist(
            plt.Line2D(
                [lx],
                [0.741],
                marker="o",
                markersize=8,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=1.6,
                linestyle="",
                transform=fig.transSubfigure,
            )
        )
        fig.text(lx + 0.01, 0.741, label, fontsize=10, color=INK_2, fontfamily=mono, va="center")
    fig.add_artist(plt.Line2D([0.42, 0.45], [0.727, 0.727], color=YELLOW, linewidth=2.4, transform=fig.transSubfigure))
    fig.text(
        0.46,
        0.727,
        "end-to-end record (dashed: holds; later controls drift with host load)",
        fontsize=10,
        color=INK_2,
        fontfamily=mono,
        va="center",
    )
    draw_index(index_fig, sans, mono)
    eyebrow(index_fig, 0.06, 0.025, "pradheep · oct 2026", mono, color=DIM, size=9.5)
    index_fig.text(
        0.94, 0.025, "evidence: examples/tiny/EXPERIMENTS.md", fontsize=9.5, color=DIM, fontfamily=mono, ha="right"
    )
    outer.savefig(output, dpi=120, facecolor=BG)


if __name__ == "__main__":
    main()
