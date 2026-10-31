#!/usr/bin/env python3
"""Render the tiny-prl hill-climb poster (examples/tiny/hillclimb.png).

Every number is embedded from examples/tiny/EXPERIMENTS.md (Experiments 1-180);
the script reads no other file. The first positional argument is accepted for
CLI compatibility and is not used.

    TINY_PRL_FONT_DIR=<dir with Lexend and IBM Plex Mono> MPLBACKEND=Agg \
        uv run --isolated --no-project --python 3.12 --with matplotlib \
        tools/plot_tiny_hillclimb.py outputs examples/tiny/hillclimb.png
"""

import argparse
import os
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

PAGE_W = 9.6  # inches; 1920 px at 200 dpi
PAGE_H = 20.95  # inches; about 4150 px at 200 dpi
DPI = 200
GUTTER = 0.5

ACCENT = "#0F766E"
ACCENT_TINT = "#A7DBD5"
INK = "#1F2328"
SUB = "#57606A"
MUTED = "#8C959F"
PALE = "#D5DAE0"
RULE = "#D8DEE4"
CARD = "#F4F6F8"

PROSE = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

# ---------------------------------------------------------------- ledger data
# Experiment 126: instrumented 135M run, before the Exp 137 cleanup fix (seconds).
RUN_126 = {"startup": 34.4, "loop": 35.0, "shutdown": 17.4}
# Exp 137 B-arm mean after the fix (full process wall); Exp 177 Torch median.
AFTER_FIX_137 = 62.8
AFTER_FIX_177 = 60.4
# Exp 180 fork-server B-arm mean (full process wall) and its A-arm mean.
FORK_180 = 41.45
FORK_180_CONTROL = 53.97

# Exp 128: decode inter-token latency inside vLLM (ms). Percent labels are the
# ledger's own. B2 at 7000 is a range: 3.34-3.86 ms stock.
CONTEXTS = [128, 1024, 4096, 7000]
ITL_STOCK = {"B1": [1.94, 1.97, 2.39, 2.57], "B2": [2.05, 2.29, 2.88, (3.34, 3.86)]}
ITL_KERNEL = {"B1": [1.57, 1.68, 2.02, 2.33], "B2": [1.62, 1.82, 2.47, 3.10]}
ITL_PCT = {
    "B1": ["-19%", "-15%", "-16%", "-9%"],
    "B2": ["-21%", "-20%", "-14%", "-7 to -20%"],
}

# Exp 131: R6 vs R5 on actual vLLM, percent change, and whether both candidates
# beat both controls (only the 2048 B1 case does not).
R6_VS_R5 = [
    ("32 B1", -10.90, True),
    ("32 B2", -6.80, True),
    ("2048 B1", +0.96, False),
    ("2048 B2", -1.21, True),
    ("8000 B1", -2.96, True),
    ("8000 B2", -3.31, True),
]

# Kernel and trainer components: (name, experiment, percent change, tag, RL note).
# Percent is negative when faster. Tags: V verified, C component-only, X rejected.
KERNEL_ROWS = [
    ("RoPE + RMSNorm, trainer step", "Exp 63", -8.70, "V", "RL not run"),
    ("Fused clip + AdamW, trainer step", "Exp 74", -2.34, "V", "RL not run"),
    ("CuTe AdamW, Qwen3-0.6B trainer", "Exp 106", -2.88, "V", "fake batch 1, seq 64"),
    ("RoPE + RMSNorm, trainer fwd/bwd", "Exp 55", -15.6, "C", "RL wall -3.6%, not attributable"),
    ("RoPE + RMSNorm, trainer fwd/bwd", "Exp 177", -7.1, "C", "RL wall -0.9%, 0 of 2 blocks"),
    ("CuTe AdamW, SmolLM2-135M trainer", "Exp 89", -0.47, "C", "overlapping; not accepted"),
    ("RoPE + RMSNorm, Qwen LoRA trainer", "Exp 175", -1.0, "X", "not kept; RL not run"),
    ("Residual add + RMSNorm fusion", "Exp 59", +1.10, "X", "both candidates slower"),
    ("Triton SwiGLU, trainer", "Exp 50", +7.6, "X", "host-overhead regression"),
    ("torch.compile (Inductor), trainer", "Exp 51", -47.2, "X", "fails strict parity, Exp 53"),
    ("Triton LM head, 135M", "Exp 176", None, "X", "fails gradient parity; not timed"),
]

# SmolLM2-360M decoder revisions, direct CUDA-event medians at context 32 (ms).
# Columns: name, experiments, parity (narrow · wide, or total), B1, B2, change.
REVISIONS = [
    ("R9", "Exp 155-156", "34/34 · 5/12", None, None, "first 360M kernel; wide set fails"),
    ("R13", "Exp 159", "34/34 · 6/12", 5.28, 8.47, "B1 four-load lookahead, -68.6% vs R12"),
    ("R15b", "Exp 161, 170", "43/46", 16.88, 8.46, "attention PV task map and QK WMMA"),
    ("R20", "Exp 170", "43/46", 4.57, 7.96, "R15b + B1 eight-load + B2 cp.async"),
    ("R22", "Exp 172", "45/46", 3.65, None, "B1 projection tails, -20.1% vs R21"),
    ("R23", "Exp 173", "45/46", None, None, "B2 A-tile zero fill, -0.57% vs R22"),
    ("R25", "Exp 174", "46/46", None, None, "B2 long-context PV, -7.9% at 2048"),
]
REVISION_B2_TEXT = {"R22": "neutral vs R21", "R23": "-0.57% vs R22", "R25": "-7.9% at 2048"}
REVISION_B1_TEXT = {"R9": "no direct decoder timing", "R23": "flat vs R22", "R25": "neutral"}

# Serving pilots (ms, in-process vLLM, scheduler-inclusive, prompt 32): (stock, candidate).
SERVING = {
    "R9 (Exp 158)": ((4.21, 3.95), (17.41, 9.16)),
    "R13 (Exp 164)": ((4.20, 3.94), (5.90, 9.16)),
    "R22 (R22_SERVING_PILOT.json, serial pilot)": ((4.20, 3.93), (4.24, 8.58)),
}

# Experiment index. Outcome codes and lanes for all 180 experiments.
OUTCOMES = {
    "V": "verified (kept)",
    "C": "component-only",
    "A": "validated",
    "X": "rejected / regression",
    "F": "fails parity",
    "B": "blocked / safe stop",
    "D": "diagnostic",
}
CODES = {
    "V": [6, 63, 74, 97, 101, 106, 128, 129, 131, 134, 137, 180],
    "C": [
        1,
        4,
        10,
        14,
        15,
        17,
        18,
        20,
        21,
        23,
        40,
        51,
        55,
        61,
        64,
        65,
        72,
        73,
        77,
        81,
        82,
        83,
        84,
        100,
        109,
        110,
        111,
        112,
        130,
        148,
        159,
        163,
        166,
        168,
        170,
        172,
        173,
        177,
        179,
    ],
    "A": [
        25,
        34,
        37,
        38,
        39,
        47,
        48,
        57,
        60,
        62,
        68,
        71,
        76,
        79,
        80,
        85,
        86,
        90,
        92,
        94,
        96,
        98,
        99,
        105,
        108,
        113,
        114,
        118,
        119,
        123,
        127,
        139,
        144,
        145,
        147,
        149,
        150,
        155,
        157,
        174,
    ],
    "X": [
        5,
        9,
        11,
        12,
        28,
        50,
        52,
        54,
        59,
        66,
        69,
        70,
        75,
        78,
        87,
        89,
        93,
        95,
        103,
        104,
        107,
        132,
        133,
        135,
        146,
        158,
        164,
        175,
        178,
    ],
    "F": [
        7,
        8,
        16,
        35,
        43,
        53,
        116,
        120,
        121,
        122,
        124,
        136,
        138,
        140,
        141,
        142,
        143,
        151,
        152,
        153,
        154,
        156,
        160,
        161,
        167,
        171,
        176,
    ],
    "B": [2, 19, 22, 27, 29, 30, 36, 41, 42, 45, 46, 56, 88, 91],
    "D": [3, 13, 24, 26, 31, 32, 33, 44, 49, 58, 67, 102, 115, 117, 125, 126, 162, 165, 169],
}
LANES = [
    (
        "R",
        "RL runtime,\nmemory, IPC",
        [
            1,
            2,
            3,
            4,
            5,
            6,
            10,
            12,
            16,
            21,
            44,
            56,
            57,
            66,
            71,
            72,
            76,
            79,
            80,
            83,
            90,
            96,
            99,
            102,
            108,
            111,
            112,
            126,
            133,
            137,
            145,
            146,
            147,
            148,
            149,
            150,
            178,
            180,
        ],
    ),
    (
        "T",
        "trainer kernels,\noptimizer",
        [
            7,
            8,
            9,
            11,
            13,
            14,
            15,
            17,
            18,
            20,
            23,
            35,
            37,
            38,
            40,
            43,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            68,
            69,
            70,
            73,
            74,
            75,
            77,
            78,
            81,
            82,
            84,
            86,
            87,
            89,
            91,
            93,
            95,
            98,
            175,
            176,
            177,
        ],
    ),
    (
        "F",
        "full-parameter\ncapacity, checkpoints",
        [
            19,
            22,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            39,
            41,
            42,
            45,
            46,
            47,
            48,
            85,
            88,
            92,
            94,
            97,
            100,
            101,
            103,
            104,
            105,
            106,
            107,
            109,
            110,
            113,
        ],
    ),
    (
        "M",
        "decode megakernel",
        [
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            127,
            128,
            129,
            130,
            131,
            132,
            134,
            135,
            136,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            179,
        ],
    ),
]
# Callouts: (experiment, lane, text, x offset in experiments, y offset in lanes, alignment).
CALLOUTS = [
    (6, "R", "6  CUDA graphs, -12.6%", 1.5, 0.55, "left"),
    (137, "R", "137  cleanup, -12.2%", -1.5, 0.55, "right"),
    (180, "R", "180  fork server, -23.2%", -1.5, -0.6, "right"),
    (19, "F", "19  host-RAM floor", 1.5, -0.6, "left"),
    (36, "F", "36  10 updates, checkpoint OOM", 1.5, 0.55, "left"),
    (94, "F", "94  Qwen bounded slices", 1.5, 0.55, "left"),
    (53, "T", "53  Inductor fails parity", 1.5, 0.55, "left"),
    (128, "M", "128  megakernel in vLLM", -1.5, 0.55, "right"),
    (174, "M", "174  46/46 fixtures", -1.5, 0.55, "right"),
]

STYLE = {
    "V": dict(marker="o", facecolor=ACCENT, edgecolor=ACCENT, linewidths=0.4),
    "C": dict(marker="o", facecolor="white", edgecolor=ACCENT, linewidths=1.0),
    "A": dict(marker="o", facecolor=ACCENT_TINT, edgecolor=ACCENT, linewidths=0.5),
    "X": dict(marker="x", facecolor=INK, linewidths=1.0),
    "F": dict(marker="^", facecolor="white", edgecolor=INK, linewidths=0.9),
    "B": dict(marker="s", facecolor=PALE, edgecolor=MUTED, linewidths=0.5),
    "D": dict(marker="o", facecolor=MUTED, edgecolor=MUTED, linewidths=0.3),
}


def register_fonts() -> tuple[str, str]:
    font_dir = Path(os.environ.get("TINY_PRL_FONT_DIR", ""))
    if font_dir.is_dir():
        for path in list(font_dir.glob("*.ttf")) + list(font_dir.glob("*.otf")):
            font_manager.fontManager.addfont(str(path))
    names = {font.name for font in font_manager.fontManager.ttflist}
    prose = "Lexend" if "Lexend" in names else "DejaVu Sans"
    mono = "IBM Plex Mono" if "IBM Plex Mono" in names else "DejaVu Sans Mono"
    return prose, mono


def check_index() -> None:
    codes_of = {e: code for code, exps in CODES.items() for e in exps}
    lanes_of = {e: key for key, _, exps in LANES for e in exps}
    assert sorted(codes_of) == list(range(1, 181)), "every experiment needs exactly one outcome"
    assert sorted(lanes_of) == list(range(1, 181)), "every experiment needs exactly one lane"


class Page:
    def __init__(self, fig, prose: str, mono: str):
        self.fig = fig
        self.prose = prose
        self.mono = mono

    def text(self, x, y, s, size=8.5, color=INK, mono=False, weight="normal", ha="left", va="top", **kw):
        family = self.mono if mono else self.prose
        return self.fig.text(
            x / PAGE_W,
            1 - y / PAGE_H,
            s,
            fontsize=size,
            color=color,
            fontfamily=family,
            fontweight=weight,
            ha=ha,
            va=va,
            **kw,
        )

    def axes(self, x, y, w, h):
        return self.fig.add_axes([x / PAGE_W, 1 - (y + h) / PAGE_H, w / PAGE_W, h / PAGE_H])

    def card(self, x, y, w, h, big, label, body):
        ax = self.axes(x, y, w, h)
        ax.set_facecolor(CARD)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.07, 0.92, big, transform=ax.transAxes, fontsize=19, color=ACCENT, fontfamily=self.mono, va="top")
        ax.text(
            0.07,
            0.62,
            label,
            transform=ax.transAxes,
            fontsize=8.2,
            color=INK,
            fontfamily=self.prose,
            fontweight="normal",
            va="top",
        )
        ax.text(
            0.07,
            0.50,
            fill(body, 33),
            transform=ax.transAxes,
            fontsize=7.4,
            color=SUB,
            fontfamily=self.prose,
            va="top",
            linespacing=1.35,
        )

    def section(self, y, num, title, sub):
        self.text(GUTTER, y, num, size=8.5, color=ACCENT, mono=True, weight="normal")
        self.text(GUTTER + 0.42, y - 0.03, title, size=13.5, color=INK, weight="normal")
        self.text(GUTTER + 0.42, y + 0.3, sub, size=7.8, color=SUB)


def clean_axes(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=0, labelsize=7.3, colors=SUB, pad=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(MONO)


def draw_run_bars(pg):
    ax = pg.axes(2.35, 4.45, PAGE_W - GUTTER - 2.35, 1.45)
    ax.set_xlim(0, 95)
    ax.set_ylim(-1.55, 1.6)
    startup, loop, shutdown = RUN_126["startup"], RUN_126["loop"], RUN_126["shutdown"]
    seam = dict(edgecolor="white", linewidth=1.5)
    ax.barh(1, startup, left=0, height=0.5, color=PALE, **seam)
    ax.barh(1, loop, left=startup, height=0.5, color=ACCENT_TINT, **seam)
    ax.barh(1, shutdown, left=startup + loop, height=0.5, color=PALE, **seam)
    ax.barh(0, AFTER_FIX_137, left=0, height=0.5, color=ACCENT, **seam)
    ax.barh(-1, FORK_180, left=0, height=0.5, color=ACCENT, **seam)
    ax.text(
        FORK_180 / 2, -1, f"{FORK_180:.1f} s", ha="center", va="center", fontsize=8.6, color="white", fontfamily=pg.mono
    )
    ax.text(
        FORK_180 + 1.2,
        -1,
        f"vs {FORK_180_CONTROL:.1f} s control mean, -23.2%",
        ha="left",
        va="center",
        fontsize=7.3,
        color=SUB,
        fontfamily=pg.prose,
    )
    for left, width, name, value in [
        (0, startup, "startup", startup),
        (startup, loop, "rollout loop", loop),
        (startup + loop, shutdown, "shutdown", shutdown),
    ]:
        ax.text(left + width / 2, 1.36, name, ha="center", va="bottom", fontsize=7.4, color=SUB, fontfamily=pg.prose)
        ax.text(
            left + width / 2, 1, f"{value:.1f} s", ha="center", va="center", fontsize=8.3, color=INK, fontfamily=pg.mono
        )
    ax.text(
        AFTER_FIX_137 / 2,
        0,
        f"{AFTER_FIX_137:.1f} s",
        ha="center",
        va="center",
        fontsize=8.6,
        color="white",
        fontfamily=pg.mono,
        fontweight="normal",
    )
    ax.text(
        AFTER_FIX_137 + 1.2,
        0,
        f"Exp 177 median {AFTER_FIX_177:.1f} s",
        ha="left",
        va="center",
        fontsize=7.3,
        color=SUB,
        fontfamily=pg.prose,
    )
    ax.set_yticks([1, 0, -1])
    ax.set_yticklabels(
        ["Exp 126 run\nbefore cleanup fix", "Exp 137 B arms\nafter cleanup fix", "Exp 180 B arms\nwarm fork server"],
        fontsize=7.6,
        color=INK,
        fontfamily=pg.prose,
        linespacing=1.3,
    )
    ax.set_xticks(range(0, 91, 10))
    ax.set_xlabel("full process wall time, seconds", fontsize=7.2, color=SUB, fontfamily=pg.prose, labelpad=2)
    clean_axes(ax)
    ax.tick_params(axis="y", length=0)
    pg.text(
        GUTTER,
        6.35,
        fill(
            "Rollout loop (Exp 126): GPU busy 18.4% on average and zero in 63% of samples. Each rollout averaged "
            "1.33 s: 0.86 s harness, 0.18 s model, 0.29 s setup. Shutdown took 17.4 s. Exp 137 cut cleanup from "
            "10.8 s to 0.9 s and wall from 71.5 s to 62.8 s. Exp 126 and Exp 137 are separate runs.",
            165,
        ),
        size=7.3,
        color=SUB,
    )


def draw_kernel_rows(pg, top):
    row_h = 0.25
    n = len(KERNEL_ROWS)
    ax = pg.axes(3.55, top, 3.45, n * row_h)
    ax.set_xlim(-62, 14)
    ax.set_ylim(n - 0.5, -0.5)
    ax.axvline(0, color=INK, linewidth=0.6)
    for i, (name, exp, value, tag, note) in enumerate(KERNEL_ROWS):
        y_row = top + i * row_h
        code_label = {"V": "VERIFIED", "C": "COMPONENT", "X": "REJECTED"}[tag]
        pg.text(GUTTER, y_row, name, size=7.7, color=INK)
        pg.text(GUTTER, y_row + 0.12, f"{code_label} · {exp}", size=6.3, color=SUB, mono=True)
        pg.text(7.2, y_row + 0.02, note, size=6.8, color=SUB)
        if value is None:
            ax.text(
                -30,
                i,
                "no timing: parity fails",
                fontsize=6.8,
                color=SUB,
                va="center",
                fontfamily=pg.prose,
                style="italic",
            )
            continue
        ax.hlines(i, 0, value, color=RULE, linewidth=1.2, zorder=1)
        ax.scatter([value], [i], s=24, zorder=3, **STYLE[tag])
        offset = -1.2 if value < 0 else 1.2
        ax.text(
            value + offset,
            i,
            f"{value:+.1f}%",
            va="center",
            ha="right" if value < 0 else "left",
            fontsize=6.9,
            color=INK,
            fontfamily=pg.mono,
        )
    ax.set_yticks([])
    ax.set_xticks([-50, -25, 0, 10])
    ax.set_xticklabels(["-50%", "-25%", "0", "+10%"])
    ax.set_xlabel(
        "change in timed step, percent (negative is faster)", fontsize=7.1, color=SUB, fontfamily=pg.prose, labelpad=2
    )
    clean_axes(ax)
    ax.spines["bottom"].set_visible(True)


def draw_decode_itl(pg, key, x, title):
    ax = pg.axes(x, 12.1, 2.45, 1.45)
    stock = ITL_STOCK[key]
    kernel = ITL_KERNEL[key]
    for i, value in enumerate(stock):
        low = value[0] if isinstance(value, tuple) else value
        ax.plot([i, i], [kernel[i], low], color=RULE, linewidth=1.2, zorder=1)
        if isinstance(value, tuple):
            ax.vlines(i, value[0], value[1], color=MUTED, linewidth=2.2, zorder=2)
            for end in value:
                ax.scatter([i], [end], s=18, facecolor="white", edgecolor=MUTED, linewidths=0.9, zorder=3)
        else:
            ax.scatter([i], [value], s=20, facecolor="white", edgecolor=MUTED, linewidths=0.9, zorder=3)
        ax.scatter([i], [kernel[i]], s=24, facecolor=ACCENT, edgecolor=ACCENT, zorder=4)
        ax.text(
            i, kernel[i] - 0.09, ITL_PCT[key][i], ha="center", va="top", fontsize=6.4, color=INK, fontfamily=pg.mono
        )
    ax.set_xlim(-0.45, len(CONTEXTS) - 0.55)
    ax.set_ylim(1.1, 4.0)
    ax.set_xticks(range(len(CONTEXTS)))
    ax.set_xticklabels([str(c) for c in CONTEXTS])
    ax.set_yticks([1.5, 2.0, 2.5, 3.0, 3.5])
    ax.set_yticklabels(["1.5", "2.0", "2.5", "3.0", "3.5"])
    ax.set_xlabel("input tokens", fontsize=7.1, color=SUB, fontfamily=pg.prose, labelpad=2)
    ax.set_ylabel("ms per token", fontsize=7.1, color=SUB, fontfamily=pg.prose, labelpad=2)
    clean_axes(ax)
    pg.text(x, 11.76, title, size=7.9, color=INK, weight="normal")


def draw_r6(pg):
    ax = pg.axes(7.0, 12.1, 2.5, 1.45)
    n = len(R6_VS_R5)
    ax.set_xlim(-14, 5.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.axvline(0, color=INK, linewidth=0.6)
    for i, (label, value, both) in enumerate(R6_VS_R5):
        ax.hlines(i, 0, value, color=RULE, linewidth=1.2, zorder=1)
        if both:
            ax.scatter([value], [i], s=24, facecolor=ACCENT, edgecolor=ACCENT, zorder=3)
        else:
            ax.scatter([value], [i], s=24, facecolor="white", edgecolor=INK, linewidths=0.9, zorder=3)
        offset = -0.4 if value < 0 else 0.4
        ax.text(
            value + offset,
            i,
            f"{value:+.1f}%",
            va="center",
            ha="right" if value < 0 else "left",
            fontsize=6.6,
            color=INK,
            fontfamily=pg.mono,
        )
    ax.set_yticks(range(n))
    ax.set_yticklabels([row[0] for row in R6_VS_R5])
    ax.set_xticks([-12, -8, -4, 0])
    ax.set_xticklabels(["-12%", "-8%", "-4%", "0"])
    clean_axes(ax)
    ax.tick_params(axis="y", labelsize=6.8, length=0)
    ax.set_xlabel("change, percent (negative is faster)", fontsize=7.1, color=SUB, fontfamily=pg.prose, labelpad=2)
    pg.text(7.0, 11.76, "R6 vs R5 on actual vLLM (Exp 131)", size=7.9, color=INK, weight="normal")


def draw_revisions(pg, top):
    row_h = 0.25
    n = len(REVISIONS)
    pg.text(GUTTER, top, "revision", size=6.6, color=SUB, mono=True)
    pg.text(1.5, top, "parity · narrow · wide", size=6.6, color=SUB, mono=True)
    pg.text(2.95, top, "B1 ms, context 32", size=6.6, color=SUB, mono=True)
    pg.text(6.45, top, "B2 ms", size=6.6, color=SUB, mono=True)
    pg.text(7.3, top, "what changed", size=6.6, color=SUB, mono=True)
    rows_top = top + 0.2
    bars = pg.axes(2.95, rows_top, 3.2, n * row_h)
    bars.set_xlim(0, 19)
    bars.set_ylim(n - 0.5, -0.5)
    for i, (name, exp, parity, b1, b2, change) in enumerate(REVISIONS):
        y_row = rows_top + i * row_h
        pg.text(GUTTER, y_row, name, size=8.2, color=INK, weight="normal")
        pg.text(GUTTER, y_row + 0.12, exp, size=6.0, color=SUB, mono=True)
        pg.text(1.5, y_row + 0.01, parity, size=7.4, color=INK, mono=True)
        if b1 is not None:
            bars.barh(i, b1, height=0.5, color=ACCENT, linewidth=0)
            bars.text(b1 + 0.3, i, f"{b1:.2f}", va="center", fontsize=6.7, color=INK, fontfamily=pg.mono)
        else:
            bars.text(
                0.3,
                i,
                REVISION_B1_TEXT[name],
                va="center",
                fontsize=6.7,
                color=SUB,
                fontfamily=pg.prose,
                style="italic",
            )
        b2_text = f"{b2:.2f}" if b2 is not None else REVISION_B2_TEXT.get(name, "")
        pg.text(6.45, y_row + 0.01, b2_text, size=7.0, color=INK, mono=True)
        pg.text(7.3, y_row + 0.01, change, size=6.8, color=SUB)
    bars.axis("off")


def draw_timeline(pg, top):
    height = 1.55
    lane_row = {key: 3 - idx for idx, (key, _, _) in enumerate(LANES)}
    ax = pg.axes(2.3, top, PAGE_W - GUTTER - 2.3, height)
    ax.set_xlim(0, 181)
    ax.set_ylim(-0.6, 3.6)
    for key, label, _ in LANES:
        ax.axhline(lane_row[key], color=RULE, linewidth=0.6, zorder=0)
        ax.text(
            -0.012,
            lane_row[key],
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.2,
            color=INK,
            fontfamily=pg.prose,
            linespacing=1.25,
        )
    lane_of = {e: key for key, _, exps in LANES for e in exps}
    for code in OUTCOMES:
        exps = sorted(CODES[code])
        ax.scatter(exps, [lane_row[lane_of[e]] for e in exps], s=16, zorder=3, **STYLE[code])
    for exp, lane, text, dx, dy, ha in CALLOUTS:
        ax.annotate(
            text,
            xy=(exp, lane_row[lane]),
            xytext=(exp + dx, lane_row[lane] + dy),
            fontsize=6.6,
            color=INK,
            fontfamily=pg.mono,
            ha=ha,
            va="center",
        )
    ax.set_yticks([])
    ax.set_xticks([1, 25, 50, 75, 100, 125, 150, 180])
    ax.set_xlabel(
        "experiment number, chronological (ledger order)", fontsize=7.1, color=SUB, fontfamily=pg.prose, labelpad=2
    )
    clean_axes(ax)
    ax.spines["bottom"].set_visible(True)
    handles = []
    labels = []
    for code, label in OUTCOMES.items():
        style = STYLE[code]
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=style["marker"],
                markersize=5.4,
                markerfacecolor=style["facecolor"],
                markeredgecolor=style.get("edgecolor", style["facecolor"]),
                markeredgewidth=1.0,
            )
        )
        labels.append(f"{label} ({len(CODES[code])})")
    pg.fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(GUTTER / PAGE_W, 1 - (top + height + 0.45) / PAGE_H),
        ncol=4,
        frameon=False,
        prop={"family": pg.prose, "size": 7.2},
        handletextpad=0.3,
        columnspacing=1.6,
        labelspacing=0.45,
    )


def draw_serving(pg, y):
    parts = []
    for label, ((s1, s2), (c1, c2)) in SERVING.items():
        parts.append(f"{label}: B1 {c1:.1f} vs stock {s1:.1f} ms, B2 {c2:.1f} vs stock {s2:.1f} ms")
    text = (
        "Serving still trails stock (in-process vLLM, scheduler-inclusive ITL, prompt 32). "
        + "; ".join(parts)
        + ". Greedy tokens match in all paired trials; exact logprob hashes match in none."
    )
    pg.text(GUTTER, y, fill(text, 150), size=7.2, color=SUB)


def draw_footer(pg, y):
    pg.text(
        GUTTER,
        y,
        fill(
            "Verified: both candidate runs beat both controls in a matched A-B-B-A (the ledger's bar). Component-only: "
            "measured on one kernel, trainer step or memory, with no end-to-end RL win established. Validated: a "
            "correctness or capacity result, not a speed claim. Rejected: not kept, or slower. Fails parity: a strict "
            "numerical gate failed. Blocked: a watchdog, OOM or host-floor stop with no result. Source: "
            "examples/tiny/EXPERIMENTS.md, Experiments 1-180; the numbers are embedded in "
            "tools/plot_tiny_hillclimb.py.",
            175,
        ),
        size=6.8,
        color=SUB,
    )


def build(output: Path, prose: str, mono: str) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=DPI, facecolor="white")
    pg = Page(fig, prose, mono)

    pg.text(GUTTER, 0.28, "examples/tiny/EXPERIMENTS.md · experiments 1-180", size=6.8, color=SUB, mono=True)
    pg.text(GUTTER, 0.52, "tiny-prl hill climb · RTX 4060 Laptop", size=24, color=INK, weight="normal")
    pg.text(
        GUTTER,
        1.15,
        "Guarded A-B-B-A tests on one 8 GiB laptop GPU: which changes moved RL wall time, which only sped up a component, and how far full-parameter training got.",
        size=8.2,
        color=SUB,
    )
    cards = [
        (
            "-12.6%",
            "Exp 6 · decode CUDA graphs",
            "Qwen3-0.6B LoRA RL, A-B-B-A: 90.7 s vs 103.8 s mean wall. Verified, one ABBA cycle.",
        ),
        (
            "-12.2%",
            "Exp 137 · early child cleanup",
            "SmolLM2-135M full-parameter, full process wall: 62.8 s vs 71.5 s mean. Verified. Cleanup 10.8 s to 0.9 s.",
        ),
        (
            "-17 to -19%",
            "Exp 128, 129 · megakernel ITL",
            "vLLM decode inter-token latency in real RL runs. Mismatch KL stays in the stock range. RL wall did not "
            "improve.",
        ),
        (
            "-23.2%",
            "Exp 180 · warm fork server",
            "SmolLM2-135M full-parameter wall: 41.5 s vs 54.0 s mean; loop -43.7%. Verified in both A-B-B-A blocks.",
        ),
    ]
    card_w = (PAGE_W - 2 * GUTTER - 3 * 0.18) / 4
    for i, (big, label, body) in enumerate(cards):
        pg.card(GUTTER + i * (card_w + 0.18), 1.6, card_w, 1.85, big, label, body)

    pg.section(
        3.75,
        "01",
        "Where RL wall time goes",
        "Exp 126 before the cleanup fix, the Exp 137 B-arm mean after it, and the Exp 180 fork-server B-arm mean, same scale.",
    )
    draw_run_bars(pg)

    pg.section(
        6.85,
        "02",
        "Kernel wins that did not move RL wall time",
        "Percent change in the timed step. Verified means both candidates beat both controls; component-only "
        "means no end-to-end RL win.",
    )
    draw_kernel_rows(pg, 7.55)

    pg.section(
        10.95,
        "03",
        "Decode megakernel",
        "Component timings and serving results kept apart; neither is an RL wall-time claim.",
    )
    draw_decode_itl(pg, "B1", 0.95, "B1 · stock vs megakernel (Exp 128)")
    draw_decode_itl(pg, "B2", 3.75, "B2 · stock vs megakernel (Exp 128)")
    draw_r6(pg)
    pg.text(GUTTER, 13.85, "SmolLM2-360M decoder revisions, parity and serving", size=9.0, color=INK, weight="normal")
    pg.text(
        GUTTER,
        14.1,
        "Direct decoder CUDA-event medians at context 32 (Exps 159-174). Parity counts are narrow "
        "fixtures · wide fixtures, as recorded.",
        size=7.0,
        color=SUB,
    )
    draw_revisions(pg, 14.35)
    draw_serving(pg, 16.5)

    pg.section(
        17.1,
        "04",
        "Index of all 180 experiments",
        "One marker per experiment on one timeline, four lanes, colored by outcome as the ledger records it.",
    )
    draw_timeline(pg, 17.95)
    draw_footer(pg, 20.55)

    fig.savefig(output, dpi=DPI, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("outputs", help="accepted for CLI compatibility; unused, the figure embeds its data")
    parser.add_argument("output", type=Path, help="PNG path to write, e.g. examples/tiny/hillclimb.png")
    args = parser.parse_args()
    check_index()
    prose, mono = register_fonts()
    build(args.output, prose, mono)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
