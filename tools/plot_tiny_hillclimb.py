#!/usr/bin/env python3
"""Render the tiny-prl hill-climb poster (examples/tiny/hillclimb.png).

Every number is embedded from examples/tiny/EXPERIMENTS.md (Experiments 1-231);
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
PAGE_H = 51.5  # inches; append the serving panel at the existing scale
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
# Exp 182 early-trainer B-arm mean and its A-arm (fork server only) mean.
EARLY_182 = 36.19
EARLY_182_CONTROL = 41.96

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

# Experiment index. Outcome codes and lanes for all 231 experiments.
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
    "V": [6, 63, 74, 97, 101, 106, 128, 129, 131, 134, 137, 180, 182],
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
        187,
        192,
        193,
        197,
        198,
        207,
        201,
        217,
        219,
        222,
        223,
        225,
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
        199,
        200,
        209,
        224,
    ],
    "X": [
        218,
        214,
        216,
        181,
        184,
        185,
        188,
        189,
        190,
        191,
        195,
        196,
        202,
        206,
        210,
        211,
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
    "B": [183, 2, 19, 22, 27, 29, 30, 36, 41, 42, 45, 46, 56, 88, 91],
    "D": [3, 13, 24, 26, 31, 32, 33, 44, 49, 58, 67, 102, 115, 117, 125, 126, 162, 165, 169, 186, 194, 203, 204, 205, 208, 212, 213, 215, 220, 221],
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
            181,
            182,
            183,
            185,
            204,
            207,
            224,
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
            200,
            209,
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
            184,
            186,
            187,
            188,
            189,
            190,
            191,
            192,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            201,
            202,
            203,
            205,
            206,
            208,
            210,
            211,
            212,
            213,
            214,
            215,
            216,
            217,
            218,
            219,
            220,
            221,
            222,
            223,
            225,
        ],
    ),
]
CODES["C"].append(226)
LANES[-1][2].append(226)
CODES["C"].append(227)
LANES[-1][2].append(227)
CODES["D"].append(228)
LANES[-1][2].append(228)
CODES["D"].append(229)
CODES["X"].append(230)
LANES[-1][2].extend([229, 230])
CODES["X"].append(231)
LANES[-1][2].append(231)

# Callouts: (experiment, lane, text, x offset in experiments, y offset in lanes, alignment).
CALLOUTS = [
    (6, "R", "6  CUDA graphs, -12.6%", 1.5, 0.55, "left"),
    (137, "R", "137  cleanup, -12.2%", -1.5, 0.55, "right"),
    (180, "R", "180  fork server, -23.2%", -1.5, -0.6, "right"),
    (182, "R", "182  early trainer, -13.7%", -1.5, 0.55, "right"),
    (19, "F", "19  host-RAM floor", 1.5, -0.6, "left"),
    (36, "F", "36  10 updates, checkpoint OOM", 1.5, 0.55, "left"),
    (94, "F", "94  Qwen bounded slices", 1.5, 0.55, "left"),
    (53, "T", "53  Inductor fails parity", 1.5, 0.55, "left"),
    (128, "M", "128  megakernel in vLLM", -1.5, 0.55, "right"),
    (174, "M", "174  46/46 fixtures", -1.5, 0.55, "right"),
    (213, "M", "213  lossless GU, 18.8% smaller", -1.5, -0.35, "right"),
    (214, "M", "214  reject, +100.4%", -1.5, 0.55, "right"),
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
    if "Lexend" not in names or "IBM Plex Mono" not in names:
        raise RuntimeError("TINY_PRL_FONT_DIR must contain Lexend and IBM Plex Mono fonts")
    prose = "Lexend"
    mono = "IBM Plex Mono"
    return prose, mono


def check_index() -> None:
    codes_of = {e: code for code, exps in CODES.items() for e in exps}
    lanes_of = {e: key for key, _, exps in LANES for e in exps}
    assert len(codes_of) == sum(map(len, CODES.values())) == 231
    assert len(lanes_of) == sum(len(exps) for _, _, exps in LANES) == 231
    assert sorted(codes_of) == list(range(1, 232)), "every experiment needs exactly one outcome"
    assert sorted(lanes_of) == list(range(1, 232)), "every experiment needs exactly one lane"


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
    ax.set_ylim(-2.55, 1.6)
    startup, loop, shutdown = RUN_126["startup"], RUN_126["loop"], RUN_126["shutdown"]
    seam = dict(edgecolor="white", linewidth=1.5)
    ax.barh(1, startup, left=0, height=0.5, color=PALE, **seam)
    ax.barh(1, loop, left=startup, height=0.5, color=ACCENT_TINT, **seam)
    ax.barh(1, shutdown, left=startup + loop, height=0.5, color=PALE, **seam)
    ax.barh(0, AFTER_FIX_137, left=0, height=0.5, color=ACCENT, **seam)
    ax.barh(-1, FORK_180, left=0, height=0.5, color=ACCENT, **seam)
    ax.barh(-2, EARLY_182, left=0, height=0.5, color=ACCENT, **seam)
    ax.text(
        EARLY_182 / 2,
        -2,
        f"{EARLY_182:.1f} s",
        ha="center",
        va="center",
        fontsize=8.6,
        color="white",
        fontfamily=pg.mono,
    )
    ax.text(
        EARLY_182 + 1.2,
        -2,
        f"vs {EARLY_182_CONTROL:.1f} s fork-server control mean, -13.7%",
        ha="left",
        va="center",
        fontsize=7.3,
        color=SUB,
        fontfamily=pg.prose,
    )
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
    ax.set_yticks([1, 0, -1, -2])
    ax.set_yticklabels(
        [
            "Exp 126 run\nbefore cleanup fix",
            "Exp 137 B arms\nafter cleanup fix",
            "Exp 180 B arms\nwarm fork server",
            "Exp 182 B arms\n+ early trainer start",
        ],
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
    ax.set_xlim(0, 231)
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
        ax.scatter(exps, [lane_row[lane_of[e]] for e in exps], s=16, zorder=3, clip_on=False, **STYLE[code])
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
    ax.set_xticks([1, 25, 50, 75, 100, 125, 150, 175, 231])
    ax.get_xticklabels()[-1].set_ha("right")
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
        "Early serving pilots trail stock (in-process vLLM, scheduler-inclusive ITL, prompt 32). "
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
            "examples/tiny/EXPERIMENTS.md, Experiments 1-231; the numbers are embedded in "
            "tools/plot_tiny_hillclimb.py.",
            175,
        ),
        size=6.8,
        color=SUB,
    )


def draw_recent_serving(pg):
    comparisons = [
        ("Exp 187 · R28 vs stock", -18.2, -7.7),
        ("Exp 193 · R30 vs R28", -2.773, -0.628),
        ("Exp 202 · R34 vs R30 decoder", +0.119, +0.784),
    ]
    ax = pg.axes(3.45, 21.25, 3.8, 1.2)
    ax.set_xlim(-20, 2.5)
    ax.set_ylim(len(comparisons) - 0.5, -0.5)
    ax.axvline(0, color=INK, linewidth=0.6)
    for i, (label, b1, b2) in enumerate(comparisons):
        rejected = i == 2
        ax.hlines(i, min(0, b1), max(0, b1), color=PALE if rejected else ACCENT, linewidth=2.2, zorder=1)
        if rejected:
            ax.scatter([b1, b2], [i, i], s=24, marker="x", color=INK, linewidths=1.0, zorder=3)
        else:
            ax.scatter([b1], [i], s=25, facecolor=ACCENT, edgecolor=ACCENT, zorder=3)
            ax.scatter([b2], [i], s=25, facecolor="white", edgecolor=ACCENT, linewidths=0.9, zorder=3)
        b1_label_x = b1 + 0.35 if b1 < -16 else b1 - 0.35
        b1_align = "left" if b1_label_x > b1 else "right"
        ax.text(b1_label_x, i - 0.12, f"B1 {b1:+.1f}%", ha=b1_align, va="bottom", fontsize=6.3, color=INK, fontfamily=pg.mono)
        b2_label_x = b2 - 0.35 if b2 > -1 else b2 + 0.35
        b2_align = "right" if b2_label_x < b2 else "left"
        ax.text(b2_label_x, i + 0.13, f"B2 {b2:+.1f}%", ha=b2_align, va="top", fontsize=6.3, color=INK, fontfamily=pg.mono)
    ax.set_yticks(range(len(comparisons)))
    ax.set_yticklabels([row[0] for row in comparisons], fontfamily=pg.prose, fontsize=6.7)
    ax.set_xticks([-20, -15, -10, -5, 0, 2])
    ax.set_xticklabels(["-20%", "-15%", "-10%", "-5%", "0", "+2%"])
    clean_axes(ax)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("context 32 change vs named baseline, percent (negative is faster)", fontsize=6.8, color=SUB, fontfamily=pg.prose, labelpad=2)
    ax.spines["bottom"].set_visible(True)
    pg.text(7.5, 21.3, fill("Serving: R28 vs stock (187), R30 vs R28 (193). R28 passes 29/46 per-case max-logit floors (187).", 28), size=6.8, color=SUB)
    pg.text(7.5, 22.05, fill("Exp 202 is direct decoder timing, not serving; R34 is rejected on B2. R30 remains unadopted.", 28), size=6.8, color=SUB)
    card = pg.axes(GUTTER, 22.7, PAGE_W - 2 * GUTTER, 1.15)
    card.set_facecolor(CARD)
    card.set_xticks([])
    card.set_yticks([])
    for spine in card.spines.values():
        spine.set_visible(False)
    card.text(0.025, 0.82, "174.079 s stock  ·  145.818 s R30", transform=card.transAxes, fontsize=12, color=ACCENT, fontfamily=pg.mono, va="top")
    card.text(0.025, 0.48, "Exp 200 · SmolLM2-360M, full parameters, two updates each", transform=card.transAxes, fontsize=7.6, color=INK, fontfamily=pg.prose, va="top")
    card.text(0.025, 0.25, "40,761 stock tokens vs 33,285 R30 tokens; unequal work does not establish an RL speedup.", transform=card.transAxes, fontsize=7.3, color=SUB, fontfamily=pg.prose, va="top")


def draw_forkserver_rl(pg):
    runs = {"off": [134.072745, 137.827501], "on": [72.792303, 72.546031]}
    means = {"off": 135.950, "on": 72.669}
    ys = {"off": 0, "on": 1}
    colors = {"off": MUTED, "on": ACCENT}
    ax = pg.axes(2.65, 25.35, PAGE_W - GUTTER - 2.65, 1.25)
    ax.set_xlim(60, 150)
    ax.set_ylim(-0.8, 1.85)
    for arm in ("off", "on"):
        y = ys[arm]
        ax.vlines(means[arm], y - 0.18, y + 0.18, color=colors[arm], linewidth=1.1)
        ax.scatter(runs[arm], [y - 0.08, y + 0.08], s=28, color=colors[arm], zorder=3)
        ax.scatter([means[arm]], [y], s=44, marker="D", facecolor="white", edgecolor=colors[arm], linewidth=1.1, zorder=4)
        ax.text(means[arm], y + 0.57, f"mean {means[arm]:.3f}s", ha="center", va="bottom", fontsize=6.2, color=colors[arm], fontfamily=pg.mono)
        tags = ("A1", "A2") if arm == "off" else ("B1", "B2")
        for value, tag, offset in zip(runs[arm], tags, (-0.28, 0.25)):
            ax.text(value, y + offset, f"{tag} {value:.3f}", ha="center", va="center", fontsize=5.8, color=INK, fontfamily=pg.mono)
    ax.set_yticks([0, 1], ["fork server off", "fork server on"])
    ax.set_xticks([60, 80, 100, 120, 140])
    ax.set_xlabel("actual RL child elapsed, seconds · full-parameter SmolLM2-360M, two updates", fontsize=6.6, color=SUB, fontfamily=pg.prose, labelpad=1)
    clean_axes(ax)
    ax.tick_params(axis="y", length=0, labelsize=6.5)
    ax.spines["bottom"].set_visible(True)
    pg.text(GUTTER, 27.22, fill("Exp 204 · 46.55% lower observed wall; unequal work, one ABBA block. Off 592 rollouts / 64,553 tokens; on 572 / 62,864. Not a verified equal-work gain.", 142), size=6.7, color=SUB)
    pg.text(GUTTER, 27.52, fill("Exp 203 · R35 unchanged-B1 placebo shifts keep selective QKV/O packing inconclusive; not adopted. Exp 204 is in the RL runtime lane.", 142), size=6.7, color=SUB)


def draw_fixed_work_runtime(pg):
    runs = {
        "off": [3.347556148, 3.302596620, 3.321897478, 3.339739535, 3.351311772, 3.339909479],
        "on": [0.564080237, 0.573481695, 0.570275755, 0.577663103, 0.557987118, 0.566924907],
    }
    medians = {"off": 3.339824507, "on": 0.568600331}
    ys = {"off": 0, "on": 1}
    colors = {"off": MUTED, "on": ACCENT}
    ax = pg.axes(2.65, 28.7, PAGE_W - GUTTER - 2.65, 1.15)
    ax.set_xlim(0, 3.65)
    ax.set_ylim(-0.55, 1.55)
    for arm in ("off", "on"):
        y = ys[arm]
        offsets = [-0.19, -0.11, -0.04, 0.04, 0.11, 0.19]
        ax.scatter(runs[arm], [y + offset for offset in offsets], s=20, color=colors[arm], zorder=3)
        ax.vlines(medians[arm], y - 0.25, y + 0.25, color=colors[arm], linewidth=1.2)
        label_x = medians[arm] - 0.08 if arm == "off" else medians[arm] + 0.08
        ax.text(label_x, y, f"median {medians[arm]:.3f} s", ha="right" if arm == "off" else "left", va="center", fontsize=6.5, color=colors[arm], fontfamily=pg.mono)
    ax.set_yticks([0, 1], ["fork server off", "fork server on"])
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax.set_xlabel("warm wall time per fixed eight-episode block, seconds", fontsize=6.7, color=SUB, fontfamily=pg.prose, labelpad=2)
    clean_axes(ax)
    ax.tick_params(axis="y", length=0, labelsize=6.5)
    ax.spines["bottom"].set_visible(True)
    pg.text(7.15, 28.66, "Exp 207 · 83.0% lower median", size=7.1, color=ACCENT, mono=True)
    pg.text(GUTTER, 30.3, fill("Six placements per arm · 48 episodes each · concurrency 2. Deterministic mock SSE, CUDA hidden; no live policy, GPU or optimizer. Warm timing excludes each arm's first invocation. Source: outputs/rollout-overhead-20261101/forkserver-fixed-work-r1/results.json.", 165), size=6.5, color=SUB)
    pg.text(GUTTER, 30.67, fill("Exp 209 · usable public 360M full-parameter preset; two-update stock-vLLM pilot passed, 7,398 MiB sampled GPU peak and zero cgroup swap. No speed claim. examples/tiny/smollm2-360m-full-stream64.toml.", 165), size=6.5, color=SUB)


def draw_simd_projection(pg):
    arms = [
        ("Dense", 43.136001, MUTED),
        ("Original compressed", 86.458500, PALE),
        ("SIMD pair decode", 75.467002, ACCENT),
    ]
    ax = pg.axes(2.3, 32.05, PAGE_W - GUTTER - 2.3, 1.2)
    ax.set_xlim(0, 103)
    ax.set_ylim(-0.55, 2.55)
    for i, (name, value, color) in enumerate(arms):
        y = 2 - i
        ax.barh(y, value, height=0.48, color=color, edgecolor=ACCENT if name == "Original compressed" else color, linewidth=0.65 if name == "Original compressed" else 0)
        ax.text(value + 1.2, y, f"{value:.3f} us", va="center", fontsize=6.7, color=INK, fontfamily=pg.mono)
    ax.set_yticks([2, 1, 0], [arm[0] for arm in arms])
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("median projection time per layer, microseconds · lower is faster", fontsize=6.7, color=SUB, fontfamily=pg.prose, labelpad=2)
    clean_axes(ax)
    ax.tick_params(axis="y", length=0, labelsize=6.5)
    ax.spines["bottom"].set_visible(True)
    pg.text(6.6, 31.91, "SIMD: -12.7% vs compressed · +75.0% vs dense", size=7.0, color=ACCENT, mono=True)
    pg.text(GUTTER, 33.65, fill("Exp 217 · balanced same-session three-arm probe, six placements × five replays × 32 layers (960 calls/arm). First two-arm run: compressed 86.304 to SIMD 75.424 us/layer. Exact decoded weights and outputs; 80 registers, 7,680 B shared, no spills. Raw projection component only; no serving adoption or RL win. Evidence: outputs/megakernel-20261025/gateup-lossless-simd-decode-r1/results-r1.json and results-r2.json.", 165), size=6.5, color=SUB)
    pg.text(GUTTER, 34.22, "Exp 218 · bulk raw hints: 83.616 us/layer (+10.814% vs SIMD control); rejected, retain per-row hints.", size=6.4, color=SUB)
    pg.text(GUTTER, 34.47, "Exp 219 · predicated decode: 61.215 us vs SIMD 79.803, dense 48.650; component-only, no adoption.", size=6.4, color=SUB)


def draw_cache_working_set(pg):
    r1 = [(1, 36.015999), (2, 36.030000), (4, 38.189501), (32, 61.209500)]
    r2_anchor = (31.392001, 31.441000, 31.485500)
    r2_all_layers = 53.647999

    pg.text(0.65, 35.48, "R1 · layer-call map", size=7.2, color=INK)
    ax = pg.axes(0.85, 35.76, 3.85, 1.35)
    xs = [0, 1, 2, 5]
    ys = [value for _, value in r1]
    ax.plot(xs, ys, color=ACCENT, linewidth=1.4, marker="o", markersize=4, markerfacecolor=ACCENT)
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=6.1,
            color=INK,
            fontfamily=pg.mono,
        )
    ax.set_xlim(-0.4, 5.4)
    ax.set_ylim(28, 68)
    ax.set_xticks(xs, ["1", "2", "4", "32"])
    ax.set_yticks([30, 45, 60])
    ax.set_ylabel("µs/call", fontsize=6.2, color=SUB, fontfamily=pg.prose, labelpad=1)
    ax.set_xlabel("unique layers touched · log₂ spacing", fontsize=6.2, color=SUB, fontfamily=pg.prose, labelpad=1)
    clean_axes(ax)
    ax.tick_params(axis="x", labelsize=6.2, pad=1)

    pg.text(5.05, 35.48, "R2 · 32 warm anchors vs all layers", size=7.2, color=INK)
    ax = pg.axes(5.1, 35.76, 3.95, 1.35)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(28, 60)
    ax.vlines(0, r2_anchor[0], r2_anchor[2], color=ACCENT, linewidth=2.0)
    ax.scatter([0], [r2_anchor[1]], s=25, color=ACCENT, zorder=3)
    ax.scatter([1], [r2_all_layers], s=25, color=INK, zorder=3)
    ax.annotate(
        "31.44 median\n31.39–31.49 range",
        (0, r2_anchor[1]),
        xytext=(8, -1),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=5.9,
        color=INK,
        fontfamily=pg.mono,
    )
    ax.annotate(
        "53.65",
        (1, r2_all_layers),
        xytext=(0, -10),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=6.1,
        color=INK,
        fontfamily=pg.mono,
    )
    ax.set_xticks([0, 1], ["warm anchor", "all layers"])
    ax.set_yticks([30, 45, 60])
    ax.set_ylabel("µs/call", fontsize=6.2, color=SUB, fontfamily=pg.prose, labelpad=1)
    clean_axes(ax)
    ax.tick_params(axis="x", labelsize=5.9, pad=1)

    pg.text(0.65, 37.45, "960 timed calls/case · all 32 layers resident; only the call map changes.", size=6.1, color=SUB)
    pg.text(5.05, 37.45, "Separate R2 block · one 32-call prewarm per placement outside timing (192/case).", size=6.1, color=SUB)
    pg.text(0.65, 37.70, "Exact FP32/BF16 outputs and decoded weights. Cache-pressure evidence, not an RL gain.", size=6.1, color=SUB)
    pg.text(5.05, 37.70, "Plus 32 proof calls/case. Same kernel; R1/R2 times not compared; no RL claim.", size=6.1, color=SUB)
    pg.text(
        GUTTER,
        37.96,
        "Exp 222 · separate same-seed block: rolling-row depth 1 48.43 µs vs control 56.74 (−14.6%), but +12.2% vs dense 43.15. Broad placement ranges; component-only, no serving/RL win.",
        size=5.9,
        color=SUB,
    )
    pg.text(
        GUTTER,
        38.07,
        "Exp 223 · staged CSR 41.584 vs dense 43.132 µs (−3.59%) in one fixed-work component block; no RL result.",
        size=5.7,
        color=SUB,
    )
    pg.text(
        GUTTER,
        38.17,
        "Exp 224 · baseline RL child 83.80 s/444 rollouts/48,871 tokens; candidate invocation 67.47 s/370/40,784 incl. post-run validation. Unequal work; no RL speedup.",
        size=5.7,
        color=SUB,
    )
    pg.text(
        GUTTER,
        38.27,
        "Decoder timing: B1 +0.01/+0.02/+0.03%; B2 +10.59/+10.20/+8.84% at contexts 32/128/1024. Three blocks; B fails all-block win gate; opt-in only.",
        size=5.7,
        color=SUB,
    )
    pg.text(
        GUTTER,
        38.37,
        "Exp 225 · packaged B2 gains 0.93–1.25% across 3 blocks; B1 neutral. Exact parity; RL pilot has no matched pair or curve point.",
        size=5.7,
        color=SUB,
    )


# Experiment 226: five process medians per arm; intervals span process medians.
SERVING_226 = {1: {'baseline': [[217.804642, 217.37466, 219.199765],
                  [220.493007, 220.294873, 221.706146],
                  [263.370947, 262.89034, 263.398127]],
     'candidate': [[219.113117, 217.789139, 219.660932],
                   [221.230579, 220.231775, 222.077904],
                   [263.353911, 263.142441, 263.567222]]},
 2: {'baseline': [[234.027362, 233.375559, 235.93566],
                  [240.531401, 240.225919, 240.622486],
                  [321.757992, 321.231344, 322.20264]],
     'candidate': [[232.03072, 231.707322, 234.362775],
                   [237.77427, 237.572208, 238.011967],
                   [319.933932, 319.38728, 320.139724]]}}


def draw_lossless_serving(pg):
    for batch, x in ((1, 0.95), (2, 5.35)):
        ax = pg.axes(x, 39.32, 3.55, 1.45)
        for arm, color, label in (("baseline", INK, "R30"), ("candidate", ACCENT, "lossless")):
            rows = SERVING_226[batch][arm]
            medians = [row[0] for row in rows]
            ax.errorbar(range(3), medians,
                        yerr=[[row[0] - row[1] for row in rows], [row[2] - row[0] for row in rows]],
                        color=color, marker="o", markersize=3, linewidth=1, capsize=3, label=label)
        ax.set_xlim(-0.25, 2.25)
        ax.set_xticks(range(3), ["32", "128", "1024"])
        ax.set_xlabel("prompt tokens", fontsize=6.6, fontfamily=pg.prose, color=SUB)
        ax.set_ylabel("ms / 64-token batch", fontsize=6.5, fontfamily=pg.prose, color=SUB)
        ax.set_title(f"Batch {batch}", fontsize=8, fontfamily=pg.prose, color=INK)
        clean_axes(ax)
        ax.legend(frameon=False, prop={"family": pg.prose, "size": 6.5}, loc="upper left")
    pg.text(GUTTER, 41.07, "B2 median change: −0.85% / −1.15% / −0.57% at 32 / 128 / 1024. Context 32 ranges overlap; B1 has no clear change.", size=6.7, color=SUB)
    pg.text(GUTTER, 41.22, "5 processes/arm × 5 repeats/case · whiskers: min–max process medians · identical tokens and sampled-token logprobs.", size=6.7, color=SUB)
    pg.text(GUTTER, 41.37, "Seeded token-ID prompts; equal KV capacity. Includes prefill, LM head and scheduling. Serving result only; no RL curve point.", size=6.7, color=SUB)



# Experiment 227: B1 only; private micro and serving processes share a frozen control.
B1_227 = {'CUDA step': {'baseline': [[2.88972806930542, 2.888192057609558, 2.8902400732040405],
                            [2.939039945602417, 2.9377280473709106, 2.9398080110549927],
                            [3.2419838905334473, 3.239936113357544, 3.2428799867630005]],
               'candidate': [[2.8639520406723022, 2.8620800971984863, 2.871295928955078],
                             [2.918976068496704, 2.9173760414123535, 2.9210879802703857],
                             [3.2182559967041016, 3.2175040245056152, 3.220863938331604]]},
 'Generation': {'baseline': [[218.46132599989687, 217.5715239998226, 219.22142399989752],
                             [220.9193429998777, 220.13550800011217, 221.90102099989417],
                             [262.9800319998594, 262.86013699996147, 263.31665399993653]],
                'candidate': [[216.78513500000918, 216.19774900000266, 218.16210300016792],
                              [219.62488199983454, 219.29522200002793, 220.55091400011406],
                              [261.76804000010634, 261.70436500001415, 262.52176300022256]]}}


def draw_b1_lossless(pg):
    for label, x, unit in (("CUDA step", 0.95, "ms / decoder step"),
                           ("Generation", 5.35, "ms / 64-token batch")):
        ax = pg.axes(x, 42.32, 3.55, 1.45)
        for arm, color, name in (("baseline", INK, "B2-only GU"), ("candidate", ACCENT, "B1+B2 GU")):
            rows = B1_227[label][arm]
            ax.errorbar(range(3), [r[0] for r in rows],
                        yerr=[[r[0] - r[1] for r in rows], [r[2] - r[0] for r in rows]],
                        color=color, marker="o", markersize=3, linewidth=1, capsize=3, label=name)
        ax.set_xlim(-0.25, 2.25)
        ax.set_xticks(range(3), ["32", "128", "1024"])
        ax.set_xlabel("prompt tokens", fontsize=6.6, fontfamily=pg.prose, color=SUB)
        ax.set_ylabel(unit, fontsize=6.5, fontfamily=pg.prose, color=SUB)
        ax.set_title(label, fontsize=8, fontfamily=pg.prose, color=INK)
        clean_axes(ax)
        ax.legend(frameon=False, prop={"family": pg.prose, "size": 6.5}, loc="upper left")
    pg.text(GUTTER, 44.07, "B1 paired CUDA change: −0.90% / −0.68% / −0.73%; strict blocks 14/15, 15/15, 15/15. One context-32 drift block retained.", size=6.7, color=SUB)
    pg.text(GUTTER, 44.22, "Serving medians: −0.77% / −0.59% / −0.46%; only context 1024 has disjoint process ranges. Tokens and sampled logprobs exact.", size=6.7, color=SUB)
    pg.text(GUTTER, 44.37, "5 processes/arm · whiskers span process medians · B2 instructions unchanged · opt-in package; no verified RL wall-time point.", size=6.7, color=SUB)



# Experiment 228: thread0 intervals, median CTA sum across 32 layers; not wall shares.
PROFILE_228 = [196721.75, 148384.25, 2911826.25, 1202563.5]


def draw_b1_profile(pg):
    ax = pg.axes(2.0, 45.32, 3.0, 1.35)
    labels = ["metadata + hints + wait", "norm2", "GU loop", "down hints + wait"]
    for i, value in enumerate(PROFILE_228):
        ax.barh(i, value / 1e6, color=ACCENT if i == 2 else MUTED, height=0.5)
        ax.text(value / 1e6 + 0.045, i, f"{value / 1e6:.3f}", va="center", fontsize=6.2, color=INK, fontfamily=pg.mono)
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 3.35)
    ax.set_xlabel("million scheduled cycles · B1, context 128", fontsize=6.3, fontfamily=pg.prose, color=SUB)
    clean_axes(ax)
    ax.tick_params(axis="y", length=0, labelsize=6.1)
    pg.text(5.45, 45.28, "Uninstrumented follow-ups", size=8.0, color=INK)
    pg.text(5.45, 45.58, fill("CTA row balance: 112/104 GU rows per CTA becomes 107/106. Exact outputs, but only 2/3, 2/3, 1/3 strict blocks at contexts 32/128/1024.", 49), size=6.8, color=SUB)
    pg.text(5.45, 46.22, fill("Compiler fences: both kernels are byte-identical to the package. No GPU rerun needed. Neither trial adopted.", 49), size=6.8, color=SUB)
    pg.text(GUTTER, 47.07, "Thread0 interval per CTA, summed across layers; median across CTAs and six placements. Barrier intervals include waiting.", size=6.7, color=SUB)
    pg.text(GUTTER, 47.22, "Instrumentation changes B1 latency by −1.22% to −1.51% across two processes. These are not pure memory costs or wall shares.", size=6.7, color=SUB)
    pg.text(GUTTER, 47.37, "Counter access is admin-restricted. Production kernel retained; no new verified speedup or RL wall-time point.", size=6.7, color=SUB)



# Experiments 229–230: counter execution effect and rejected uninstrumented trials.
TRIALS_230 = {'counters': [0.0953361760832383, 0.033983823085881326, 0.016108370666989558],
 'merged PTX': [-0.08813650110476301, 0.2316929345395069, 0.04771028758212559],
 'rolled GU': [1.2299748943759292, 1.5018214268050833, 0.9569180429838786]}


COUNTER_RANGES_229 = [[0.01816214332743815, 0.16240842049539683], [0.0, 0.07023466168394332], [0.003526257550201528, 0.02569334018940328]]


def draw_counter_controls(pg):
    ax = pg.axes(1.05, 48.32, 3.55, 1.45)
    for i, value in enumerate(TRIALS_230["counters"]):
        low, high = COUNTER_RANGES_229[i]
        ax.barh(i, value, color=MUTED, height=0.5, xerr=[[value - low], [high - value]], ecolor=INK, capsize=3)
        ax.text(high + 0.006, i, f"{value:+.3f}%", va="center", fontsize=6.3, color=INK, fontfamily=pg.mono)
    ax.set_yticks(range(3), ["32", "128", "1024"])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.24)
    ax.set_title("Counters on vs off · same binary", fontsize=8, fontfamily=pg.prose, color=INK)
    ax.set_xlabel("B1 latency change, %", fontsize=6.6, fontfamily=pg.prose, color=SUB)
    ax.set_ylabel("prompt tokens", fontsize=6.5, fontfamily=pg.prose, color=SUB)
    clean_axes(ax)
    ax = pg.axes(5.45, 48.32, 3.4, 1.45)
    ax.axvline(0, color=RULE, linewidth=0.8)
    for label, color, offset in (("merged PTX", INK, -0.08), ("rolled GU", MUTED, 0.08)):
        ax.scatter(TRIALS_230[label], [i + offset for i in range(3)], marker="x", s=25, color=color, label=label)
    ax.set_yticks(range(3), ["32", "128", "1024"])
    ax.invert_yaxis()
    ax.set_xlim(-0.25, 1.8)
    ax.set_title("Uninstrumented trials · not kept", fontsize=8, fontfamily=pg.prose, color=INK)
    ax.set_xlabel("B1 latency change, % · positive is slower", fontsize=6.3, fontfamily=pg.prose, color=SUB)
    clean_axes(ax)
    ax.legend(frameon=False, prop={"family": pg.prose, "size": 6.3}, loc="lower right")
    pg.text(GUTTER, 50.07, "229 · 3 processes × 3 blocks; whiskers span process deltas. Off-poison/on-coverage checked; no useful counter acceleration.", size=6.7, color=SUB)
    pg.text(GUTTER, 50.22, "Guarded-off vs package: B1 −1.24–1.41%, B2 +0.91–1.16% in one diagnostic process. Different compiled build; not adopted.", size=6.7, color=SUB)
    pg.text(GUTTER, 50.37, "230 · one process/variant: merged PTX exact on BF16/mixed-delta proofs, no timing gain; compact GU regresses. Package and RL curve retained.", size=6.7, color=SUB)


def build(output: Path, prose: str, mono: str) -> None:
    fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=DPI, facecolor="white")
    pg = Page(fig, prose, mono)

    pg.text(GUTTER, 0.28, "examples/tiny/EXPERIMENTS.md · experiments 1-231", size=6.8, color=SUB, mono=True)
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
            "-33%",
            "Exp 180 + 182 · rollout & startup",
            "SmolLM2-135M full-parameter wall 54.0 s to 36.2 s: fork server -23.2%, then early trainer start "
            "-13.7%, each verified in both A-B-B-A blocks.",
        ),
    ]
    card_w = (PAGE_W - 2 * GUTTER - 3 * 0.18) / 4
    for i, (big, label, body) in enumerate(cards):
        pg.card(GUTTER + i * (card_w + 0.18), 1.6, card_w, 1.85, big, label, body)

    pg.section(
        3.75,
        "01",
        "Where RL wall time goes",
        "Exp 126 before the cleanup fix, then the B-arm means of Exp 137, Exp 180 and Exp 182, same scale.",
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
        "Index of all 231 experiments",
        "One marker per experiment on one timeline, four lanes, colored by outcome as the ledger records it.",
    )
    draw_timeline(pg, 17.95)
    pg.section(20.55, "05", "360M serving and full-parameter RL", "Serving comparisons keep their own baselines; the two-update RL pair performed unequal work.")
    draw_recent_serving(pg)
    pg.text(GUTTER, 24.15, fill("Exp 198: R33's small B2 serving gain is not independently repeated. Exp 199 verifies filesystem weight refresh reaches R30. Exp 202 rejects B2 packing.", 165), size=6.6, color=SUB)
    pg.section(24.72, "06", "Fork-server runtime experiment", "Actual full-parameter 360M RL child wall times; individual ABBA runs and arm means are shown.")
    draw_forkserver_rl(pg)
    pg.section(28.05, "07", "Fixed-work runtime component", "Warm mock-runtime blocks with identical requests and outputs; seconds are not comparable to the RL chart above.")
    draw_fixed_work_runtime(pg)
    pg.section(31.2, "08", "SIMD pair decode: faster compressed projection, still slower than dense", "Same-session component medians; the lower compressed latency does not establish a serving or RL improvement.")
    draw_simd_projection(pg)
    pg.section(34.82, "09", "Cache working-set diagnostic", "Predicated gate/up projection · fixed kernel and resident weights · R1 and R2 are independent timing blocks.")
    draw_cache_working_set(pg)
    pg.section(38.68, "10", "Lossless 360M: fixed-work generation", "Exp 226 · R30 versus the packaged decoder; 64 greedy tokens, five fresh processes per arm.")
    draw_lossless_serving(pg)
    pg.section(41.68, "11", "Lossless GU reaches batch one", "Exp 227 · same weights, reductions and rounding; compare the existing package with the B1-enabled variant.")
    draw_b1_lossless(pg)
    pg.section(44.68, "12", "B1 phase profile and two scheduling trials", "Exp 228 · diagnostic instrumentation plus uninstrumented parity/timing checks; no adoption.")
    draw_b1_profile(pg)
    pg.section(47.68, "13", "Counter control and rejected kernel trials", "Exps 229–230 · controlled profiling effect, followed by two uninstrumented changes.")
    draw_counter_controls(pg)
    pg.text(GUTTER, 50.59, "231 · paired metadata B1 +1.16% / +1.40% / +0.97%; exact, not kept. B2 machine code unchanged; no gain claimed.", size=6.7, color=SUB)
    pg.text(GUTTER, 50.76, "Original-address control is byte-identical at both batches. No extra GPU sweep, production change or RL curve point.", size=6.7, color=SUB)
    draw_footer(pg, 51.05)

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
