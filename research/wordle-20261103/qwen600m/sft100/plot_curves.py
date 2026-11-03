"""Plot the recorded Qwen Wordle SFT loss and learning rate separately."""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "tools"))
import plot_tiny_hillclimb as theme

prose, mono = theme.register_fonts()
plt.rcParams.update({"font.family": prose, "font.size": 11, "axes.labelcolor": theme.SUB,
                     "text.color": theme.INK, "xtick.color": theme.SUB, "ytick.color": theme.SUB})
with (HERE / "curves.csv").open() as file:
    rows = list(csv.DictReader(file))
steps = [int(row["step"]) for row in rows]
loss = [float(row["loss"]) for row in rows]
lr = [float(row["learning_rate"]) for row in rows]


def canvas(title, subtitle, ylabel):
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    fig.subplots_adjust(left=0.11, right=0.95, bottom=0.17, top=0.74)
    fig.text(0.11, 0.93, "TINY-PRL  /  QWEN3-600M", fontfamily=mono, fontsize=10, color=theme.ACCENT)
    fig.text(0.11, 0.85, title, fontsize=23, weight="semibold")
    fig.text(0.11, 0.79, subtitle, fontsize=11, color=theme.SUB)
    fig.text(0.11, 0.05, "Wordle SFT | LoRA rank 8 | batch 4 | 100 updates", fontfamily=mono, fontsize=9, color=theme.SUB)
    ax.set_facecolor(theme.CARD)
    ax.grid(axis="y", color=theme.RULE, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["bottom", "left"]:
        ax.spines[side].set_color(theme.RULE)
    ax.set(xlabel="Training step", ylabel=ylabel, xlim=(1, 100), xticks=[1, 25, 50, 75, 100])
    return fig, ax


fig, ax = canvas("Training loss", "Measured batch loss; the darker line is a 10-step trailing mean.", "Cross-entropy loss")
ax.plot(steps, loss, color=theme.ACCENT_TINT, linewidth=1.2, label="Raw step loss")
mean = [sum(loss[i - 9:i + 1]) / 10 for i in range(9, len(loss))]
ax.plot(steps[9:], mean, color=theme.ACCENT, linewidth=2, label="10-step mean")
ax.scatter([steps[0], steps[-1]], [loss[0], loss[-1]], color=theme.ACCENT, s=25, zorder=3)
ax.annotate(f"{loss[0]:.3f}", (steps[0], loss[0]), xytext=(9, 6), textcoords="offset points", fontfamily=mono, fontsize=10)
ax.annotate(f"{loss[-1]:.3f}", (steps[-1], loss[-1]), xytext=(-9, -17), ha="right", textcoords="offset points", fontfamily=mono, fontsize=10)
ax.set_ylim(0, max(loss) * 1.12)
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.savefig(HERE / "loss.png", facecolor="white")
plt.close(fig)

fig, ax = canvas("Learning rate", "Constant schedule: 1.0e-4 throughout the run.", "Learning rate")
ax.step(steps, lr, where="post", color=theme.ACCENT, linewidth=2)
ax.set_ylim(0, max(lr) * 1.2)
formatter = ScalarFormatter(useMathText=False)
formatter.set_powerlimits((-4, -4))
ax.yaxis.set_major_formatter(formatter)
fig.savefig(HERE / "learning-rate.png", facecolor="white")
plt.close(fig)
