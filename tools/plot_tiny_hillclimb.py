#!/usr/bin/env python3
"""Plot the full measured tiny-prl optimization chronology."""

import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt


def read_json(path: Path) -> object | None:
    return json.loads(path.read_text()) if path.exists() else None


def runs(root: Path, filename: str) -> list[dict]:
    data = read_json(root / filename)
    return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []


def complete(run: dict) -> bool:
    return (
        run.get("returncode") == 0
        and run.get("completed_updates") == 10
        and run.get("expected_updates", 10) == 10
        and run.get("finite_loss_grad") is True
        and run.get("stop_reason") is None
    )


def arm_mean(rows: list[dict], source: str, field: str) -> float | None:
    values = [row.get(field) for row in rows if row.get("source") == source and complete(row)]
    if len(values) != 2 or not all(isinstance(value, (int, float)) for value in values):
        return None
    return sum(values) / 2


def paired_effect(root: Path, filename: str, control: str, candidate: str) -> tuple[float, float] | None:
    rows = runs(root, filename)
    a_time, b_time = arm_mean(rows, control, "elapsed_seconds"), arm_mean(rows, candidate, "elapsed_seconds")
    a_gpu, b_gpu = arm_mean(rows, control, "peak_gpu_mib"), arm_mean(rows, candidate, "peak_gpu_mib")
    if any(value is None for value in (a_time, b_time, a_gpu, b_gpu)) or a_time == 0:
        return None
    return (b_time / a_time - 1) * 100, b_gpu - a_gpu


def mib(value: int | float) -> float:
    return value / 1024**2


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_tiny_hillclimb.py EXPERIMENT_ROOT OUTPUT.png")
    root, output = Path(sys.argv[1]), Path(sys.argv[2])

    font_dir_value = os.environ.get("TINY_PRL_FONT_DIR")
    font_dir = Path(font_dir_value) if font_dir_value else None
    families = set()
    if font_dir and font_dir.is_dir():
        from matplotlib import font_manager

        for font_path in font_dir.glob("*.ttf"):
            font_manager.fontManager.addfont(font_path)
            families.add(font_manager.FontProperties(fname=font_path).get_name())
    sans = "Lexend" if "Lexend" in families else "DejaVu Sans"
    mono = "IBM Plex Mono" if "IBM Plex Mono" in families else "DejaVu Sans Mono"
    plt.rcParams.update({"font.family": sans, "font.size": 9, "axes.unicode_minus": False})

    bg, ink, soft, dim, faint = "#0b0b0c", "#f2efe6", "#cfcabe", "#7d786e", "#3a3833"
    yellow, mint, coral, sky, lilac = "#ffd166", "#7ee0b5", "#ff8a7a", "#8cc8ff", "#c9b2ff"
    stages = [
        ("Baseline", "resume verified", mint),
        ("Imports + lazy transport", "RSS -29%", mint),
        ("Offload\npilot", "0/2 RAM stop", coral),
        ("Direct\nlaunch", "pilot pass; RAM stop", yellow),
        ("Inline env", "memory; no speed", yellow),
        ("Seq 4", "reject +9.3%", coral),
        ("Decode\ngraph", "retain -12.6%", mint),
        ("Profile +\nchunks", "parity failed", coral),
        ("Vocab\ntiles", "parity failed", coral),
        ("Offload\nretest", "keep offload", yellow),
        ("KV .25", "GPU headroom", mint),
        ("AC off", "keep full AC", yellow),
        ("Queue cap 4", "reject +0.8%", coral),
        ("Compile\nscreen", "pilot only; no claim", yellow),
        ("CuTe\nonline update", "local gain; RL unproven", lilac),
        ("Kernel v2 +\nNsight", "parity + local only;\nRL impact unknown", lilac),
        ("LoRA\nsplit-K", "component pass;\nmodel stress fails", sky),
        ("Head\nbalanced", "same fixture;\nlocal only", lilac),
        ("Backward\nepilogue", "R2 pass; local gain;\nR1 JSON failed; no RL", lilac),
        ("CuTe v2\nfull head", "parity; local\ncomponent only", lilac),
        ("135M\nfull param", "10/10 + DCP\ncheckpoint audit", yellow),
        ("135M resume\n10->20", "10/10 finite;\nMFU 4.18%", mint),
        ("64 MB\nbroadcast", "2/2 + ACK;\nfunctional only", mint),
        ("CuTe RL\nA-B-B-A", "10/10 finite;\nno wall gain", lilac),
        ("Stream\nwriter ABBA", "273 exact;\nPSS -192 MiB", sky),
        ("Stream\nRL pilot", "2/2; 59.10 s;\nfunctional only", mint),
        ("SmolLM v1\ntask gate", "519/520 zero;\n0 updates; gate", coral),
        ("TVM-FFI\nhead", "host submit lower;\nevent similar", sky),
        ("TVM-FFI\nRL pilot", "2 updates;\nbranch + ACK", mint),
        ("360M\ntrainer", "ABBA; -446 MiB\nsampled peak; synth", yellow),
        ("360M\ninference", "2x512 reqs;\n.15/.13 separate", sky),
        ("Bounded\nstate", "small parity r5;\n360M capacity r3", yellow),
        ("360M co-res", "both hit cap;\ntrim -904 KiB", coral),
        ("Packed DCP", "QKV + tied;\n3-update exact", mint),
        ("Mixed state", "strict parity;\ncapacity probes", yellow),
        ("360M RL\npilot", "1 update; host stop;\nv1 broadcast incomplete", coral),
        ("Lazy Muon\nimport", "9 modules fewer;\nPSS +412 KiB", coral),
        ("360M phase\ndiagnostic", "host-floor stop;\noptimizer incomplete", coral),
        ("360M phase r2", "host-floor stop;\nF/B +496 MiB", coral),
        ("Allocator phase", "1 trainer-only step;\ntrim -44 KiB", coral),
        ("Tracemalloc", "Python <5 MiB;\nVMA source unknown", coral),
        ("Memray\nnative heap", "0.83 GiB heap;\nCUDA loading labels", yellow),
        ("360M RL\npilot 2", "2/2 steps;\nload HTTP 200s", mint),
        ("CuTe FFI\n360M parity", "strict gradient\ngate failed", coral),
        ("360M 10-update", "10 finite steps;\ncheckpoint OOM", coral),
        ("PTX no-FMA", "strict pass;\n512 rows only", yellow),
        ("PTX no-FMA\nextended", "7 strict cases;\nint32 labels pass", mint),
        ("Serial DCP", "fresh resume to 2;\ntrainer-only", mint),
        ("Registered FFI\nfull head", "23.6% lower;\ncomponent only", sky),
        ("360M serial\nDCP full-stack", "safe host-floor\nstop after 1 update", yellow),
        ("360M 2.25 GiB\nopt budget", "safe host-floor stop;\nnegative capacity", yellow),
        ("135M no-FMA\nFFI head", "strict grad gate\nfailed; not timed", coral),
        ("Inference head\nGEMM roofline", "~93% of BW floor;\n<=0.24% left, no run", lilac),
        ("135M full\nproduct stop", "safe host-floor\nstop; not OOM", yellow),
        ("135M full\n10 updates", "10/10 updates; ckpt-save host stop", yellow),
        ("135M serial\nDCP resume", "trainer-only save+resume pass", mint),
        ("135M full-RL\nbaseline", "ckpt + resume pass", mint),
        ("135M step\nprofile", "launch-bound: GPU\n30% busy", sky),
        ("Triton RoPE\n+ SwiGLU", "RoPE -0.8% / SwiGLU\n+7.6%: no gain/reject", coral),
        ("Inductor\ncompile", "compile -47% trainer\n(later invalid)", yellow),
        ("Triton RMSNorm\n+ RoPE", "-4.7% fixed-shape;\nRL regress (recompile)", coral),
        ("Inductor\nparity gate", "compile fails strict\nparity: rejected", coral),
        ("Triton RL\nrecompiles", "recompiles fixed;\nstart-up autotune +12.8% wall", coral),
        ("Pinned Triton\nRL", "RL fwd/bwd -15.6%;\nwall gain not attributable", mint),
    ]
    x = list(range(len(stages)))
    runtime_comparisons = {
        4: ("pool-comparison.json", "pool-control", "single-worker"),
        5: ("rollout-comparison.json", "seqs2", "seqs4"),
        6: ("graph-comparison.json", "eager", "decode-graph"),
        9: ("activation-retest-comparison.json", "activation-offload-on", "activation-offload-off"),
        10: ("kv-comparison.json", "kv-budget-35", "kv-budget-25"),
        11: ("ac-comparison.json", "checkpoint-full", "checkpoint-off"),
        12: ("queue-comparison.json", "queue-adaptive", "queue-cap4"),
    }
    effects = {stage: paired_effect(root, *comparison) for stage, comparison in runtime_comparisons.items()}

    baseline_rss = read_json(root / "import-control.json")
    latest_rss = read_json(root / "import-decoupled.json")
    import_drop = None
    if isinstance(baseline_rss, dict) and isinstance(latest_rss, dict):
        import_drop = mib(latest_rss["rss_bytes"] - baseline_rss["rss_bytes"])
    health = read_json(root / "single-worker" / "health-comparison.json")
    health_drop = (
        mib(health["pss_reduction_bytes"]) if isinstance(health, dict) and health.get("both_healthy") else None
    )
    pool_rows = runs(root, "pool-comparison.json")
    pool_a = next(
        (
            r
            for r in reversed(pool_rows)
            if r.get("source") == "pool-control" and r.get("peak_cgroup_anon_plus_swap_bytes")
        ),
        None,
    )
    pool_b = next(
        (
            r
            for r in reversed(pool_rows)
            if r.get("source") == "single-worker" and r.get("peak_cgroup_anon_plus_swap_bytes")
        ),
        None,
    )
    job_drop = None
    if pool_a and pool_b:
        job_drop = mib(pool_b["peak_cgroup_anon_plus_swap_bytes"] - pool_a["peak_cgroup_anon_plus_swap_bytes"])

    fig = plt.figure(figsize=(56, 11.5), facecolor=bg)
    grid = fig.add_gridspec(
        3, 1, height_ratios=[1.05, 1.45, 1.45], left=0.075, right=0.985, top=0.835, bottom=0.36, hspace=0.30
    )
    memory_ax, runtime_ax, gpu_ax = [fig.add_subplot(grid[i]) for i in range(3)]
    axes = [memory_ax, runtime_ax, gpu_ax]
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors=soft, labelsize=8, length=0, pad=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", color=faint, linewidth=0.65, alpha=0.75)
        for stage in x:
            ax.axvline(stage, color=faint, linewidth=0.45, alpha=0.48, zorder=0)
        ax.set_xlim(-0.55, len(stages) - 0.45)

    memory_ax.set_ylim(-400, 55)
    memory_ax.set_yticks([-400, -300, -200, -100, 0])
    memory_ax.set_ylabel("MiB change", color=soft, labelpad=10)
    memory_ax.set_title("Host memory · named measurement contexts", loc="left", color=ink, fontsize=12, pad=11)
    memory_ax.tick_params(axis="x", labelbottom=False, bottom=False)
    memory_ax.axhline(0, color=dim, linewidth=0.9)
    memory_points = []
    if import_drop is not None:
        memory_points.append((1, import_drop, yellow, f"import RSS\n{import_drop:+.0f} MiB"))
    memory_points.append((3, -292, sky, "parent USS -292 MiB"))
    if health_drop is not None:
        memory_points.append((4, -health_drop, lilac, f"health PSS\n-{health_drop:.0f} MiB"))
    if job_drop is not None:
        memory_points.append((4, job_drop, mint, f"job anon + swap\n{job_drop:+.0f} MiB"))
    for stage, value, color, label in memory_points:
        memory_ax.scatter(stage, value, color=color, edgecolor=ink, linewidth=0.65, s=42, zorder=3)
        if stage == 4 and value > -220:
            offset, align = (10, 11), "left"
        elif stage == 4:
            offset, align = (-10, -11), "right"
        else:
            offset, align = (0, 10), "center"
        memory_ax.annotate(
            label,
            (stage, value),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            va="bottom" if offset[1] > 0 else "top",
            color=color,
            fontsize=7.3,
            fontfamily=mono,
        )

    inset = memory_ax.inset_axes([0.68, 0.16, 0.30, 0.77])
    inset.set_facecolor(bg)
    import_rows = [
        read_json(root / name)
        for name in ("import-control.json", "import-tokenizer-only.json", "import-decoupled.json")
    ]
    rss_values = [
        mib(row["rss_bytes"])
        for row in import_rows
        if isinstance(row, dict) and isinstance(row.get("rss_bytes"), (int, float))
    ]
    if len(rss_values) == 3:
        inset.plot([0, 1, 2], rss_values, color=dim, linewidth=1.1, zorder=1)
        inset.step(
            [0, 1, 2], [rss_values[0], min(rss_values[:2]), min(rss_values)], where="post", color=mint, linewidth=1.6
        )
        inset.scatter([0, 1, 2], rss_values, color=[dim, sky, yellow], edgecolor=ink, s=25, linewidth=0.5, zorder=3)
        inset.set_xticks([0, 1, 2], ["base", "tokenizer", "lazy"])
        inset.set_ylim(760, 1140)
        inset.set_yticks([800, 950, 1100])
        inset.tick_params(colors=soft, labelsize=6.5, length=0, pad=2)
        inset.grid(axis="y", color=faint, linewidth=0.5, alpha=0.7)
        for spine in inset.spines.values():
            spine.set_visible(False)
        inset.set_title("Import RSS after import · MiB", loc="left", color=yellow, fontsize=7.8, pad=2)
        for ix, value in enumerate(rss_values):
            inset.annotate(
                f"{value:.0f}",
                (ix, value),
                xytext=(0, 8 if ix == 2 else -8),
                textcoords="offset points",
                ha="center",
                va="bottom" if ix == 2 else "top",
                color=soft,
                fontsize=6.8,
                fontfamily=mono,
            )

    runtime_ax.set_ylim(-19, 12)
    runtime_ax.set_yticks([-15, -10, -5, 0, 5, 10])
    runtime_ax.set_ylabel("B vs A (%)", color=soft, labelpad=10)
    runtime_ax.set_title(
        "End-to-end time · local ABBA arm means, complete 10-update runs only",
        loc="left",
        color=ink,
        fontsize=12,
        pad=11,
    )
    runtime_ax.tick_params(axis="x", labelbottom=False, bottom=False)
    runtime_ax.axhline(0, color=yellow, linewidth=1.0, alpha=0.85)
    for stage, (change, _) in effects.items():
        color = stages[stage][2]
        runtime_ax.vlines(stage, 0, change, color=color, linewidth=1.6, alpha=0.85, zorder=2)
        runtime_ax.scatter(stage, change, marker="D", color=color, edgecolor=ink, linewidth=0.6, s=47, zorder=3)
        runtime_ax.annotate(
            f"{change:+.1f}%",
            (stage, change),
            xytext=(0, 8 if change >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va="bottom" if change >= 0 else "top",
            color=color,
            fontsize=8,
            fontfamily=mono,
        )

    kernel_points = {
        58: [(-0.8, "-0.8%"), (7.6, "+7.6%")],
        60: [(-4.7, "-4.7%")],
        62: [(-4.9, "-4.9%")],
        63: [(-15.6, "-15.6%")],
    }
    for stage, points in kernel_points.items():
        color = stages[stage][2]
        for change, text in points:
            runtime_ax.vlines(stage, 0, change, color=color, linewidth=1.6, alpha=0.85, zorder=2)
            runtime_ax.scatter(stage, change, marker="D", facecolor=bg, edgecolor=color, linewidth=1.6, s=47, zorder=3)
            runtime_ax.annotate(
                text,
                (stage, change),
                xytext=(0, 8 if change >= 0 else -10),
                textcoords="offset points",
                ha="center",
                va="bottom" if change >= 0 else "top",
                color=color,
                fontsize=8,
                fontfamily=mono,
            )
    runtime_ax.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="D",
                linestyle="",
                markerfacecolor=bg,
                markeredgecolor=soft,
                markeredgewidth=1.6,
                markersize=7,
                label="trainer fwd/bwd or step time (not end-to-end)",
            )
        ],
        loc="lower left",
        bbox_to_anchor=(0.62, 0.0),
        frameon=False,
        labelcolor=soft,
        fontsize=8,
    )
    runtime_ax.text(
        61.5,
        -17.5,
        "RL wall -3.6% not attributable (wait variance)",
        ha="right",
        va="center",
        color=mint,
        fontsize=7,
        fontfamily=mono,
    )
    runtime_ax.text(
        13.2,
        11.5,
        "TVM-FFI · REAL 298x1024x151936 HEAD · STRICT PARITY\nFrozen weights host submit CuTe/FFI 9.96/4.64 ms; events 15.41/15.33\nFull dW host submit 10.37/4.72 ms; events 31.61/31.57\nEvent intervals include queue gaps; host-submit difference is not GPU or RL speed.",
        ha="left",
        va="top",
        color=sky,
        fontsize=5.5,
        fontfamily=mono,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": bg, "edgecolor": sky, "linewidth": 0.8},
    )
    runtime_ax.text(
        len(stages) - 0.68,
        11.5,
        "LOCAL HEAD FIXTURES · 298 ROWS · STRICT PARITY\nForward-only T/Cu/Tri 25.075 / 20.720 / 20.342 ms\nTriton backward T/fwd/full 24.801 / 20.148 / 14.790 ms\nCuTe v2 full T/Triton/mixed/CuTe 24.751 / 14.692 / 16.555 / 15.192 ms\n8 samples/arm · CuTe 38.6% below Torch · local only\nTriton RL ABBA control/candidate 97.07 / 96.36 s (-0.7%)\nCuTe RL ABBA Torch/CuTe 96.65 / 95.98 s (-0.69%; within control spread)\nCuTe F/B all +7.42%, warm -4.22%; first-step cost\nNo attributable whole-RL gain",
        ha="right",
        va="top",
        color=lilac,
        fontsize=5.05,
        fontfamily=mono,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": bg, "edgecolor": lilac, "linewidth": 0.8},
    )
    runtime_ax.text(
        13.2,
        -2.5,
        "360M CAPACITY + FULL-RL EVIDENCE · DIFFERENT SCOPE\nStaged allocator ABBA 7197->6751 MiB; vLLM peaks 1283/1125 MiB in separate runs, not combined.\nCo-res 1/2 GiB staged probes: GPU 6316/7336 MiB, cgroup 7.47/6.24 GB, min host 2.35/3.50 GB; each 3 updates + 2 requests.\nFull RL r1 stopped after update 1 at 1.949 GB host; 362/366 no_signal, v1 incomplete, no checkpoint.\nPhase probes: F/B +496-559 MiB PSS, clip +80 MiB; trim -44 KiB. Diagnostic correlation only.\nTracemalloc: PSS +620 MiB, Python peak 4.32 MB; smaps heap +432 MiB and unnamed VMAs ~192 MiB, no owner.\nMemray: 893 MB requested heap excludes 9.23 GB virtual MMAP; top-100 heap has 466 MB library-load and 218 MB other CUDA labels; no app-leak attribution.\nPilot2 r2: 2/2 finite steps, 4/4 trainable episodes, v0-v2 transfers finished; three load HTTP 200s; 167.94 s. Min host 5.155 GB, cgroup 6.749 GB, GPU 7224 MiB.\nTwo-step admission pass only; no checkpoint. loss/std undefined for n=1. Host availability ~10 vs ~7 GiB; no code/performance gain claim. CuTe FFI 512-row H960/V49152: strict grad gate failed; no integration.\nBaseline10 RL: 10 finite steps, v10 HTTP 200; 337.52 s. Final step-10 save started, then systemd OOM at 7 GiB RAM + 2 GiB swap caps. Partial 2.126 GB distcp, no metadata, not resumable; min host 3.381 GB, GPU 7224 MiB. No full-RL speed claim.\nPTX no-FMA: strict N=512 pass; 0 BF16 mismatches across 25.166M dlogits, grad errors 0.000977/0.001953 within unchanged tolerances. 9.66 s, GPU 1175 MiB, cgroup 1.893 GB. Extended r1 hit int32 label-pointer specialization; r2 corrected labels dtype and passed 7 strict cases (Smol 512/511, Qwen 298/286, small masked/unmasked/int32, non-default stream). Integration pending.\nSerial DCP fake-data save and fresh resume passed: saved step 1 -> resumed step 2; 226 counters at 2 cover 361.8M FP32 parameters. Resume 17.84 s; cgroup 5.455 GB, GPU 6051 MiB, min host 4.844 GB; no new checkpoint. Actual-RL acceptance pending.\nRegistered FFI no-FMA full head, H960/V49152/N512: strict 7-case gate; 2 warmups + 6 alternating samples. Wall Torch/FFI 18.783/14.343 ms (-23.64%); events 18.774/14.333, enqueue 5.971/2.691 ms. Clones/autograd/head component only, not RL speed.",
        ha="left",
        va="top",
        color=yellow,
        fontsize=4.45,
        fontfamily=mono,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": bg, "edgecolor": yellow, "linewidth": 0.8},
    )
    runtime_ax.text(
        len(stages) - 6.7,
        -7.5,
        "LORA SHRINK · RANK8/K1024 · SPLIT-K 64->4\nQKV M1 23.024->18.452; M2 19.111->13.942\nM2 all-active 18.188->13.583 µs\nGate/up M1 21.525->19.020; M2 17.332->14.449\n100 replays/sample · component parity passed\nModel stress and stock repeat exceeded 1e-3",
        ha="right",
        va="top",
        color=sky,
        fontsize=6.2,
        fontfamily=mono,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": bg, "edgecolor": sky, "linewidth": 0.8},
    )

    gpu_ax.set_ylim(-900, 520)
    gpu_ax.set_yticks([-800, -400, 0, 400])
    gpu_ax.set_ylabel("MiB change", color=soft, labelpad=10)
    gpu_ax.set_title("Whole-GPU peak · same local ABBA comparisons", loc="left", color=ink, fontsize=12, pad=11)
    gpu_ax.axhline(0, color=yellow, linewidth=1.0, alpha=0.85)
    for stage, (_, change) in effects.items():
        color = stages[stage][2]
        gpu_ax.vlines(stage, 0, change, color=color, linewidth=1.6, alpha=0.85, zorder=2)
        gpu_ax.scatter(stage, change, marker="D", color=color, edgecolor=ink, linewidth=0.6, s=47, zorder=3)
        gpu_ax.annotate(
            f"{change:+.0f}",
            (stage, change),
            xytext=(0, 8 if change >= 0 else -10),
            textcoords="offset points",
            ha="center",
            va="bottom" if change >= 0 else "top",
            color=color,
            fontsize=8,
            fontfamily=mono,
        )

    tick_labels = [
        f"#{i}\n{textwrap.fill(name.replace(chr(10), ' '), width=10)}\n{textwrap.fill(status, width=10)}"
        for i, (name, status, _) in enumerate(stages)
    ]
    gpu_ax.set_xticks(x, tick_labels)
    for label, (_, _, color) in zip(gpu_ax.get_xticklabels(), stages):
        label.set_color(color)
        label.set_fontfamily(mono)
        label.set_fontsize(7.8)
    gpu_ax.tick_params(axis="x", pad=10)

    fig.text(
        0.075,
        0.985,
        "TINY RL  /  MEASURED OPTIMIZATION LOG",
        color=yellow,
        fontsize=9,
        fontfamily=mono,
        va="top",
    )
    fig.text(
        0.075,
        0.94,
        "tiny-prl / the whole hill climb",
        color=ink,
        fontsize=27,
        fontweight="normal",
        va="top",
    )
    fig.add_artist(plt.Line2D([0.075, 0.19], [0.89, 0.89], transform=fig.transFigure, color=yellow, linewidth=3))
    fig.text(
        0.075,
        0.185,
        "Baseline 115.1s includes startup/checkpoint. Time/GPU points are local B-vs-A means, 2 runs per arm; no cumulative score or confidence interval.",
        color=soft,
        fontsize=8.4,
        ha="left",
    )
    fig.text(
        0.075,
        0.142,
        "Memory scopes are not additive: import RSS, process USS, health PSS, RL cgroup anon+swap. GC+trim process anon fell 405 MiB at startup, 203 MiB after step 1. Host availability differed (about 11 vs 8.5 GiB); trim alone is not causal proof.",
        color=dim,
        fontsize=8.2,
        ha="left",
    )
    fig.text(
        0.075,
        0.096,
        "KV .25 stays for GPU headroom; full AC stays. Three verified archives replaced 8 older broadcasts; a fourth archives pilot step 2 and MFU step 20 with originals retained. Startup, step 10 and checkpoints stay raw. Later cleanup removed 183 finished-run broadcasts (7.21 GB / 6.71 GiB; manifest); UV/Triton cache cleanup tracked separately. Storage maintenance only.",
        color=dim,
        fontsize=7.6,
        ha="left",
    )
    fig.text(
        0.075,
        0.078,
        "Streaming writer ABBA: 273 tensors exact; post-write PSS 1,545 -> 1,353 MiB. HWM appears 136.73 MiB lower but stream value is setup-censored; phase 0.316925 -> 0.308159 s (-2.77%). No exact writer peak or end-to-end RL claim.",
        color=sky,
        fontsize=7.3,
        ha="left",
    )
    fig.text(
        0.075,
        0.059,
        "32-update acceptance: 32/32 finite · 3.52 GiB min RAM · 5908 MiB GPU · stability only  |  Earlier co-res control/trim both reached cap; trim reclaimed 904 KiB only and was not adopted.",
        color=mint,
        fontsize=6.7,
        ha="left",
    )
    fig.text(
        0.075,
        0.040,
        "Nsight 15 s partial: kernel-interval union vLLM 3.219 s / trainer 0.356 s; report-generation OOM at 6/10. Attribution only, not utilization or RL timing. NVML device-wide utilization (trainer+vLLM): 14.62% full loop, 16.90% warm 39.71 s; sampled values, not SM occupancy. Packed DCP parity covers QKV + tied embedding/head with two LR groups.",
        color=sky,
        fontsize=6.8,
        ha="left",
    )
    fig.text(
        0.075,
        0.020,
        "135M resume 11-20: 10/10 finite; weighted model-FLOP MFU 4.18% all / 6.24% warm 12-20; RTX 4060 peak 29.12256 TFLOPS (not A100); no occupancy claim.",
        color=yellow,
        fontsize=6.4,
        ha="left",
    )
    fig.text(
        0.075,
        0.004,
        "Exps 49–55: trainer-step metrics; no attributable end-to-end RL gain.",
        color=soft,
        fontsize=7.6,
        ha="left",
    )
    fig.text(
        0.985,
        0.020,
        "RTX 4060 Laptop · 8 GiB VRAM · 7 GiB cgroup guard · 2 GiB host-RAM reserve",
        color=dim,
        fontsize=8,
        ha="right",
        fontfamily=mono,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=140, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    main()
