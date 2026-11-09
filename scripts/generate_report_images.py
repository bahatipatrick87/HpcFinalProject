"""
Generate all report images from the performance data tables.
Produces PNG files in report/images/ that match the tables in the LaTeX report.
The strong-scaling, Amdahl and comparative plots read the measured CSVs in
results/ (strong_scaling.csv, sequential_baseline.csv) directly.

Run from project root:
    python3 scripts/generate_report_images.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "report" / "images"
IMAGES.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv

RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "experiments"))
from analyze_amdahl import amdahl_speedup, fit_f  # noqa: E402  (single source for the fit)


def read_strong_scaling():
    """(P, mean T_P, std T_P) from results/strong_scaling.csv."""
    with (RESULTS / "strong_scaling.csv").open() as f:
        rows = list(csv.DictReader(f))
    return ([int(r["ranks"]) for r in rows],
            [float(r["elapsed_mean_s"]) for r in rows],
            [float(r["elapsed_std_s"]) for r in rows])


def read_comparative():
    """Mean time per implementation from results/comparative_benchmark.csv
    (sequential and hybrid P=1,2,4,8 timed in the same rounds)."""
    with (RESULTS / "comparative_benchmark.csv").open() as f:
        rows = list(csv.DictReader(f))
    order = ["sequential", "hybrid_P1", "hybrid_P2", "hybrid_P4", "hybrid_P8"]
    return [sum(float(r["elapsed_seconds"]) for r in rows if r["implementation"] == k)
            / sum(1 for r in rows if r["implementation"] == k) for k in order]


def read_csv_columns(name, *cols):
    """Columns of results/<name> as lists of floats, in file order."""
    with (RESULTS / name).open() as f:
        rows = list(csv.DictReader(f))
    return [[float(r[c]) for r in rows] for c in cols]


def read_sequential_mean():
    """Mean full-dataset time from results/sequential_baseline.csv."""
    with (RESULTS / "sequential_baseline.csv").open() as f:
        ts = [float(r["elapsed_seconds"]) for r in csv.DictReader(f)]
    return sum(ts) / len(ts)


# ---- colour palette ----
BLUE   = "#1f4e96"
ORANGE = "#e06c00"
GREEN  = "#1a7a40"
GRAY   = "#888888"
RED    = "#cc2200"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ================================================================
# 1.  Strong Scaling
# ================================================================
def plot_strong_scaling():
    P, T, T_std = read_strong_scaling()
    S   = [T[0] / t for t in T]           # speedup
    E   = [s / p * 100 for s, p in zip(S, P)]

    # Amdahl's Law fit: serial fraction f solved from the P=8 data point,
    # f = (P/S_P - 1) / (P - 1)
    f_amdahl = fit_f(8, S[3])
    P_smooth = np.linspace(1, 8, 200)
    S_amdahl = amdahl_speedup(f_amdahl, P_smooth)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.suptitle("Strong Scaling — EuroSAT (54,597 images)", fontweight="bold")

    # --- left: speedup ---
    ax = axes[0]
    ax.plot(P, P, "--", color=GRAY, lw=1.4, label="Ideal $S_P = P$")
    ax.plot(P_smooth, S_amdahl, "-.", color="#7a3ba0", lw=1.8,
            label=f"Amdahl ($f$={f_amdahl:.3f})")
    ax.plot(P, S, "o-", color=BLUE, lw=2.2, ms=8, label="Measured $S_P$")
    for xi, yi, ei in list(zip(P, S, E))[1:]:  # P=1 is 1.00x by definition
        ax.annotate(f"{yi:.2f}× ({ei:.0f}%)",
                    xy=(xi, yi), xytext=(4, -15 if xi < 8 else -22), textcoords="offset points",
                    fontsize=9, color=BLUE, ha="left" if xi < 8 else "right")
    ax.set_xlabel("MPI Processes $P$")
    ax.set_ylabel("Speedup $S_P = T_1 / T_P$")
    ax.set_title("Speedup")
    ax.set_xticks(P)
    ax.set_ylim(0, 8.5)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.fill_between(P, S, P, alpha=0.08, color=RED, label="Gap to ideal")

    # annotation explaining sublinear
    ax.annotate("Gap to ideal from P=4: uneven\nload between ranks (primary),\nshared GPU (secondary)",
                xy=(4, S[2]), xytext=(1.15, 4.4),
                arrowprops=dict(arrowstyle="->", color=RED),
                fontsize=8.5, color=RED)

    # --- right: execution time ---
    ax2 = axes[1]
    T_ideal = [T[0] / p for p in P]
    ax2.plot(P, T_ideal, "--", color=GRAY, lw=1.4, label="Ideal $T_1/P$")
    ax2.plot(P, T, "s-", color=ORANGE, lw=2.2, ms=8,
             label="Measured $T_P$ [s] (average of 5 runs)")
    for xi, yi in zip(P, T):
        ax2.annotate(f"{yi:.2f}s",
                     xy=(xi, yi), xytext=(3, 6), textcoords="offset points",
                     fontsize=9, color=ORANGE)
    ax2.set_xlabel("MPI Processes $P$")
    ax2.set_ylabel("Wall-clock time [s]")
    ax2.set_title("Execution Time")
    ax2.set_xticks(P)
    ax2.legend(fontsize=9)

    fig.tight_layout()
    path = IMAGES / "strong_scaling_optimized.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# 2.  Weak Scaling
# ================================================================
def plot_weak_scaling():
    P     = [1, 2, 4, 8]
    T, T_std = read_csv_columns("weak_scaling.csv",
                                "elapsed_seconds_mean", "elapsed_seconds_std")
    E     = [round(T[0] / t * 100, 1) for t in T]   # E_weak = T1/Tp * 100

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.suptitle("Weak Scaling — 1,000 images per MPI rank", fontweight="bold")

    # --- left: wall time ---
    ax = axes[0]
    ax.axhline(T[0], linestyle="--", color=GRAY, lw=1.4, label=f"$T_1$ = {T[0]:.3f}s")
    ax.fill_between(P, T[0], max(T) * 1.15, alpha=0.08, color=RED,
                    label="Contended-resource zone")
    ax.plot(P, T, "o-", color=BLUE, lw=2.2, ms=8,
            label="Measured $T_P$ (average of 4 runs)")
    for xi, yi in zip(P, T):
        # P=1 label below the line, the rest above it, so they don't collide
        below = xi == P[0]
        ax.annotate(f"{yi:.3f}s",
                    xy=(xi, yi), xytext=(8, -16) if below else (-4, 10),
                    textcoords="offset points", ha="left" if below else "right",
                    fontsize=8.5, color=BLUE)
    ax.set_xlabel("MPI Processes $P$")
    ax.set_ylabel("Wall-clock time [s]")
    ax.set_title("Execution Time (constant load/rank)")
    ax.set_ylim(0.13, max(T) * 1.18)
    ax.set_xticks(P)
    ax.legend(fontsize=8.5, loc="upper left")

    # --- right: efficiency ---
    ax2 = axes[1]
    ax2.axhline(100, linestyle="--", color=GRAY, lw=1.4, label="100% (ideal)")
    ax2.plot(P, E, "D-", color=GREEN, lw=2.2, ms=8, label="$E_P^{\\mathrm{weak}}=T_1/T_P\\times100\\%$")
    for xi, yi in zip(P, E):
        ax2.annotate(f"{yi}%",
                     xy=(xi, yi), xytext=(3, 6 if xi != 2 else -14), textcoords="offset points",
                     fontsize=9, color=GREEN)
    ax2.set_xlabel("MPI Processes $P$")
    ax2.set_ylabel("Weak-scaling efficiency [%]")
    ax2.set_title("Efficiency $E_P^{\\mathrm{weak}}$")
    ax2.set_xticks(P)
    ax2.legend(fontsize=8.5)
    ax2.set_ylim(0, 120)

    fig.tight_layout()
    path = IMAGES / "weak_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# 3.  Parallel Patterns diagram
#     (replaces the old parallel_patterns.png)
# ================================================================
def plot_parallel_patterns():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    fig.suptitle("Parallel Patterns in the Hybrid Pipeline", fontweight="bold", y=1.01)

    # --- (A) Data Parallelism ---
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("(A) Data Parallelism — MPI", fontsize=12, pad=8)

    # Dataset bar
    ds = mpatches.FancyBboxPatch((0.5, 5.8), 9, 0.9, boxstyle="round,pad=0.1",
                                  fc="#d0e8ff", ec=BLUE, lw=1.5)
    ax.add_patch(ds)
    ax.text(5, 6.25, "Full Dataset (N=54,597 images)", ha="center", va="center",
            fontsize=9.5, fontweight="bold")

    # Rank boxes
    colours = ["#aed6f1", "#a9dfbf", "#f1948a"]
    labels  = ["Rank 0\n$D_0$", "Rank 1\n$D_1$", "Rank P-1\n$D_{P-1}$"]
    xs = [0.5, 2.8, 7.5]
    for x, col, lbl in zip(xs, colours, labels):
        r = mpatches.FancyBboxPatch((x, 3.6), 2.0, 1.8,
                                     boxstyle="round,pad=0.08", fc=col, ec="gray", lw=1.2)
        ax.add_patch(r)
        ax.text(x + 1, 4.5, lbl, ha="center", va="center", fontsize=8.5)
        # arrow from dataset to rank
        ax.annotate("", xy=(x + 1, 5.4), xytext=(x + 1, 5.8),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))

    # ... dots
    ax.text(6.15, 4.5, "···", ha="center", va="center", fontsize=18)

    ax.text(5, 3.0, "No MPI messages between ranks during compute.\n"
            "Only Bcast (paths) and Gather (features).",
            ha="center", va="center", fontsize=8.5, color=GRAY,
            style="italic")

    # --- (B) Task / Pipeline Parallelism ---
    ax2 = axes[1]
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 7); ax2.axis("off")
    ax2.set_title("(B) Pipeline / Task Parallelism — 3-Thread Queue", fontsize=12, pad=8)

    thread_ys   = [5.5, 3.9, 2.3]
    thread_lbls = ["Thread 1 (I/O)", "Thread 2 (GPU)", "Thread 3 (CPU)"]
    thread_cols = ["#aed6f1", "#f0a500", "#a9dfbf"]
    # Batch k must be loaded before the GPU can process it, and filtered
    # before features are extracted, so each lane starts one slot later.
    batch_lbls  = [
        ["Load k",   "Load k+1",   "Load k+2"],
        ["",         "FFT+Filt k", "FFT+Filt k+1"],
        ["",         "",           "Feature k"],
    ]
    xs = [1.0, 4.0, 7.0]
    widths = [2.5, 2.5, 2.5]

    for ti, (ty, tc, tlbl) in enumerate(zip(thread_ys, thread_cols, thread_lbls)):
        ax2.text(1.0, ty + 0.85, tlbl, ha="left", va="center", fontsize=8.5,
                 fontweight="bold")
        for bi, (bx, bw, blbl) in enumerate(zip(xs, widths, batch_lbls[ti])):
            if not blbl:
                continue
            r = mpatches.FancyBboxPatch((bx, ty - 0.35), bw, 0.9,
                                         boxstyle="round,pad=0.05",
                                         fc=tc, ec="gray", lw=1.0, alpha=0.85)
            ax2.add_patch(r)
            ax2.text(bx + bw/2, ty + 0.1, blbl, ha="center", va="center",
                     fontsize=7.5)

    # dashed "clock tick" lines
    for x in [3.75, 6.75, 9.75]:
        ax2.axvline(x, ymin=0.28, ymax=0.93, linestyle="--", color=GRAY, lw=0.9)

    # overlap bracket: the first slot in which all three threads are busy
    ax2.annotate("", xy=(9.75, 6.75), xytext=(6.75, 6.75),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
    ax2.text(8.25, 6.85, "All three stages overlap", ha="center", va="bottom",
             fontsize=8.5, color=RED, fontweight="bold")

    ax2.text(5, 1.2,
             "Successive batches overlap in wall-clock time:\n"
             "Load k+1 runs while GPU processes k, CPU processes k-1.",
             ha="center", va="center", fontsize=8.5, color=GRAY, style="italic")

    fig.tight_layout()
    path = IMAGES / "parallel_patterns.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# 4.  Mapping Hierarchy diagram
# ================================================================
def plot_mapping_hierarchy():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 15.2); ax.set_ylim(0, 8); ax.axis("off")
    ax.set_title("Hierarchical Hardware Mapping (Apple M4 SoC)",
                 fontweight="bold", fontsize=13)

    def box(x, y, w, h, label, fc, ec="gray", fs=9.5, bold=False):
        r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                     fc=fc, ec=ec, lw=1.4)
        ax.add_patch(r)
        fw = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, fontweight=fw)

    # SoC
    r = mpatches.FancyBboxPatch((0.3, 0.3), 11.4, 7.2,
                                  boxstyle="round,pad=0.15",
                                  fc="#f5f5f5", ec="#555555", lw=2)
    ax.add_patch(r)
    ax.text(6, 7.25, "Apple M4 SoC — unified memory (16 GB, 120 GB/s)",
            ha="center", va="center", fontsize=9, color="#555555", style="italic")

    # Level 1: MPI ranks (OS processes, each with 3 pipeline threads)
    rank_cols = ["#aed6f1", "#a9dfbf", "#f9e79f", "#f1948a"]
    rank_xs   = [1.2, 3.6, 6.0, 8.4]
    for rx, rc, ri in zip(rank_xs, rank_cols, range(4)):
        box(rx, 5.15, 2.1, 1.45, f"MPI Rank {ri}\n(OS process,\n3 pipeline threads)", rc, fs=8)
    ax.text(10.95, 5.85, "···", ha="center", va="center", fontsize=14)

    # Level 2: every rank opens its own OpenCL context on the one GPU
    for rx in rank_xs:
        mid = rx + 1.05
        ax.annotate("", xy=(mid, 4.25), xytext=(mid, 5.1),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    box(1.2, 3.35, 9.3, 0.85,
        "Shared GPU — 1 integrated device, one OpenCL context per rank\n"
        "(device = rank % n_gpus); work-items over rows, columns, elements",
        "#ffe8c0", ec=ORANGE, fs=8.5, bold=True)

    # Level 3: the OS schedules every rank's threads onto the CPU cores
    ax.plot([1.08, 0.75, 0.75], [5.85, 5.85, 1.7], color=BLUE, lw=1.4, ls="--")
    ax.annotate("", xy=(1.08, 1.7), xytext=(0.75, 1.7),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.text(0.55, 3.8, "OS scheduler\n(no pinning)", ha="center", va="center",
            fontsize=7.5, color=BLUE, rotation=90)
    cpu_xs = [1.2, 2.75, 4.3, 5.85, 7.4]
    cpu_lbls = ["P-core", "P-core", "P-core", "P-core", "6 × E-core"]
    for cx, cl in zip(cpu_xs, cpu_lbls):
        box(cx, 1.05, 1.35, 1.3, cl, "#d5f5e3", fs=8.5)
    ax.text(9.95, 1.7, "pipeline threads +\nOpenMP threads (S4)", ha="center",
            va="center", fontsize=8, color="#333333")
    ax.text(6, 0.55, "CPU cores: 4 performance + 6 efficiency",
            ha="center", va="center", fontsize=8.5, color=GRAY)

    # Level labels, outside the SoC box
    for lbl, y in [("Level 1: MPI ranks → OS processes", 5.85),
                   ("Level 2: ranks → OpenCL contexts\non the shared GPU", 4.65),
                   ("Level 4: OpenCL work-items →\nGPU execution units", 3.3),
                   ("Level 3: pipeline + OpenMP\nthreads → CPU cores", 1.7)]:
        ax.text(12.0, y, lbl, ha="left", va="center", fontsize=8.5,
                color="#333333", style="italic")

    fig.tight_layout()
    path = IMAGES / "mapping_hierarchy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# 5.  System Architecture overview
# ================================================================
def plot_system_architecture():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0.6, 7.7); ax.axis("off")
    ax.set_title("Hybrid MPI + OpenCL + OpenMP Pipeline Architecture",
                 fontweight="bold", fontsize=13)

    def box(x, y, w, h, label, fc, ec="gray", fs=9.0, bold=False):
        r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                     fc=fc, ec=ec, lw=1.5)
        ax.add_patch(r)
        fw = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, wrap=True)

    def arrow(x1, y1, x2, y2, lbl="", col="gray"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        if lbl:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.1, my, lbl, fontsize=7.5, color=col)

    # User / YAML config
    box(4.5, 6.75, 3.0, 0.65, "YAML Config\n(user-customisable workflow)", "#fffde7",
        ec="#bfa000", fs=8)

    # Python orchestrator
    box(3.5, 5.3, 5.0, 0.9, "Python Orchestrator  (main_mpi.py)\nDAG builder · MPI init · timing",
        "#e3f2fd", ec=BLUE, fs=8.5, bold=True)
    arrow(6, 6.63, 6, 6.32, col="#bfa000")

    # MPI layer
    box(0.4, 3.95, 11.2, 0.8,
        "MPI (mpi4py)  —  Data Parallelism: Bcast paths · Ranks process disjoint subsets · Gather features\n"
        "every rank runs the four stages below",
        "#e8f5e9", ec=GREEN, fs=8.5)
    arrow(6, 5.18, 6, 4.87, col=BLUE)

    # 4 stages
    stage_x  = [0.6, 3.4, 6.2, 9.0]
    stage_lbl= ["S1: Load & Preprocess\n(CPU, I/O)\nThread 1",
                 "S2: FFT Magnitude\n(GPU/OpenCL)\nThread 2",
                 "S3: High-Pass Filter\n(GPU/OpenCL)\nThread 2",
                 "S4: Feature Extract\n(CPU/OpenMP)\nThread 3"]
    stage_fc = ["#e3f2fd", "#fff3e0", "#fff3e0", "#e8f5e9"]
    stage_ec = [BLUE, ORANGE, ORANGE, GREEN]
    for sx, sl, sf, se in zip(stage_x, stage_lbl, stage_fc, stage_ec):
        box(sx, 1.8, 2.2, 1.25, sl, sf, ec=se, fs=8.0)
        arrow(sx + 1.1, 3.83, sx + 1.1, 3.17, col=se)

    # Pipeline arrows between stages
    for sx in stage_x[:-1]:
        arrow(sx + 2.32, 2.42, sx + 2.68, 2.42, col=GRAY)

    # 3-thread pipeline bracket
    r = mpatches.FancyBboxPatch((0.32, 1.5), 11.36, 1.85,
                                  boxstyle="round,pad=0.05",
                                  fc="none", ec=RED, lw=1.5, linestyle="dashed")
    ax.add_patch(r)
    ax.text(6.0, 1.1,
            "3-Thread Intra-Rank Pipeline: Thread1(Load) ∥ Thread2(GPU) ∥ Thread3(CPU)  — task parallelism",
            ha="center", va="center", fontsize=8.0, color=RED, fontweight="bold")

    fig.tight_layout()
    path = IMAGES / "system_architecture.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# 6.  GPU Concurrency Experiment (compact, single-panel)
# ================================================================
def plot_gpu_concurrency():
    P, thr = read_csv_columns("gpu_scaling.csv", "n_ranks", "throughput_img_s")
    P, thr = [int(p) for p in P], [round(t) for t in thr]
    peak = thr.index(max(thr))
    colors = [ORANGE if i == peak else BLUE for i in range(len(P))]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.bar([str(p) for p in P], thr, color=colors, width=0.55)
    for b, v in zip(bars, thr):
        ax.annotate(f"{v:,}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=11.5)
    top = 0.9 * max(thr)
    ax.plot([1.7, 1.7, 3.3, 3.3], [top - 0.03 * max(thr), top, top, top - 0.03 * max(thr)],
            color=RED, lw=1.2)
    ax.text(2.5, top + 0.02 * max(thr), "GeoTIFF ranks; disk reads at $P=8$", ha="center",
            va="bottom", fontsize=10, color=RED)
    ax.set_xlabel("MPI Processes $P$", fontsize=12.5)
    ax.set_ylabel("Throughput [img/s]", fontsize=12.5)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.set_title("GPU Concurrency — 5,000 images/rank, mean of 4 sweeps",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(thr) * 1.2)
    fig.tight_layout()
    path = IMAGES / "gpu_concurrency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# 7.  OpenMP Thread Scaling (compact, single-panel)
# ================================================================

# ================================================================
# 8.  Comparative Benchmarking: Sequential vs Hybrid (compact)
# ================================================================

# ================================================================
# 9.  Amdahl's Law: sensitivity of the ceiling to the fitted f
# ================================================================


# ================================================================
# 7.  OpenMP Thread Scaling (compact, single-panel)
# ================================================================
def plot_openmp_scaling():
    threads, thr = read_csv_columns("openmp_scaling.csv", "threads",
                                    "throughput_img_per_sec")
    threads, thr = [int(t) for t in threads], [round(t) for t in thr]
    spread = 100 * (max(thr) - min(thr)) / max(thr)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.bar([str(t) for t in threads], thr, color=GREEN, width=0.55)
    for b, v in zip(bars, thr):
        ax.annotate(f"{v:,}", xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=11.5)
    mean_thr = sum(thr) / len(thr)
    ax.axhline(mean_thr, color=GRAY, lw=1.2, ls="--")
    ax.annotate(f"varies by {spread:.1f}% — no measurable effect",
                xy=(2.5, mean_thr), xytext=(2.5, mean_thr * 1.28),
                ha="center", fontsize=11, color="#333333")
    ax.set_xlabel("OMP_NUM_THREADS", fontsize=12.5)
    ax.set_ylabel("Throughput [img/s]", fontsize=12.5)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.set_title("OpenMP Thread Scaling — $P=1$, 20,000 images",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(thr) * 1.45)
    fig.tight_layout()
    path = IMAGES / "openmp_scaling.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# 8.  Comparative Benchmarking: Sequential vs Hybrid (compact)
# ================================================================
def plot_comparative_benchmark():
    labels = ["Sequential", "Hybrid\n$P{=}1$", "Hybrid\n$P{=}2$",
              "Hybrid\n$P{=}4$", "Hybrid\n$P{=}8$"]
    T = read_comparative()
    colors = [GRAY, BLUE, BLUE, BLUE, BLUE]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bars = ax.bar(labels, T, color=colors, width=0.55)
    for i, (b, v) in enumerate(zip(bars, T)):
        lbl = f"{v:.2f}s" if i == 0 else f"{v:.2f}s\n{T[0] / v:.2f}× faster"
        ax.annotate(lbl, xy=(b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=11, color=GRAY if i == 0 else BLUE)
    ax.set_ylabel("Wall-clock time [s]", fontsize=12.5)
    ax.tick_params(axis="both", labelsize=11.5)
    ax.set_title("Sequential Baseline vs. Hybrid Pipeline ($N=54{,}597$)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(T) * 1.25)
    fig.tight_layout()
    path = IMAGES / "comparative_benchmark.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ================================================================
# 9.  Amdahl's Law: sensitivity of the ceiling to the fitted f
# ================================================================
def plot_amdahl_sensitivity():
    P_tested, T_tested, _ = read_strong_scaling()
    S_tested = [T_tested[0] / t for t in T_tested]

    f_fit = fit_f(8, S_tested[3])             # fitted from measured P=8 point
    f_low, f_high = f_fit - 0.06, f_fit + 0.06  # +/- sensitivity bracket

    P_max = 32
    P_curve = np.linspace(1, P_max, 400)

    fig, ax = plt.subplots(figsize=(9.0, 4.1))

    amdahl = amdahl_speedup

    # Sensitivity brackets (hypothetical, NOT separate measurements)
    ax.plot(P_curve, amdahl(f_low, P_curve), "--", color="#1a7a40", lw=2.2,
            label=f"$f$={f_low:.3f} (sensitivity, optimistic)")
    ax.plot(P_curve, amdahl(f_high, P_curve), "--", color="#cc2200", lw=2.2,
            label=f"$f$={f_high:.3f} (sensitivity, conservative)")

    # Our real fitted curve, visually distinct (thicker, dash-dot, matches
    # the colour used in Figures 5.2/5.3 for the same fitted curve)
    ax.plot(P_curve, amdahl(f_fit, P_curve), "-.", color="#7a3ba0", lw=3.0,
            label=f"$f$={f_fit:.3f} (fitted from measured data)", zorder=4)

    # Ceiling markers for each curve
    for f, color in [(f_low, "#1a7a40"), (f_fit, "#7a3ba0"), (f_high, "#cc2200")]:
        ceil = 1.0 / f
        ax.axhline(ceil, color=color, lw=1.1, ls=":", alpha=0.6)
        ax.text(P_max + 0.8, ceil + 0.05, f"{ceil:.2f}×", va="bottom", ha="left",
                fontsize=11, color=color, fontweight="bold")

    # Measured data anchors the whole plot to reality
    ax.plot(P_tested, S_tested, "o-", color=BLUE, lw=2.8, ms=11,
            label="Measured $S_P$", zorder=5)
    for xi, yi in list(zip(P_tested, S_tested))[1:]:  # P=1 is 1.00x by definition
        ax.annotate(f"{yi:.2f}×", xy=(xi, yi), xytext=(8, -16),
                    textcoords="offset points", ha="left",
                    fontsize=11, color=BLUE, fontweight="bold", zorder=6,
                    bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))

    ax.set_xlabel("MPI Processes $P$", fontsize=15)
    ax.set_ylabel("Speedup $S_P$", fontsize=15)
    ax.tick_params(axis="both", labelsize=13.5)
    ax.set_xlim(0, P_max + 5)
    ax.set_ylim(0, 1.0 / f_low * 1.15)
    ax.set_xticks([1, 4, 8, 16, 24, 32])
    ax.set_title("Amdahl's Law: Ceiling Sensitivity Around the Fitted $f$",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=2, frameon=True)

    fig.tight_layout()
    path = IMAGES / "amdahl_sensitivity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == "__main__":
    print("Generating report images ...")
    plot_strong_scaling()
    plot_weak_scaling()
    plot_parallel_patterns()
    plot_mapping_hierarchy()
    plot_system_architecture()
    plot_gpu_concurrency()
    plot_openmp_scaling()
    plot_comparative_benchmark()
    plot_amdahl_sensitivity()
    print("Done. All images written to", IMAGES)
