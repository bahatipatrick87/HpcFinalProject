"""
Strong scaling on the full EuroSAT dataset, P in {1, 2, 4, 8}.

Trials are interleaved (round r runs P=1, 2, 4, 8 once each), so a slow
drift in machine state affects every P alike. Every timed run is
immediately preceded by an untimed run of the same P, so it starts from
the same file-cache state. Run under experiments/hold_memory.py for the
report's numbers, so that the images are read from disk on every run.

Usage:
    python experiments/run_scaling_experiments.py [n_rounds]

Output:
    results/strong_scaling.csv             mean/std per P (report Table 5.1)
    results/strong_scaling_raw_trials.csv  every trial, incl. per-rank min/max
"""
import statistics as st
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RANKS = [1, 2, 4, 8]


def run_experiment(num_ranks: int) -> tuple[float, float, float, int]:
    """
    Run the MPI pipeline once; return (T_P, fastest rank, slowest rank,
    images) as printed by rank 0. T_P is the slowest rank's time.
    """
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    python_exe = str(venv_python if venv_python.exists() else "python3")
    cmd = ["mpirun", "-n", str(num_ranks), python_exe, "src/main_mpi.py"]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    (RESULTS_DIR / f"scaling_{num_ranks}ranks.log").write_text(proc.stdout)

    total = rmin = rmax = None
    images = 0
    for line in proc.stdout.splitlines():
        if line.startswith("Total Time:"):
            total = float(line.split()[2])
        elif line.startswith("Total Images Processed:"):
            images = int(line.split()[-1])
        elif line.startswith("Per-rank wall time"):
            # "Per-rank wall time [s]: min=1.59, max=4.36 (throughput uses max)"
            rmin = float(line.split("min=")[1].split(",")[0])
            rmax = float(line.split("max=")[1].split()[0])
    if total is None:
        raise RuntimeError(f"P={num_ranks}: no 'Total Time' in output:\n{proc.stdout}")
    if rmin is None:  # P=1 prints no per-rank line
        rmin = rmax = total
    return total, rmin, rmax, images


def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    raw = []
    for rnd in range(1, n_rounds + 1):
        for r in RANKS:
            run_experiment(r)  # untimed: settles the file cache for this P
            t, rmin, rmax, images = run_experiment(r)
            raw.append((rnd, r, t, rmin, rmax, images))
            print(f"  round {rnd}  P={r}: T={t:.3f} s  ranks min={rmin:.2f} max={rmax:.2f}")

    with (RESULTS_DIR / "strong_scaling_raw_trials.csv").open("w") as f:
        f.write("round,ranks,elapsed_seconds,rank_min_s,rank_max_s,images\n")
        for row in raw:
            f.write("{},{},{:.4f},{:.4f},{:.4f},{}\n".format(*row))

    means = {r: st.mean(t for _, p, t, *_ in raw if p == r) for r in RANKS}
    with (RESULTS_DIR / "strong_scaling.csv").open("w") as f:
        f.write("ranks,elapsed_mean_s,elapsed_std_s,rank_min_mean_s,speedup,"
                "efficiency_pct,throughput_img_per_sec\n")
        for r in RANKS:
            ts = [t for _, p, t, *_ in raw if p == r]
            mins = [m for _, p, _, m, *_ in raw if p == r]
            images = next(i for _, p, *_, i in raw if p == r)
            sd = st.stdev(ts) if len(ts) > 1 else 0.0
            s = means[1] / means[r]
            f.write(f"{r},{means[r]:.4f},{sd:.4f},{st.mean(mins):.4f},{s:.4f},"
                    f"{100 * s / r:.2f},{images / means[r]:.2f}\n")
            print(f"P={r}: T={means[r]:.3f} +- {sd:.3f} s  S={s:.2f}  E={100 * s / r:.1f}%")

    print(f"\nResults written to {RESULTS_DIR / 'strong_scaling.csv'}")


if __name__ == "__main__":
    main()
