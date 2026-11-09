"""
Sequential baseline versus the hybrid pipeline, measured in one experiment
(report Table 5.10, Figure 5.7).

Each round times the single-threaded sequential baseline
(run_sequential_baseline.py, full dataset) and the hybrid pipeline at
P = 1, 2, 4, 8, one after the other, so every implementation sees the same
machine state. Every timed run is preceded by an untimed run of the same
configuration. Run under experiments/hold_memory.py for the report's numbers.

Usage:
    python experiments/run_comparative_benchmark.py [n_rounds]

Output: results/comparative_benchmark.csv (one row per round and implementation)
"""
import statistics as st
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from dataset_loader import _discover_files  # noqa: E402
from run_sequential_baseline import highpass_mask, run_once  # noqa: E402

RANKS = [1, 2, 4, 8]


def hybrid(num_ranks: int) -> float:
    python_exe = str(PROJECT_ROOT / "venv" / "bin" / "python")
    if not Path(python_exe).exists():
        python_exe = "python3"
    proc = subprocess.run(["mpirun", "-n", str(num_ranks), python_exe, "src/main_mpi.py"],
                          cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("Total Time:"):
            return float(line.split()[2])
    raise RuntimeError(proc.stdout)


def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    files = _discover_files(str(PROJECT_ROOT / "data" / "EuroSAT"))
    H = highpass_mask(64, 64)

    rows = []
    for rnd in range(1, n_rounds + 1):
        run_once(files, H)                      # untimed: settles the cache
        t, _ = run_once(files, H)
        rows.append((rnd, "sequential", t))
        print(f"round {rnd}  sequential : {t:.3f} s")
        for p in RANKS:
            hybrid(p)                           # untimed: settles the cache
            t = hybrid(p)
            rows.append((rnd, f"hybrid_P{p}", t))
            print(f"round {rnd}  hybrid P={p} : {t:.3f} s")

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / "comparative_benchmark.csv").open("w") as f:
        f.write("round,implementation,elapsed_seconds\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.4f}\n")

    seq = st.mean(t for _, impl, t in rows if impl == "sequential")
    print(f"\nsequential: {seq:.2f} s")
    for p in RANKS:
        ts = [t for _, impl, t in rows if impl == f"hybrid_P{p}"]
        print(f"hybrid P={p}: {st.mean(ts):.2f} +- {st.stdev(ts):.2f} s  "
              f"-> {seq / st.mean(ts):.2f}x vs sequential "
              f"(per-round range {min(r_s / r_h for r_s, r_h in per_round(rows, p)):.2f}"
              f"-{max(r_s / r_h for r_s, r_h in per_round(rows, p)):.2f}x)")


def per_round(rows, p):
    out = []
    for rnd in sorted({r[0] for r in rows}):
        s = next(t for r_, impl, t in rows if r_ == rnd and impl == "sequential")
        h = next(t for r_, impl, t in rows if r_ == rnd and impl == f"hybrid_P{p}")
        out.append((s, h))
    return out


if __name__ == "__main__":
    main()
