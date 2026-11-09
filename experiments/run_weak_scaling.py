"""
Weak scaling experiment (proposal 0.7): fixed per-rank workload, growing P.

Each rank processes a constant number of images (default 1000; override with
WEAK_SCALING_IMAGES_PER_RANK) while the number of MPI ranks increases, so the
total problem size grows proportionally with P. Run from project root:

  python experiments/run_weak_scaling.py [n_trials]

Output: results/weak_scaling.csv, results/weak_scaling_raw_trials.csv
"""
import os
import statistics
import sys
import time
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

IMAGES_PER_RANK = int(os.environ.get("WEAK_SCALING_IMAGES_PER_RANK", "1000"))


def run_experiment(num_ranks: int) -> float:
    env = os.environ.copy()
    env["WEAK_SCALING_IMAGES_PER_RANK"] = str(IMAGES_PER_RANK)
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    python_exe = str(venv_python if venv_python.exists() else "python3")

    cmd = [
        "mpirun",
        "-n",
        str(num_ranks),
        python_exe,
        "src/main_mpi.py",
    ]

    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    end = time.time()

    log_file = LOG_DIR / f"weak_scaling_{num_ranks}ranks.log"
    log_file.write_text(proc.stdout)

    elapsed = end - start
    for line in proc.stdout.splitlines():
        if "Total Time:" in line:
            try:
                parts = line.split()
                elapsed = float(parts[-2])
            except Exception:
                pass
            break
    return elapsed


def main():
    # Report Table 5.3: 4 interleaved rounds over P = 1, 2, 4, 8; every timed
    # run is preceded by an untimed run of the same P (warm OS file cache).
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ranks_list = [1, 2, 4, 8]
    raw = []
    for trial in range(1, n_trials + 1):
        for r in ranks_list:
            run_experiment(r)  # untimed warm-up
            elapsed = run_experiment(r)
            raw.append((trial, r, elapsed))
            print(f"  trial {trial}  P={r}: {elapsed:.4f} s "
                  f"({r * IMAGES_PER_RANK} images total)")

    with (RESULTS_DIR / "weak_scaling_raw_trials.csv").open("w") as f:
        f.write("trial,ranks,images_per_rank,total_images,elapsed_seconds\n")
        for trial, r, t in raw:
            f.write(f"{trial},{r},{IMAGES_PER_RANK},{r * IMAGES_PER_RANK},{t:.6f}\n")

    csv_path = RESULTS_DIR / "weak_scaling.csv"
    with csv_path.open("w") as f:
        f.write("ranks,images_per_rank,total_images,elapsed_seconds_mean,elapsed_seconds_std\n")
        for r in ranks_list:
            ts = [t for _, p, t in raw if p == r]
            sd = statistics.stdev(ts) if len(ts) > 1 else 0.0
            f.write(f"{r},{IMAGES_PER_RANK},{r * IMAGES_PER_RANK},"
                    f"{statistics.mean(ts):.4f},{sd:.4f}\n")
            print(f"P={r}: {statistics.mean(ts):.4f} +- {sd:.4f} s")

    print(f"\nWeak-scaling results written to {csv_path}")


if __name__ == "__main__":
    main()
