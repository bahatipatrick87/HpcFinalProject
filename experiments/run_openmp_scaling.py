"""
OpenMP thread scaling experiment (1 MPI process, fixed image count).

Varies OMP_NUM_THREADS to isolate CPU-side parallelism in the feature-extraction
stage. Run from project root:

  python experiments/run_openmp_scaling.py [n_trials]

Output: results/openmp_scaling.csv, results/openmp_scaling_raw_trials.csv
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

# Fixed workload: 20,000 images (with 1 rank, this is the total; the first
# 20,000 files of the sorted list, all JPEGs), large enough that process and
# MPI start-up do not swamp the signal (report Table 5.6).
IMAGES_FOR_OPENMP_SCALING = int(os.environ.get("OPENMP_SCALING_IMAGES", "20000"))


def run_experiment(num_threads: int) -> float:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(num_threads)
    env["WEAK_SCALING_IMAGES_PER_RANK"] = str(IMAGES_FOR_OPENMP_SCALING)
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    python_exe = str(venv_python if venv_python.exists() else "python3")

    cmd = [
        "mpirun",
        "-n",
        "1",
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

    log_file = LOG_DIR / f"openmp_scaling_{num_threads}threads.log"
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
    # Report Table 5.6: 3 interleaved rounds over 1, 2, 4, 8 threads; every
    # timed run is preceded by an untimed run of the same thread count.
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    threads_list = [1, 2, 4, 8]
    raw = []
    for trial in range(1, n_trials + 1):
        for t in threads_list:
            run_experiment(t)  # untimed warm-up
            elapsed = run_experiment(t)
            raw.append((trial, t, elapsed))
            print(f"  trial {trial}  threads={t}: {elapsed:.4f} s")

    with (RESULTS_DIR / "openmp_scaling_raw_trials.csv").open("w") as f:
        f.write("trial,threads,elapsed_seconds\n")
        for trial, t, e in raw:
            f.write(f"{trial},{t},{e:.4f}\n")

    means = {t: statistics.mean(e for _, th, e in raw if th == t) for t in threads_list}
    t1 = means[1]
    csv_path = RESULTS_DIR / "openmp_scaling.csv"
    with csv_path.open("w") as f:
        f.write("threads,elapsed_seconds,speedup,throughput_img_per_sec\n")
        for t in threads_list:
            f.write(f"{t},{means[t]:.4f},{t1 / means[t]:.4f},"
                    f"{IMAGES_FOR_OPENMP_SCALING / means[t]:.2f}\n")
            print(f"threads={t}: {means[t]:.3f} s  speedup {t1 / means[t]:.3f}")

    print(f"\nOpenMP scaling results written to {csv_path}")


if __name__ == "__main__":
    main()
