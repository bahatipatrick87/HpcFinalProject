"""
Per-stage time breakdown of the production pipeline (report Table 5.7).

Wraps the stage functions used by the three-thread pipeline with timers
(no production code is changed), runs the full dataset under mpirun, and
reports, per rank, the busy time of each stage and the rank's wall time.
Stages overlap in pipeline mode, so busy times are not additive; the
table reports them for the critical-path (slowest) rank.

Usage:
    python experiments/run_stage_breakdown.py [n_ranks] [n_trials]

Output: results/stage_breakdown.csv (one row per trial per rank)
"""
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
STAGES = ["load", "fft", "filter", "feature"]


def worker():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    os.chdir(PROJECT_ROOT)
    import main_mpi
    from pipeline import batched_pipeline as bp
    from mpi4py import MPI

    busy = {s: 0.0 for s in STAGES}
    n_tif = [0]

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                busy[name] += time.perf_counter() - t0
        return wrapper

    orig_load = bp.load_images_from_paths

    def load_and_count(paths, *a, **k):
        n_tif[0] += sum(p.lower().endswith((".tif", ".tiff")) for p in paths)
        return orig_load(paths, *a, **k)

    bp.load_images_from_paths = timed("load", load_and_count)
    bp.fft_frequency_stage = timed("fft", bp.fft_frequency_stage)
    bp.filter_stage = timed("filter", bp.filter_stage)
    bp.feature_extraction_stage = timed("feature", bp.feature_extraction_stage)

    wall = [0.0]
    orig_run = main_mpi.run_batched_prefetch_pipeline

    def timed_run(*a, **k):
        t0 = time.time()
        orig_run(*a, **k)
        wall[0] = time.time() - t0
    main_mpi.run_batched_prefetch_pipeline = timed_run

    main_mpi.main()

    comm = MPI.COMM_WORLD
    rows = comm.gather((comm.Get_rank(), wall[0], n_tif[0], dict(busy)), root=0)
    if comm.Get_rank() == 0:
        for rank, w, tif, b in rows:
            print("STAGEROW", rank, f"{w:.4f}", tif,
                  *(f"{b[s]:.4f}" for s in STAGES), flush=True)


def main():
    n_ranks = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    python_exe = str(PROJECT_ROOT / "venv" / "bin" / "python")
    if not Path(python_exe).exists():
        python_exe = "python3"

    rows = []
    for trial in range(1, n_trials + 1):
        # untimed run first, so every trial starts from the same file-cache state
        subprocess.run(["mpirun", "-n", str(n_ranks), python_exe, __file__, "--worker"],
                       cwd=str(PROJECT_ROOT), capture_output=True)
        proc = subprocess.run(
            ["mpirun", "-n", str(n_ranks), python_exe, __file__, "--worker"],
            cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        trial_rows = [l.split()[1:] for l in proc.stdout.splitlines()
                      if l.startswith("STAGEROW")]
        if not trial_rows:
            raise RuntimeError(proc.stdout)
        for r in trial_rows:
            rows.append([trial, int(r[0]), float(r[1]), int(r[2])]
                        + [float(x) for x in r[3:]])
        crit = max((r for r in rows if r[0] == trial), key=lambda r: r[2])
        print(f"trial {trial}: slowest rank {crit[1]} wall={crit[2]:.2f}s "
              f"tif={crit[3]} " + " ".join(
                  f"{s}={v:.2f}" for s, v in zip(STAGES, crit[4:])))

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / "stage_breakdown.csv").open("w") as f:
        f.write("trial,rank,wall_s,geotiff_images," +
                ",".join(f"{s}_busy_s" for s in STAGES) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    crit = [max((r for r in rows if r[0] == t), key=lambda r: r[2])
            for t in range(1, n_trials + 1)]
    print("\nCritical-path rank, mean over trials:")
    print(f"  wall    : {st.mean(r[2] for r in crit):.3f} s")
    for i, s in enumerate(STAGES):
        print(f"  {s:8s}: {st.mean(r[4 + i] for r in crit):.3f} s busy")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        main()
