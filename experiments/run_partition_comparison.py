"""
Contiguous (production) versus cyclic data partitioning, full dataset.

The production split (dataset_loader.get_partitioned_paths) gives each rank
a contiguous block of the file list, which holds 14,000 JPEGs, then all
27,597 GeoTIFFs, then 13,000 JPEGs, so at most P the outer ranks get JPEGs
and the middle ranks GeoTIFFs. This script reruns the production
pipeline with the split replaced, from outside, by a cyclic one
(all_files[rank::size]) and compares per-rank times. Production code is
not modified. Trials alternate contiguous/cyclic, and each timed run
is preceded by an untimed run of the same configuration.

Usage:
    python experiments/run_partition_comparison.py [n_trials]

Output: results/partition_comparison.csv
"""
import statistics as st
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RANKS = [4, 8]


def worker(mode: str):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import main_mpi
    from pipeline import batched_pipeline as bp
    import dataset_loader

    if mode == "cyclic":
        def cyclic_paths(data_dir, rank, size, comm=None, max_images=None):
            files = []
            if rank == 0:
                files = dataset_loader._discover_files(data_dir)
            if comm is not None:
                files = comm.bcast(files, root=0)
            return files[rank::size]
        bp.get_partitioned_paths = cyclic_paths

    main_mpi.main()


def run(n_ranks: int, mode: str):
    python_exe = str(PROJECT_ROOT / "venv" / "bin" / "python")
    if not Path(python_exe).exists():
        python_exe = "python3"
    proc = subprocess.run(
        ["mpirun", "-n", str(n_ranks), python_exe, __file__, "--worker", mode],
        cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    total = rmin = None
    images = 0
    for line in proc.stdout.splitlines():
        if line.startswith("Total Time:"):
            total = float(line.split()[2])
        elif line.startswith("Total Images Processed:"):
            images = int(line.split()[-1])
        elif line.startswith("Per-rank wall time"):
            rmin = float(line.split("min=")[1].split(",")[0])
    if total is None:
        raise RuntimeError(proc.stdout)
    return total, rmin, images


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rows = []
    for p in RANKS:
        for trial in range(1, n_trials + 1):
            for mode in ("contiguous", "cyclic"):
                run(p, mode)  # untimed: settles the file cache
                t, rmin, images = run(p, mode)
                rows.append((p, mode, trial, t, rmin, images))
                print(f"P={p} {mode:10s} trial {trial}: T={t:.3f}s "
                      f"fastest rank {rmin:.3f}s  images={images}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / "partition_comparison.csv").open("w") as f:
        f.write("ranks,partition,trial,elapsed_seconds,rank_min_s,images\n")
        for r in rows:
            f.write("{},{},{},{:.4f},{:.4f},{}\n".format(*r))

    print()
    for p in RANKS:
        for mode in ("contiguous", "cyclic"):
            ts = [r[3] for r in rows if r[0] == p and r[1] == mode]
            ms = [r[4] for r in rows if r[0] == p and r[1] == mode]
            print(f"P={p} {mode:10s}: T={st.mean(ts):.3f} +- {st.stdev(ts):.3f}s  "
                  f"fastest rank {st.mean(ms):.3f}s")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(sys.argv[sys.argv.index("--worker") + 1])
    else:
        main()
