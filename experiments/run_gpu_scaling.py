"""
GPU Scaling Experiment
=======================
On a single-GPU system (Apple Silicon or otherwise), full multi-GPU variation
is not feasible. This script instead:

  1. Reports detected OpenCL GPU device(s) and their names.
  2. Sweeps the number of MPI ranks (1..max_ranks), which exercise the GPU
     under varying degrees of concurrent demand, and records throughput.
  3. Reports per-rank and aggregate throughput to quantify the sharing effect.
  4. Records results to results/gpu_scaling.csv.

This is the GPU contention experiment of report Section 5.5; it sweeps concurrent
demand on the single GPU because the GPU count cannot be varied on this testbed.

Usage (the report's table is the mean of four sweeps, one reversed):
    python3 experiments/run_gpu_scaling.py            # forward sweep 1,2,4,8
    python3 experiments/run_gpu_scaling.py --reverse  # reverse sweep 8,4,2,1

Each run appends one sweep to results/gpu_scaling_raw_sweeps.csv and
rewrites results/gpu_scaling.csv as the mean over all recorded sweeps.
"""
import os
import sys
import subprocess
import time
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _detect_gpu_info():
    """Return (n_gpus, name_str) by calling the shared library."""
    try:
        import ctypes
        ext  = ".dylib" if sys.platform == "darwin" else ".so"
        lib  = ctypes.CDLL(str(ROOT / "lib" / f"libpipeline_kernels{ext}"))
        lib.query_gpu_device_count.argtypes = []
        lib.query_gpu_device_count.restype  = ctypes.c_int
        lib.query_gpu_device_name.argtypes  = [ctypes.c_int,
                                                ctypes.c_char_p, ctypes.c_int]
        lib.query_gpu_device_name.restype   = ctypes.c_int
        lib.init_opencl(0)
        n = lib.query_gpu_device_count()
        names = []
        for i in range(max(n, 1)):
            buf = ctypes.create_string_buffer(256)
            lib.query_gpu_device_name(i, buf, 256)
            names.append(buf.value.decode(errors="replace"))
        return n, names
    except Exception as e:
        return 0, [f"(detection failed: {e})"]


def _run_mpi(n_ranks: int, max_images: int) -> float:
    """Run pipeline with n_ranks MPI processes and return throughput img/s."""
    venv_py = ROOT / "venv" / "bin" / "python"
    py = str(venv_py if venv_py.exists() else sys.executable)

    env = os.environ.copy()
    env["HPC_MAX_IMAGES_PER_RANK"] = str(max_images)
    env["TASK_PARALLEL_PREFETCH"] = "1"

    cmd = ["mpirun", "-n", str(n_ranks), py, str(ROOT / "src" / "main_mpi.py")]
    t0  = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env)
    wall = time.perf_counter() - t0

    log_dir = ROOT / "results" / "gpu_scaling_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"ranks_{n_ranks}.log").write_text(res.stdout)

    total_time   = wall
    total_images = n_ranks * max_images

    for line in res.stdout.splitlines():
        if "Total Time:" in line:
            try:
                total_time = float(line.split()[-2])
            except ValueError:
                pass
        if "Total Images Processed:" in line:
            try:
                total_images = int(line.split()[-1])
            except ValueError:
                pass

    return total_images / total_time if total_time > 0 else 0.0


def main():
    n_gpus, gpu_names = _detect_gpu_info()
    print("=" * 64)
    print("GPU Scaling Experiment")
    print("=" * 64)
    print(f"Detected OpenCL GPU devices : {n_gpus}")
    for i, name in enumerate(gpu_names):
        print(f"  GPU[{i}]: {name}")
    if n_gpus <= 1:
        print("\nNote: Only 1 physical GPU available (Apple Silicon SoC).")
        print("Experiment varies MPI rank count to measure GPU contention,")
        print("which documents how shared-GPU throughput saturates.")
    print()

    reverse = "--reverse" in sys.argv
    ranks_list   = [8, 4, 2, 1] if reverse else [1, 2, 4, 8]
    images_p_rank = 5000  # large enough that per-process/OpenCL warm-up
                            # overhead no longer dominates the measurement
                            # (500 images/rank was found to be too short to
                            # give a stable signal: run-to-run throughput
                            # varied by up to 3x at fixed P).
    n_trials = 3            # first invocation of each P is discarded as a
                            # warm-up (kernel-cache / driver warm-up cost);
                            # the remaining trials are averaged.

    results = []
    for nr in ranks_list:
        trial_throughputs = []
        print(f"Running {nr} MPI rank(s) × {images_p_rank} images/rank "
              f"({n_trials + 1} trials, first discarded as warm-up):")
        for t in range(n_trials + 1):
            thr = _run_mpi(nr, images_p_rank)
            tag = "warm-up" if t == 0 else f"trial {t}"
            print(f"    [{tag}]  {thr:.1f} img/s")
            if t > 0:
                trial_throughputs.append(thr)
        avg_thr = sum(trial_throughputs) / len(trial_throughputs)
        print(f"  -> mean of {n_trials} measured trials: {avg_thr:.1f} img/s")
        results.append({"n_ranks": nr,
                         "images_per_rank": images_p_rank,
                         "total_images": nr * images_p_rank,
                         "throughput_img_s": avg_thr,
                         "n_gpus": n_gpus,
                         "gpu_name": gpu_names[0] if gpu_names else "N/A"})

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    # Append this sweep to the raw file, then average every recorded sweep.
    raw_path = out_dir / "gpu_scaling_raw_sweeps.csv"
    sweeps = []
    if raw_path.exists():
        with raw_path.open() as f:
            sweeps = list(csv.DictReader(f))
    sweep_id = 1 + max((int(r["sweep"]) for r in sweeps), default=0)
    note = "reverse order 8,4,2,1" if reverse else "forward order 1,2,4,8"
    for r in results:
        sweeps.append({"n_ranks": str(r["n_ranks"]), "sweep": str(sweep_id),
                       "throughput_img_s": f"{r['throughput_img_s']:.1f}",
                       "note": note})
    with raw_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_ranks", "sweep", "throughput_img_s", "note"])
        w.writeheader()
        w.writerows(sorted(sweeps, key=lambda r: (int(r["n_ranks"]), int(r["sweep"]))))

    results.sort(key=lambda r: r["n_ranks"])
    for r in results:
        vals = [float(x["throughput_img_s"]) for x in sweeps
                if int(x["n_ranks"]) == r["n_ranks"]]
        r["throughput_img_s"] = round(sum(vals) / len(vals), 1)

    csv_path = out_dir / "gpu_scaling.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\nResults written to {csv_path}")
    print("\nSummary:")
    print(f"{'Ranks':>6}  {'Throughput (img/s)':>20}  {'Speedup vs 1':>14}")
    base = results[0]["throughput_img_s"] if results else 1.0
    for r in results:
        su = r["throughput_img_s"] / base if base > 0 else 0
        print(f"{r['n_ranks']:>6}  {r['throughput_img_s']:>20.1f}  {su:>14.2f}x")

    print("\nInterpretation:")
    print("  All MPI ranks share the single integrated GPU.")
    print("  Both ranks process JPEGs at P=2; from P=4 later ranks receive the")
    print("  costlier GeoTIFFs and the slowest rank sets the time (report §5.5).")
    print("  On a multi-GPU cluster, each rank would own an exclusive GPU, which")
    print("  removes the device contention but not the load imbalance.")


if __name__ == "__main__":
    main()
