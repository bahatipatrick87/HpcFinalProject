"""
OpenMP thread scaling of the CPU FFT path (report Section 5.6.1).

The pipeline sweep (run_openmp_scaling.py) shows no thread effect because on a
GPU machine the only OpenMP work on the hot path is Stage 4. This script runs
the OpenMP code that matters on a GPU-less machine: process_images_fft_cpu, the
same transform as the GPU kernel. Random images in memory, so no disk or file
cache is involved. Rounds interleave the thread counts.

Usage:
    python experiments/run_openmp_fft_scaling.py [n_rounds]

Output: results/openmp_fft_scaling.csv
"""
import ctypes
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THREADS = [1, 2, 4, 8, 10]
BATCH = 1024
REPEATS = 10


def worker():
    lib = ctypes.CDLL(str(ROOT / "lib" / "libpipeline_kernels.dylib"))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.process_images_fft_cpu.argtypes = [fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, fp]
    lib.process_images_fft_cpu.restype = ctypes.c_int
    data = np.random.default_rng(0).random((BATCH, 64, 64, 3), dtype=np.float32)
    out = np.empty_like(data)
    args = (data.ctypes.data_as(fp), BATCH, 64, 64, 3, out.ctypes.data_as(fp))
    lib.process_images_fft_cpu(*args)  # untimed: first touch of the buffers
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        if lib.process_images_fft_cpu(*args) != 0:
            raise RuntimeError("process_images_fft_cpu failed")
        times.append(time.perf_counter() - t0)
    print(st.mean(times) * 1e3)


def main():
    n_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    samples = {t: [] for t in THREADS}
    for rnd in range(n_rounds):
        for t in THREADS:
            env = dict(os.environ, OMP_NUM_THREADS=str(t))
            out = subprocess.run([sys.executable, __file__, "--worker"], env=env,
                                 capture_output=True, text=True, check=True).stdout
            samples[t].append(float(out.split()[-1]))
        print(f"round {rnd + 1}/{n_rounds}", {t: round(samples[t][-1], 1) for t in THREADS})
    base = st.mean(samples[1])
    with (ROOT / "results" / "openmp_fft_scaling.csv").open("w") as f:
        f.write("threads,time_ms,speedup,efficiency_pct,images_per_s\n")
        for t in THREADS:
            m = st.mean(samples[t])
            f.write(f"{t},{m:.3f},{base / m:.3f},{100 * base / m / t:.1f},{BATCH / m * 1e3:.0f}\n")
    print((ROOT / "results" / "openmp_fft_scaling.csv").read_text())


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        main()
