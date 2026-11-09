"""
What the GPU stages spend their time on (report Section 5.5.3, Table 5.5).

Times the production calls process_images_fft_timed and
apply_frequency_filter_gpu_timed with the host clock (clock_gettime around
each phase, with clFinish after each) at several batch sizes. A straight-line
fit of time against batch size separates the fixed cost per dispatch (the
intercept) from the work per image (the slope).

It also times the same FFT on the CPU: NumPy on one thread, and the project's
OpenMP fallback path, to measure what offloading the FFT to the GPU gains.

Usage:
    python experiments/run_gpu_stage_timing.py

Output: results/gpu_stage_timing.csv
"""
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from main_mpi import configure_c_api, load_library  # noqa: E402

BATCHES = [16, 64, 256, 1024]
N_TRIALS = 30
N_DISCARD = 5
W = H = 64
C = 3


class Timing(ctypes.Structure):
    _fields_ = [(n, ctypes.c_double) for n in
                ("t_host_to_device", "t_kernel", "t_device_to_host", "t_total_gpu")]


def ptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def stage(fn, img):
    out = np.empty_like(img)
    rows = []
    for i in range(N_TRIALS):
        t = Timing()
        t0 = time.perf_counter()
        err = fn(ptr(img), len(img), W, H, C, ptr(out), ctypes.byref(t))
        call = time.perf_counter() - t0
        if err != 0:
            raise RuntimeError(f"kernel call failed: {err}")
        if i >= N_DISCARD:
            rows.append((t.t_host_to_device, t.t_kernel, t.t_device_to_host, call))
    return [1e3 * float(np.mean(col)) for col in zip(*rows)]


def wall(fn, repeats=10):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.mean(ts[2:]))


def main():
    lib = load_library()
    configure_c_api(lib)
    lib.init_opencl(0)
    rng = np.random.default_rng(0)
    rows = []
    for b in BATCHES:
        img = rng.random((b, H, W, C), dtype=np.float32)
        out = np.empty_like(img)
        fft = stage(lib.process_images_fft_timed, img)
        filt = stage(lib.apply_frequency_filter_gpu_timed, img)
        npy = wall(lambda: np.abs(np.fft.fft2(img, axes=(1, 2))))
        omp = wall(lambda: lib.process_images_fft_cpu(ptr(img), b, W, H, C, ptr(out)))
        rows.append((b, *fft, *filt, npy, omp))
        print(f"batch {b:5d}: FFT h2d {fft[0]:.3f} kernel {fft[1]:.3f} d2h {fft[2]:.3f} call {fft[3]:.3f} | "
              f"filter kernel {filt[1]:.3f} d2h {filt[2]:.3f} call {filt[3]:.3f} | "
              f"numpy {npy:.2f} openmp {omp:.2f} ms")

    with (ROOT / "results" / "gpu_stage_timing.csv").open("w") as f:
        f.write("batch,fft_h2d_ms,fft_kernel_ms,fft_d2h_ms,fft_call_ms,"
                "filter_h2d_ms,filter_kernel_ms,filter_d2h_ms,filter_call_ms,"
                "numpy_fft_ms,openmp_fft_ms\n")
        for r in rows:
            f.write(",".join(f"{x:.4f}" if i else str(x) for i, x in enumerate(r)) + "\n")

    a = np.array(rows, dtype=float)
    for name, col in (("FFT call", 4), ("filter call", 8)):
        slope, icept = np.polyfit(a[:, 0], a[:, col], 1)
        print(f"{name}: fixed {icept:.3f} ms per dispatch + {slope * 1e3:.2f} us per image; "
              f"at batch 256 fixed cost is {100 * icept / np.interp(256, a[:, 0], a[:, col]):.1f}%")
    print("Written to results/gpu_stage_timing.csv")


if __name__ == "__main__":
    main()
