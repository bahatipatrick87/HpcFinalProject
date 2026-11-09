"""
Decompose one GPU-stage dispatch into: buffer/H2D transfer, kernel
execution, and D2H transfer, using OpenCL profiling events for every step,
next to the host-clock time of the same kernel. Batch size matches Table 5.8
(batch size 64). The driver timestamps are far below the host clock on the
testbed, so they are evidence of execution only, not timings (report §5.5.3).

Usage:
    python3 experiments/verify_gpu_breakdown.py
"""
import re
import time
from pathlib import Path

import numpy as np
import pyopencl as cl

def production_kernel_source() -> str:
    """The kernel source compiled into the C++ library, read from its file."""
    cpp = (Path(__file__).resolve().parents[1] / "src" / "pipeline_kernels.cpp").read_text()
    return re.search(r'R"CL\((.*?)\)CL"', cpp, re.S).group(1)


KERNEL_SRC = production_kernel_source()

WIDTH = HEIGHT = 64
CHANNELS = 3
BATCH_SIZE = 64  # matches Table 5.8
N_TRIALS = 20


def ns(evt):
    return evt.profile.end - evt.profile.start


def main():
    dev = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)[0]
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    prg = cl.Program(ctx, KERNEL_SRC).build()
    kernel = cl.Kernel(prg, "fft_magnitude")

    n_elem = BATCH_SIZE * WIDTH * HEIGHT * CHANNELS
    host_in = np.random.rand(n_elem).astype(np.float32)
    mf = cl.mem_flags

    def fft_pass(d_in, d_scr, d_out, n_pass):
        lines = HEIGHT if n_pass == 0 else WIDTH
        return kernel(queue, (BATCH_SIZE * CHANNELS * lines,), None, d_in, d_scr, d_out,
                      np.int32(WIDTH), np.int32(HEIGHT), np.int32(CHANNELS), np.int32(n_pass))

    h2d_ns, kern_ns, d2h_ns, wall_kern_ms = [], [], [], []
    for _ in range(N_TRIALS):
        d_in = cl.Buffer(ctx, mf.READ_ONLY, size=host_in.nbytes)
        d_scr = cl.Buffer(ctx, mf.READ_WRITE, size=2 * host_in.nbytes)
        d_out = cl.Buffer(ctx, mf.WRITE_ONLY, size=host_in.nbytes)

        evt_h2d = cl.enqueue_copy(queue, d_in, host_in, is_blocking=True)
        queue.finish()
        t0 = time.perf_counter()
        evt_k0 = fft_pass(d_in, d_scr, d_out, 0)
        evt_k1 = fft_pass(d_in, d_scr, d_out, 1)
        queue.finish()
        wall_kern_ms.append((time.perf_counter() - t0) * 1e3)
        host_out = np.empty_like(host_in)
        evt_d2h = cl.enqueue_copy(queue, host_out, d_out, is_blocking=True)
        queue.finish()

        h2d_ns.append(ns(evt_h2d))
        kern_ns.append(ns(evt_k0) + ns(evt_k1))
        d2h_ns.append(ns(evt_d2h))

    def summarize(name, vals):
        vals_ms = [v / 1e6 for v in vals]
        print(f"  {name:22s}: mean={sum(vals_ms)/len(vals_ms):.4f} ms  "
              f"min={min(vals_ms):.4f}  max={max(vals_ms):.4f}")

    print("=" * 64)
    print(f"GPU dispatch breakdown, batch={BATCH_SIZE} images "
          f"({WIDTH}x{HEIGHT}x{CHANNELS}), {N_TRIALS} trials, "
          "device-clock timestamps")
    print("=" * 64)
    summarize("H2D transfer", h2d_ns)
    summarize("Kernel execution", kern_ns)
    summarize("D2H transfer", d2h_ns)
    print(f"  {'Kernel, host clock':22s}: mean={sum(wall_kern_ms)/len(wall_kern_ms):.4f} ms  "
          f"min={min(wall_kern_ms):.4f}  max={max(wall_kern_ms):.4f}")
    total_mean_ms = (sum(h2d_ns) + sum(kern_ns) + sum(d2h_ns)) / len(h2d_ns) / 1e6
    kern_share = sum(kern_ns) / (sum(h2d_ns) + sum(kern_ns) + sum(d2h_ns)) * 100
    print(f"\n  Total device-side time per batch: {total_mean_ms:.4f} ms")
    print(f"  Kernel execution share of that total: {kern_share:.1f}%")


if __name__ == "__main__":
    main()
