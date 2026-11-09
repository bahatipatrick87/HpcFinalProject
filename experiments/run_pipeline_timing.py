"""
Pipeline Stage Overlap Timing Experiment
=========================================
Measures the wall-clock timeline of each stage in both modes:

  Mode A: Sequential (default off) — Load all, then GPU all, then CPU all.
  Mode B: Pipeline (3-thread queue) — Load || GPU || CPU on successive batches.

Produces results/pipeline_timing.csv with per-batch stage timestamps and
prints a Gantt-style summary to stdout suitable for inclusion in the report.

Usage:
    cd /Users/bahatipatrick/HpcFinalProject
    mpirun -n 1 python3 experiments/run_pipeline_timing.py
or:
    python3 experiments/run_pipeline_timing.py   (single-process)
"""
import os
import sys
import ctypes
import queue
import threading
import time
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from dataset_loader import get_partitioned_paths, load_images_from_paths
from utils.dataset_utils import resolve_dataset


# ------------------------------------------------------------------ helpers --

def _load_lib():
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    lib = ctypes.CDLL(str(ROOT / "lib" / f"libpipeline_kernels{ext}"))

    lib.init_opencl.argtypes  = [ctypes.c_int]
    lib.init_opencl.restype   = ctypes.c_int

    lib.process_images_frequency_domain.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.process_images_frequency_domain.restype = ctypes.c_int

    lib.extract_features_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.extract_features_cpu.restype = ctypes.c_int

    lib.query_gpu_device_count.argtypes = []
    lib.query_gpu_device_count.restype  = ctypes.c_int
    return lib


def _gpu_stage(lib, images, width, height, channels):
    bs = images.shape[0]
    inp = np.ascontiguousarray(images, dtype=np.float32)
    out = np.empty_like(inp)
    ip  = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    op  = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.process_images_frequency_domain(ip, bs, width, height, channels, op)
    return out


def _cpu_stage(lib, processed, width, height, channels):
    bs   = processed.shape[0]
    feat = np.empty(bs, dtype=np.float32)
    pp   = np.ascontiguousarray(processed, dtype=np.float32)
    fp   = pp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rp   = feat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.extract_features_cpu(fp, bs, width, height, channels, rp)
    return feat


# ------------------------------------------------------------------ modes --

def run_sequential(lib, chunks, width, height, channels):
    """Mode A: sequential — complete load, then GPU, then CPU per batch."""
    records = []
    T0 = time.perf_counter()
    for bi, chunk in enumerate(chunks):
        t_load_s = time.perf_counter() - T0
        imgs     = load_images_from_paths(chunk, (width, height))
        t_load_e = time.perf_counter() - T0

        t_gpu_s  = time.perf_counter() - T0
        proc     = _gpu_stage(lib, imgs, width, height, channels)
        t_gpu_e  = time.perf_counter() - T0

        t_cpu_s  = time.perf_counter() - T0
        _cpu_stage(lib, proc, width, height, channels)
        t_cpu_e  = time.perf_counter() - T0

        records.append({
            "mode": "sequential", "batch": bi,
            "load_start": t_load_s, "load_end": t_load_e,
            "gpu_start":  t_gpu_s,  "gpu_end":  t_gpu_e,
            "cpu_start":  t_cpu_s,  "cpu_end":  t_cpu_e,
        })
    return records


def run_pipeline(lib, chunks, width, height, channels):
    """Mode B: 3-thread queue pipeline — Load || GPU || CPU on successive batches."""
    SENTINEL = object()
    q1 = queue.Queue(maxsize=3)
    q2 = queue.Queue(maxsize=3)
    records_lock = threading.Lock()
    records = []
    T0 = time.perf_counter()

    def io_worker():
        for bi, chunk in enumerate(chunks):
            t_s = time.perf_counter() - T0
            imgs = load_images_from_paths(chunk, (width, height))
            t_e = time.perf_counter() - T0
            with records_lock:
                records.append({"mode": "pipeline", "batch": bi,
                                 "stage": "load", "start": t_s, "end": t_e})
            q1.put((bi, imgs))
        q1.put(SENTINEL)

    def gpu_worker():
        while True:
            item = q1.get()
            if item is SENTINEL:
                q2.put(SENTINEL)
                break
            bi, imgs = item
            t_s  = time.perf_counter() - T0
            proc = _gpu_stage(lib, imgs, width, height, channels)
            t_e  = time.perf_counter() - T0
            with records_lock:
                records.append({"mode": "pipeline", "batch": bi,
                                 "stage": "gpu", "start": t_s, "end": t_e})
            q2.put((bi, proc))
            q1.task_done()

    def cpu_worker():
        while True:
            item = q2.get()
            if item is SENTINEL:
                break
            bi, proc = item
            t_s = time.perf_counter() - T0
            _cpu_stage(lib, proc, width, height, channels)
            t_e = time.perf_counter() - T0
            with records_lock:
                records.append({"mode": "pipeline", "batch": bi,
                                 "stage": "cpu", "start": t_s, "end": t_e})
            q2.task_done()

    threads = [
        threading.Thread(target=io_worker,  name="IO"),
        threading.Thread(target=gpu_worker, name="GPU"),
        threading.Thread(target=cpu_worker, name="CPU"),
    ]
    for t in threads: t.start()
    for t in threads: t.join()
    return records


# ------------------------------------------------------------------ main --

def main():
    try:
        from mpi4py import MPI
        comm  = MPI.COMM_WORLD
        rank  = comm.Get_rank()
        nrank = comm.Get_size()
    except ImportError:
        class _FakeComm:
            def Get_rank(self): return 0
            def Get_size(self): return 1
            def bcast(self, v, root=0): return v
        comm  = _FakeComm()
        rank  = 0
        nrank = 1

    lib = _load_lib()
    lib.init_opencl(rank)

    dataset_root, (width, height) = resolve_dataset(comm)
    channels = 3
    BATCH = 64   # small batches so overlap is visible
    MAX   = 512  # limit images for the timing experiment

    paths = get_partitioned_paths(dataset_root, rank, nrank, comm=comm,
                                  max_images=MAX)
    chunks = [paths[i:i+BATCH] for i in range(0, len(paths), BATCH)]
    if rank == 0:
        print(f"Pipeline timing experiment: {len(paths)} images, "
              f"{len(chunks)} batches of {BATCH}")
        print(f"GPU devices: {lib.query_gpu_device_count()}")

    # --- warm up (discard timings) ---
    if chunks:
        _gpu_stage(lib, load_images_from_paths(chunks[0][:4], (width, height)),
                   width, height, channels)

    # --- Sequential ---
    t0 = time.perf_counter()
    seq_records = run_sequential(lib, chunks, width, height, channels)
    t_seq = time.perf_counter() - t0

    # --- Pipeline ---
    t0 = time.perf_counter()
    pip_records = run_pipeline(lib, chunks, width, height, channels)
    t_pip = time.perf_counter() - t0

    if rank == 0:
        out_dir = ROOT / "results"
        out_dir.mkdir(exist_ok=True)

        # Write sequential records
        seq_csv = out_dir / "pipeline_timing_sequential.csv"
        with seq_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode","batch","load_start","load_end",
                                               "gpu_start","gpu_end","cpu_start","cpu_end"])
            w.writeheader(); w.writerows(seq_records)

        # Write pipeline records
        pip_csv = out_dir / "pipeline_timing_pipeline.csv"
        pip_flat = []
        for r in pip_records:
            pip_flat.append({"mode": r["mode"], "batch": r["batch"],
                              "stage": r["stage"], "start": r["start"], "end": r["end"]})
        with pip_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mode","batch","stage","start","end"])
            w.writeheader(); w.writerows(pip_flat)

        speedup = t_seq / t_pip if t_pip > 0 else float("inf")
        print(f"\n{'='*60}")
        print(f"  Sequential total  : {t_seq:.3f} s")
        print(f"  Pipeline total    : {t_pip:.3f} s")
        print(f"  Pipeline speedup  : {speedup:.2f}x")
        print(f"  CSV written to    : {out_dir}")
        print(f"{'='*60}")

        # ASCII Gantt summary (first 3 batches)
        print("\nPipeline Gantt (first 3 batches, relative seconds):")
        print(f"{'Batch':>5}  {'Stage':>8}  {'Start':>8}  {'End':>8}  {'Dur':>7}")
        shown = sorted(pip_flat, key=lambda x: (x["batch"], x["stage"]))
        for r in shown:
            if r["batch"] >= 3: break
            dur = r["end"] - r["start"]
            print(f"{r['batch']:>5}  {r['stage']:>8}  {r['start']:>8.3f}  "
                  f"{r['end']:>8.3f}  {dur:>7.3f}s")

        # Check overlap: does Load(batch k+1) run concurrently with GPU(batch k)?
        # (the adjacent pair the 3-thread pipeline is actually designed to overlap;
        # comparing non-adjacent stages, e.g. Load(0) vs GPU(1), is the wrong pair
        # and can falsely report "no overlap" even when the pipeline is working.)
        def _interval(stage, batch):
            r = next((r for r in pip_flat
                      if r["batch"] == batch and r["stage"] == stage), None)
            return (r["start"], r["end"]) if r else None

        n_batches = max(r["batch"] for r in pip_flat) + 1
        found_overlap = False
        for k in range(n_batches - 1):
            gpu_k = _interval("gpu", k)
            load_k1 = _interval("load", k + 1)
            if not gpu_k or not load_k1:
                continue
            overlap = min(gpu_k[1], load_k1[1]) - max(gpu_k[0], load_k1[0])
            if overlap > 0:
                print(f"\n[VERIFIED] Pipeline overlap: Load batch-{k+1} "
                      f"({load_k1[0]:.3f}-{load_k1[1]:.3f}s) overlaps GPU batch-{k} "
                      f"({gpu_k[0]:.3f}-{gpu_k[1]:.3f}s) for {overlap:.3f}s "
                      f"=> stages overlap in wall-clock time.")
                found_overlap = True
                break
        if not found_overlap:
            print(f"\n[INFO] No overlapping Load/GPU interval found across "
                  f"{n_batches} batches. Batches may be too small relative to "
                  f"GPU dispatch overhead at this scale.")


if __name__ == "__main__":
    main()
