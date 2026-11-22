"""
Task parallelism inside each MPI rank: a 3-thread pipeline over batches.

Thread 1 loads batch k+1, Thread 2 runs the GPU stages on batch k, and
Thread 3 extracts features from batch k-1, so loading, GPU work and CPU work
overlap in time. The threads are linked by two bounded FIFO queues.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
from mpi4py import MPI

from dataset_loader import get_partitioned_paths, load_eurosat_dataset, load_images_from_paths
from stages.fft_stage import fft_frequency_stage          # Stage 2 -> C++ / OpenCL
from stages.filter_stage import filter_stage               # Stage 3 -> C++ / OpenCL
from stages.feature_stage import feature_extraction_stage  # Stage 4 -> C++ / OpenMP
from utils.dataset_utils import resolve_dataset


def _weak_scaling_max_images(pipeline_cfg: Dict[str, Any]) -> Optional[int]:
    """Images per rank for weak scaling (None = use the full dataset).

    Same priority as io_stage.py (environment first, then YAML), so both
    execution modes use the same setting.
    """
    raw = (
        os.environ.get("HPC_MAX_IMAGES_PER_RANK")
        or os.environ.get("WEAK_SCALING_IMAGES_PER_RANK")
        or pipeline_cfg.get("weak_scaling_images_per_rank")
    )
    if raw is None or raw == "":
        return None
    return int(raw)


def run_batched_prefetch_pipeline(context: Dict[str, Any], batch_size: int) -> None:
    """
    Run this rank's images through the 3-thread pipeline (report Section 3.3).

    Thread 1: Stage 1, load and preprocess the next batch (CPU)
    Thread 2: Stages 2 and 3, FFT and high-pass filter (GPU, OpenCL)
    Thread 3: Stage 4, feature extraction of the previous batch (CPU, OpenMP)
    Result: context["features"], one float32 per image, in file order.
    """
    import queue
    import threading

    comm: MPI.Comm = context["comm"]
    rank = comm.Get_rank()
    size = comm.Get_size()
    pipeline_cfg = context.get("pipeline_config") or {}

    dataset_root, (width, height) = resolve_dataset(comm)   # dataset folder, 64x64
    max_images = _weak_scaling_max_images(pipeline_cfg)

    context["width"] = width
    context["height"] = height
    context["channels"] = 3                                  # RGB

    # MPI data parallelism: rank 0 lists the files and broadcasts them; each
    # rank keeps only its own contiguous block of paths (dataset_loader.py)
    paths: List[str] = get_partitioned_paths(
        dataset_root, rank, size, comm=comm, max_images=max_images
    )

    # No files for this rank: run the stages once, sequentially, instead of
    # starting three threads for no work (same behaviour as io_stage.py)
    if not paths:
        context["input_data"] = load_eurosat_dataset(
            dataset_root,
            rank,
            size,
            comm=comm,
            target_size=(width, height),
            max_images=max_images,
        )
        fft_frequency_stage(context)
        filter_stage(context)
        feature_extraction_stage(context)
        return

    # Agglomeration (report §4.3): split the paths into batches of batch_size
    # (256). Larger batches amortise the fixed cost of each GPU launch;
    # smaller ones use less memory and fill the pipeline sooner.
    bs = max(1, int(batch_size))
    chunks = [paths[i : i + bs] for i in range(0, len(paths), bs)]

    def load_chunk(chunk: List[str]) -> np.ndarray:
        """Stage 1: read and resize one batch -> float32 (N, 64, 64, 3)."""
        if not chunk:
            return np.zeros((0, height, width, 3), dtype=np.float32)
        return load_images_from_paths(chunk, (width, height), max_pad_to=None)

    # End-of-stream marker. A unique object cannot be confused with real data.
    SENTINEL = object()

    # Two bounded FIFO queues link the three threads. maxsize=3 gives
    # backpressure: put() blocks when a queue is full, so a fast loader cannot
    # get ahead of a slow consumer and fill RAM with the whole dataset.
    q1 = queue.Queue(maxsize=3) # I/O -> Compute
    q2 = queue.Queue(maxsize=3) # Compute -> Feature

    all_features = []             # filled by Thread 3 only, in batch order

    # Thread 1 (I/O, producer): loads batches one after another; while it
    # reads batch k+1 from disk, Threads 2 and 3 work on earlier batches.
    def io_worker():
        for chunk in chunks:
            if not chunk:
                continue
            input_data = load_chunk(chunk)        # Stage 1 (disk read + decode)
            # Own shallow copy of the context per batch, so no two threads
            # ever write to the same dictionary
            ctx = context.copy()
            ctx["input_data"] = input_data
            q1.put(ctx)                           # blocks while q1 is full
        q1.put(SENTINEL)                          # no more batches

    # Thread 2 (GPU, consumer of q1 / producer of q2): FFT + filter on batch k
    def compute_worker():
        while True:
            ctx = q1.get()                        # blocks until a batch is ready
            if ctx is SENTINEL:
                q2.put(SENTINEL)                  # pass shutdown on to Thread 3
                break

            fft_frequency_stage(ctx)              # Stage 2: GPU FFT (ctypes, GIL released)
            filter_stage(ctx)                     # Stage 3: GPU filter

            q2.put(ctx)                           # hand batch k to Thread 3
            q1.task_done()                        # queue bookkeeping (no join() on queues)

    # Thread 3 (CPU, consumer of q2): features of batch k-1 while Thread 2
    # already runs the GPU stages on batch k
    def feature_worker():
        while True:
            ctx = q2.get()
            if ctx is SENTINEL:
                break

            feature_extraction_stage(ctx)         # Stage 4: OpenMP mean per image
            all_features.append(ctx["features"])  # FIFO queues keep batch order

            q2.task_done()

    t1 = threading.Thread(target=io_worker, name="Pipeline_IO_Thread")
    t2 = threading.Thread(target=compute_worker, name="Pipeline_GPU_Thread")
    t3 = threading.Thread(target=feature_worker, name="Pipeline_CPU_Feature_Thread")

    # Start all three threads before waiting on any of them; this is what
    # lets them run at the same time. The GIL is released during file I/O,
    # image decoding and every ctypes call into the C++ library, so the
    # threads make real concurrent progress, not just take turns.
    t1.start()
    t2.start()
    t3.start()

    # Wait for all three to finish (Thread 3 ends after the sentinel)
    t1.join()
    t2.join()
    t3.join()

    # One array of features for this rank, later gathered by main_mpi.py
    if all_features:
        context["features"] = np.concatenate(all_features)
    else:
        context["features"] = np.array([], dtype=np.float32)
