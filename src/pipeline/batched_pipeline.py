"""
Batched execution with optional pipeline (task) parallelism.

Three worker threads per MPI rank (queue-based): I/O loads successive batches,
one thread runs FFT+filter (GPU-bound), one runs feature extraction (CPU-bound),
so load, GPU stages, and features can overlap on different batches.
"""
import os
from typing import Any, Dict, List, Optional

import numpy as np
from mpi4py import MPI

from dataset_loader import get_partitioned_paths, load_eurosat_dataset, load_images_from_paths
from stages.fft_stage import fft_frequency_stage
from stages.filter_stage import filter_stage
from stages.feature_stage import feature_extraction_stage
from utils.dataset_utils import resolve_dataset


def _weak_scaling_max_images(pipeline_cfg: Dict[str, Any]) -> Optional[int]:
    """Same env-var/config priority order as io_stage.py, kept in sync so
    both execution modes honour the same weak-scaling override."""
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
    Run io (partitioned paths + per-batch load), fft, filter, feature using
    a full 3-stage pipeline (Task Parallelism).
    
    Stage 1: Load & Preprocess (CPU thread)
    Stage 2: FFT & Filter (GPU thread)
    Stage 3: Feature Extraction (CPU thread)
    
    This fully implements pipeline/task parallelism over the distinct stages.
    """
    import queue
    import threading

    comm: MPI.Comm = context["comm"]
    rank = comm.Get_rank()
    size = comm.Get_size()
    pipeline_cfg = context.get("pipeline_config") or {}

    dataset_root, (width, height) = resolve_dataset(comm)
    max_images = _weak_scaling_max_images(pipeline_cfg)

    context["width"] = width
    context["height"] = height
    context["channels"] = 3

    paths: List[str] = get_partitioned_paths(
        dataset_root, rank, size, comm=comm, max_images=max_images
    )

    # No on-disk images for this rank (or empty dataset): fall back to the
    # plain sequential path so behaviour matches io_stage.py exactly instead
    # of standing up three threads for zero work.
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

    # Task agglomeration (report §4.3): group this rank's images into fixed
    # batches rather than pipelining one image at a time. Larger batches
    # amortise the fixed per-dispatch GPU overhead; smaller batches keep the
    # in-flight memory footprint and pipeline latency low. io_batch_size=256
    # is the chosen middle ground (see the report for the full trade-off).
    bs = max(1, int(batch_size))
    chunks = [paths[i : i + bs] for i in range(0, len(paths), bs)]

    def load_chunk(chunk: List[str]) -> np.ndarray:
        if not chunk:
            return np.zeros((0, height, width, 3), dtype=np.float32)
        return load_images_from_paths(chunk, (width, height), max_pad_to=None)

    # Sentinel marks "no more batches" on a queue; using a unique object
    # (rather than e.g. None) avoids any ambiguity with a legitimate value.
    SENTINEL = object()

    # Two bounded queues chain the three stages: I/O -> GPU compute -> CPU
    # feature extraction. maxsize=3 caps how many loaded-but-unprocessed
    # batches can pile up in memory if a downstream stage falls behind —
    # without a bound, a slow GPU stage would let the I/O thread race ahead
    # and buffer the entire dataset in RAM.
    q1 = queue.Queue(maxsize=3) # I/O -> Compute
    q2 = queue.Queue(maxsize=3) # Compute -> Feature

    all_features = []

    # Thread 1 (I/O): continuously loads the next batch from disk while the
    # other two threads are still working on earlier batches — this is what
    # makes disk I/O overlap with GPU/CPU compute in wall-clock time.
    def io_worker():
        for chunk in chunks:
            if not chunk:
                continue
            input_data = load_chunk(chunk)
            # Each batch gets its own shallow copy of the shared context so
            # the three threads never mutate the same dict concurrently.
            ctx = context.copy()
            ctx["input_data"] = input_data
            q1.put(ctx)
        q1.put(SENTINEL)

    # Thread 2 (GPU): runs FFT + filter for one batch while Thread 1 loads
    # the next batch and Thread 3 finishes features for the previous one.
    def compute_worker():
        while True:
            ctx = q1.get()
            if ctx is SENTINEL:
                q2.put(SENTINEL)  # propagate shutdown to the next stage
                break

            # FFT and Filter (GPU bound stages)
            fft_frequency_stage(ctx)
            filter_stage(ctx)

            q2.put(ctx)
            q1.task_done()

    # Thread 3 (CPU/OpenMP): extracts features for one batch while Thread 2
    # is already running the GPU stages on the next one.
    def feature_worker():
        while True:
            ctx = q2.get()
            if ctx is SENTINEL:
                break

            # Feature Extraction (CPU bound stage)
            feature_extraction_stage(ctx)
            all_features.append(ctx["features"])

            q2.task_done()

    t1 = threading.Thread(target=io_worker, name="Pipeline_IO_Thread")
    t2 = threading.Thread(target=compute_worker, name="Pipeline_GPU_Thread")
    t3 = threading.Thread(target=feature_worker, name="Pipeline_CPU_Feature_Thread")

    # Starting all three before joining any of them is what actually enables
    # the overlap: Python's GIL is released during I/O waits and during the
    # ctypes calls into the C++/OpenCL library, so these three threads do
    # genuinely make concurrent progress rather than only alternating.
    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    if all_features:
        context["features"] = np.concatenate(all_features)
    else:
        context["features"] = np.array([], dtype=np.float32)
