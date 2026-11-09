import os
import sys
import ctypes
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI

from pipeline.workflow_builder import build_pipeline_from_config, pipeline_warnings
from pipeline.scheduler import run_pipeline
from pipeline.batched_pipeline import run_batched_prefetch_pipeline
from utils.timing import timed_block
from utils.config_loader import load_pipeline_config


LIB_PATH = os.path.join(os.path.dirname(__file__), "../lib/libpipeline_kernels.so")


def load_library() -> ctypes.CDLL:
    """Locate and dlopen the compiled C++ kernel library. macOS uses .dylib,
    Linux uses .so — every MPI rank loads its own independent copy of the
    same compiled code (shared libraries, not shared library *instances*)."""
    # Detect OS to choose extension
    ext = ".dylib" if sys.platform == "darwin" else ".so"
    path = LIB_PATH.replace(".so", ext)

    if not os.path.exists(path):
        # Fallback to current directory or different build location checks
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "libpipeline_kernels" + ext))

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find shared library at {path}")

    lib = ctypes.CDLL(path)
    return lib


def configure_c_api(lib: ctypes.CDLL) -> None:
    """Declare each C function's argument and return types to ctypes.

    Without this, ctypes assumes every argument is a plain int and every
    return value is also an int — which happens to work by accident for some
    calls but silently corrupts float-pointer arguments for others. This
    must be called once per rank, right after load_library(), before any
    other lib.* call.
    """
    _float_p = ctypes.POINTER(ctypes.c_float)
    _5ints   = [_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _float_p]

    lib.init_opencl.argtypes = [ctypes.c_int]
    lib.init_opencl.restype  = ctypes.c_int

    # Stage 2: FFT magnitude (GPU)
    lib.process_images_fft.argtypes = _5ints
    lib.process_images_fft.restype  = ctypes.c_int

    # Stage 3: high-pass filter (GPU)
    lib.apply_frequency_filter_gpu.argtypes = _5ints
    lib.apply_frequency_filter_gpu.restype  = ctypes.c_int

    # Combined Stage 2+3 (backward compat)
    lib.process_images_frequency_domain.argtypes = _5ints
    lib.process_images_frequency_domain.restype  = ctypes.c_int

    # Stage 4: feature extraction (CPU / OpenMP)
    lib.extract_features_cpu.argtypes = _5ints
    lib.extract_features_cpu.restype  = ctypes.c_int

    # GPU device queries
    lib.query_gpu_device_count.argtypes = []
    lib.query_gpu_device_count.restype  = ctypes.c_int
    lib.query_gpu_device_name.argtypes  = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.query_gpu_device_name.restype   = ctypes.c_int


def main() -> None:
    """
    Entry point launched by `mpirun -n P python3 src/main_mpi.py`.

    MPI starts P independent copies of this exact script as separate OS
    processes (data parallelism); every rank runs the same steps below on
    its own disjoint slice of the dataset, with no code branching needed for
    "which rank am I" beyond reading comm.Get_rank().
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print("============================================================")
        print(" HPC Final Project: Hybrid MPI + OpenMP + OpenCL Pipeline")
        print("============================================================")
        print(f" Master Rank: {rank} | Total Processes: {size}")
        print(" Optimization: MPI (Nodes) + OpenMP (Threads) + OpenCL (GPU)")
        print("============================================================")

    # 1. Load and configure C++ kernel library
    try:
        lib = load_library()
    except Exception as e:
        print(f"Rank {rank}: Error loading library: {e}")
        return

    configure_c_api(lib)

    # 2. Initialize OpenCL (one per rank), passing the rank for multi-GPU selection
    if lib.init_opencl(rank) != 0:
        print(f"Rank {rank}: Failed to init OpenCL. Check logs (CPU fallback may be used).")

    # Wait for every rank to finish GPU/context setup before any rank starts
    # timing its pipeline run — otherwise a rank with a slow init could make
    # its own measured duration (and the max() used for "total time" below)
    # include setup cost that other ranks don't pay.
    comm.Barrier()

    # 3. Load config and build task graph dynamically (user-customizable workflow, proposal 0.5.5).
    # Every rank reads the same YAML, so an invalid stage list fails on every
    # rank alike, before any data is loaded.
    config = load_pipeline_config()
    try:
        graph = build_pipeline_from_config(comm, config)
    except ValueError as e:
        if rank == 0:
            print(f"[CONFIG ERROR] {e}", flush=True)
        sys.exit(1)
    if rank == 0:
        for w in pipeline_warnings(graph):
            print(f"[WARN] {w}")

    pipeline_cfg = config.get("pipeline", {})
    context: Dict[str, Any] = {
        "comm": comm,
        "lib": lib,
        "pipeline_config": pipeline_cfg,
    }

    default_stages = ["io", "fft", "filter", "feature"]
    stages_norm = [
        str(s).strip().lower() for s in pipeline_cfg.get("stages", default_stages)
    ]
    use_prefetch = bool(pipeline_cfg.get("task_parallel_prefetch", False))
    env_pf = str(os.environ.get("TASK_PARALLEL_PREFETCH", "")).lower()
    if env_pf in ("1", "true", "yes"):
        use_prefetch = True

    # 4. Run pipeline and measure wall-clock time explicitly (per rank).
    # This is the single switch between the two execution modes compared
    # throughout the report: the task-parallel 3-thread pipeline (used for
    # every "default" result) versus the plain sequential scheduler (used
    # only for the §5.6.4 pipeline-on/off comparison). The prefetch path
    # additionally requires the default stage order, since the 3-thread
    # pipeline's queue wiring assumes io->fft->filter->feature specifically.
    start_time = time.time()
    with timed_block("Full pipeline per-rank", comm=comm):
        if use_prefetch and stages_norm == default_stages:
            batch_sz = int(pipeline_cfg.get("io_batch_size", 256))
            if rank == 0:
                print(
                    f"[CONFIG] task_parallel_prefetch=True, io_batch_size={batch_sz} "
                    "(3-thread queue pipeline: load || GPU FFT/filter || CPU features)"
                )
            run_batched_prefetch_pipeline(context, batch_size=batch_sz)
        else:
            if use_prefetch and stages_norm != default_stages and rank == 0:
                print(
                    "[WARN] task_parallel_prefetch ignored: requires stages "
                    "[io, fft, filter, feature] in that order."
                )
            run_pipeline(graph, context)
    end_time = time.time()
    local_duration = end_time - start_time

    # 5. Gather features (if feature stage was in the pipeline).
    # MPI_Gather: the only other inter-rank communication besides the
    # MPI_Bcast in dataset_loader.py. Each rank sends its features as one
    # float32 NumPy array (one value per image; 213 KB in total for the
    # 54,597-image dataset) rather than raw pixel data (2.68 GB) — the
    # >12,000x reduction the report's communication-cost analysis is built
    # on. Sending the array, not a Python list, keeps the pickled message
    # at 4 bytes per image.
    local_list = np.asarray(context.get("features", []), dtype=np.float32)

    all_lists = comm.gather(local_list, root=0)
    all_durations = comm.gather(local_duration, root=0)

    if rank == 0:
        all_features = np.concatenate(all_lists).tolist() if all_lists else []
        total_images = len(all_features)

        # For global time, take the max of per-rank wall-clock (strong-scaling
        # style): the slowest rank determines when the whole job is actually
        # done, which is what a real deployment would wait for.
        total_time = max(all_durations) if all_durations else 0.0
        throughput = (total_images / total_time) if total_time > 0 else 0.0

        print("Processing Complete.")
        print(f"Total Images Processed: {total_images}")
        if all_durations and len(all_durations) > 1:
            print(
                f"Per-rank wall time [s]: min={min(all_durations):.4f}, "
                f"max={max(all_durations):.4f} (throughput uses max)"
            )
        print(f"Total Time: {total_time:.4f} seconds")
        if total_time > 0:
            print(f"Throughput: {throughput:.2f} img/sec")
        print(f"Feature Vector Sample (First 10): {all_features[:10]}")

        # Persist results for experiments and the report
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
        os.makedirs(results_dir, exist_ok=True)

        # Save features as NumPy binary for later analysis
        features_path = os.path.join(results_dir, "features.npy")
        np.save(features_path, np.array(all_features, dtype=np.float32))

        # Save a human-readable summary
        summary_path = os.path.join(results_dir, "run_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("HPC Final Project Run Summary\n")
            f.write("=====================================\n")
            f.write(f"Total MPI processes: {size}\n")
            f.write(f"Total images processed: {total_images}\n")
            f.write(f"Total time [s]: {total_time:.6f}\n")
            f.write(f"Throughput [img/s]: {throughput:.2f}\n")
            f.write(f"Features file: {features_path}\n")



if __name__ == "__main__":
    main()

