"""
Main program of the hybrid MPI + OpenCL + OpenMP image-processing pipeline.

Run with:  mpirun -n P python3 src/main_mpi.py

MPI starts P copies of this program (ranks, SPMD). Each rank loads the C++
library, processes its own block of the dataset, and rank 0 (the root)
gathers the features and reports the total time and throughput.
"""
import os
import sys
import ctypes                     # Python <-> C bridge to the C++ library
import time
from typing import Dict, Any

import numpy as np
from mpi4py import MPI            # MPI for Python

from pipeline.workflow_builder import build_pipeline_from_config, pipeline_warnings
from pipeline.scheduler import run_pipeline                        # sequential mode
from pipeline.batched_pipeline import run_batched_prefetch_pipeline  # 3-thread pipeline
from utils.timing import timed_block
from utils.config_loader import load_pipeline_config


# Compiled pipeline_kernels.cpp, relative to src/ (".so" becomes ".dylib" on macOS)
LIB_PATH = os.path.join(os.path.dirname(__file__), "../lib/libpipeline_kernels.so")


def load_library() -> ctypes.CDLL:
    """Load the compiled C++ library (OpenCL kernels + OpenMP loops).

    Each MPI rank is a separate process, so each rank loads its own copy
    and later creates its own OpenCL context and command queue.
    """
    ext = ".dylib" if sys.platform == "darwin" else ".so"   # macOS or Linux
    path = LIB_PATH.replace(".so", ext)

    if not os.path.exists(path):
        # Second location: a library built next to this file in src/
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "libpipeline_kernels" + ext))

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find shared library at {path}")

    lib = ctypes.CDLL(path)       # dlopen; C names are visible thanks to extern "C"
    return lib


def configure_c_api(lib: ctypes.CDLL) -> None:
    """Declare the C argument and return types (mirror of pipeline_kernels.h).

    Without argtypes, ctypes passes every argument as a C int, which would
    truncate 64-bit array pointers. Must be called before any lib.* call.
    """
    _float_p = ctypes.POINTER(ctypes.c_float)               # C: float*
    # (input*, batch_size, width, height, channels, output*)
    _5ints   = [_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, _float_p]

    # OpenCL set-up: int init_opencl(int mpi_rank)
    lib.init_opencl.argtypes = [ctypes.c_int]
    lib.init_opencl.restype  = ctypes.c_int

    # Stage 2: FFT magnitude (GPU)
    lib.process_images_fft.argtypes = _5ints
    lib.process_images_fft.restype  = ctypes.c_int

    # Stage 3: high-pass filter (GPU)
    lib.apply_frequency_filter_gpu.argtypes = _5ints
    lib.apply_frequency_filter_gpu.restype  = ctypes.c_int

    # Combined Stage 2+3 (backward compatibility only)
    lib.process_images_frequency_domain.argtypes = _5ints
    lib.process_images_frequency_domain.restype  = ctypes.c_int

    # Stage 4: feature extraction (CPU, OpenMP)
    lib.extract_features_cpu.argtypes = _5ints
    lib.extract_features_cpu.restype  = ctypes.c_int

    # GPU device queries
    lib.query_gpu_device_count.argtypes = []
    lib.query_gpu_device_count.restype  = ctypes.c_int
    lib.query_gpu_device_name.argtypes  = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.query_gpu_device_name.restype   = ctypes.c_int


def main() -> None:
    """Entry point run by every rank: set up, run the pipeline, gather results."""
    comm = MPI.COMM_WORLD         # all P processes
    rank = comm.Get_rank()        # this process: 0 .. P-1
    size = comm.Get_size()        # P

    if rank == 0:                 # print the banner once, not P times
        print("============================================================")
        print(" HPC Final Project: Hybrid MPI + OpenMP + OpenCL Pipeline")
        print("============================================================")
        print(f" Root rank: {rank} | MPI processes: {size}")
        print(" Parallelism: MPI (processes) + OpenMP (threads) + OpenCL (GPU)")
        print("============================================================")

    # 1. Load the C++ library and declare its C types
    try:
        lib = load_library()
    except Exception as e:
        print(f"Rank {rank}: Error loading library: {e}")
        return

    configure_c_api(lib)

    # 2. OpenCL set-up in C++; the rank selects the GPU (rank % number of GPUs)
    if lib.init_opencl(rank) != 0:
        print(f"Rank {rank}: Failed to init OpenCL. Check logs (CPU fallback may be used).")

    # All ranks wait here, so timing starts together and excludes GPU set-up
    comm.Barrier()

    # 3. Configurable workflow: every rank reads the same YAML and builds the
    #    same task graph; an invalid stage order stops every rank before any
    #    data is loaded.
    config = load_pipeline_config()
    try:
        graph = build_pipeline_from_config(comm, config)
    except ValueError as e:
        if rank == 0:
            print(f"[CONFIG ERROR] {e}", flush=True)
        sys.exit(1)
    if rank == 0:
        for w in pipeline_warnings(graph):              # e.g. unused stage output
            print(f"[WARN] {w}")

    # Shared state passed to every stage (stages add their results to it)
    pipeline_cfg = config.get("pipeline", {})
    context: Dict[str, Any] = {
        "comm": comm,                 # MPI communicator (used for the file broadcast)
        "lib": lib,                   # C++ library handle (used by the stages)
        "pipeline_config": pipeline_cfg,
    }

    # Execution mode: pipeline ON if task_parallel_prefetch is true in the
    # YAML (the default) or TASK_PARALLEL_PREFETCH=1 is set in the environment
    default_stages = ["io", "fft", "filter", "feature"]
    stages_norm = [
        str(s).strip().lower() for s in pipeline_cfg.get("stages", default_stages)
    ]
    use_prefetch = bool(pipeline_cfg.get("task_parallel_prefetch", False))
    env_pf = str(os.environ.get("TASK_PARALLEL_PREFETCH", "")).lower()
    if env_pf in ("1", "true", "yes"):
        use_prefetch = True

    # 4. Run the pipeline on this rank's block and time it (per rank).
    #    Pipeline ON : 3-thread pipeline, batches of io_batch_size (task parallelism).
    #    Pipeline OFF: sequential mode, each stage over the whole block
    #                  (used only for the comparison in report §5.6.4).
    #    The 3-thread pipeline is wired for io -> fft -> filter -> feature,
    #    so it requires that exact stage order.
    start_time = time.time()
    with timed_block("Full pipeline per-rank", comm=comm):   # rank 0 prints its time
        if use_prefetch and stages_norm == default_stages:
            batch_sz = int(pipeline_cfg.get("io_batch_size", 256))
            if rank == 0:
                print(
                    f"[CONFIG] task_parallel_prefetch=True, io_batch_size={batch_sz} "
                    "(3-thread queue pipeline: load || GPU FFT/filter || CPU features)"
                )
            run_batched_prefetch_pipeline(context, batch_size=batch_sz)   # pipeline ON
        else:
            if use_prefetch and stages_norm != default_stages and rank == 0:
                print(
                    "[WARN] task_parallel_prefetch ignored: requires stages "
                    "[io, fft, filter, feature] in that order."
                )
            run_pipeline(graph, context)                                  # pipeline OFF
    end_time = time.time()
    local_duration = end_time - start_time      # this rank's wall-clock time

    # 5. Gather on rank 0: one float32 feature per image (213 KB for 54,597
    #    images, instead of 2.68 GB of pixels) and one duration per rank.
    #    With the file-list broadcast in dataset_loader.py, this is the only
    #    data exchanged between ranks. Sending a NumPy array (not a list)
    #    keeps the pickled message at about 4 bytes per image.
    local_list = np.asarray(context.get("features", []), dtype=np.float32)

    all_lists = comm.gather(local_list, root=0)       # list of P arrays, in rank order
    all_durations = comm.gather(local_duration, root=0)   # list of P times (None on other ranks)

    if rank == 0:
        # Rank order + contiguous blocks -> features are in original file order
        all_features = np.concatenate(all_lists).tolist() if all_lists else []
        total_images = len(all_features)

        # T_P = slowest rank: the job finishes only when every rank has finished
        total_time = max(all_durations) if all_durations else 0.0
        throughput = (total_images / total_time) if total_time > 0 else 0.0

        print("Processing Complete.")
        print(f"Total Images Processed: {total_images}")
        if all_durations and len(all_durations) > 1:      # min vs max shows load imbalance
            print(
                f"Per-rank wall time [s]: min={min(all_durations):.4f}, "
                f"max={max(all_durations):.4f} (throughput uses max)"
            )
        print(f"Total Time: {total_time:.4f} seconds")
        if total_time > 0:
            print(f"Throughput: {throughput:.2f} img/sec")
        print(f"Feature Vector Sample (First 10): {all_features[:10]}")

        # 6. Save results for the experiments and the report
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
        os.makedirs(results_dir, exist_ok=True)

        # All features (compared against NumPy by verify_fft_against_numpy.py)
        features_path = os.path.join(results_dir, "features.npy")
        np.save(features_path, np.array(all_features, dtype=np.float32))

        # Human-readable run summary
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
