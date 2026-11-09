import os
from typing import Dict, Any

from mpi4py import MPI

from dataset_loader import load_eurosat_dataset
from utils.dataset_utils import resolve_dataset


def io_load_and_partition(context: Dict[str, Any]) -> None:
    """
    IO stage (Stage 1): load and partition the EuroSAT dataset for this MPI rank.
    Data parallelism: each rank loads only its disjoint subset of images
    (partitioning logic lives in dataset_loader.get_partitioned_paths).
    Stores the loaded batch in context['input_data'] as a NumPy array for the
    downstream FFT/filter/feature stages to consume.

    This is the "sequential mode" entry point — it loads this rank's entire
    share of the dataset in one call. The task-parallel alternative
    (batched_pipeline.run_batched_prefetch_pipeline) instead loads one batch
    at a time on a background thread so loading overlaps with GPU compute.
    """
    comm: MPI.Comm = context["comm"]
    rank = comm.Get_rank()
    size = comm.Get_size()

    dataset_root, (width, height) = resolve_dataset(comm)

    pipeline_cfg = context.get("pipeline_config") or {}
    # Three ways to request a fixed per-rank image count, checked in priority
    # order: an explicit env var for ad-hoc runs, the experiment-specific env
    # var set by the weak-scaling driver script, or the config file default.
    # When unset, every rank instead gets its natural share of the full
    # dataset (strong-scaling behaviour).
    max_images = (
        os.environ.get("HPC_MAX_IMAGES_PER_RANK")
        or os.environ.get("WEAK_SCALING_IMAGES_PER_RANK")
        or pipeline_cfg.get("weak_scaling_images_per_rank")
    )
    if max_images is not None:
        max_images = int(max_images)

    input_data = load_eurosat_dataset(
        dataset_root,
        rank,
        size,
        comm=comm,
        target_size=(width, height),
        max_images=max_images,
    )

    context["input_data"] = input_data
    context["width"] = width
    context["height"] = height
    context["channels"] = 3  # EuroSAT RGB

