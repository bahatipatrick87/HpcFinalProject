import os
from typing import Tuple

from mpi4py import MPI

from .config_loader import load_pipeline_config, get_dataset_root


def resolve_dataset(comm: MPI.Comm) -> Tuple[str, Tuple[int, int]]:
    """
    Load config and resolve the dataset root directory and target size.
    Returns (dataset_root_abs_path, (width, height)).

    Shared by both the sequential scheduler (io_stage.py) and the task-
    parallel pipeline (batched_pipeline.py) so the two execution modes agree
    on exactly where the data lives and how it's resized, with no risk of
    drifting apart.
    """
    cfg = load_pipeline_config()
    dataset_root = get_dataset_root(cfg)

    dataset_cfg = cfg.get("dataset", {})
    width = int(dataset_cfg.get("target_width", 64))
    height = int(dataset_cfg.get("target_height", 64))

    # Logged once from rank 0 only, to avoid P duplicate lines when running
    # with many MPI ranks.
    if comm.Get_rank() == 0:
        print(f"[CONFIG] Dataset root: {dataset_root}")
        print(f"[CONFIG] Target size: {width}x{height}")

    return dataset_root, (width, height)

