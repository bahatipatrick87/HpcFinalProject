import ctypes
from typing import Dict, Any

import numpy as np


def feature_extraction_stage(context: Dict[str, Any]) -> None:
    """
    Feature extraction stage (Stage 4): uses the C++ OpenMP kernel to reduce
    each filtered image down to a single scalar (mean intensity).

    This is a deliberate dimensionality collapse: gathering per-pixel data
    from every rank back to rank 0 would mean transferring gigabytes over
    MPI, so each rank reduces its own batch to one float per image first —
    the MPI_Gather at the end of main_mpi.py only ever moves 4 bytes/image.
    """
    lib = context["lib"]
    processed_data: np.ndarray = context["processed_data"]
    width: int = context["width"]
    height: int = context["height"]
    channels: int = context["channels"]

    batch_size = processed_data.shape[0]

    features = np.zeros(batch_size, dtype=np.float32)

    processed_ptr = processed_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    features_ptr = features.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    err = lib.extract_features_cpu(
        processed_ptr,
        batch_size,
        width,
        height,
        channels,
        features_ptr,
    )
    if err != 0:
        rank = context["comm"].Get_rank()
        print(f"Rank {rank}: Error in CPU feature extraction (err={err})")

    context["features"] = features

