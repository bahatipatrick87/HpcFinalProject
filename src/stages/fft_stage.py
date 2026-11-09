"""
Stage 2: FFT Magnitude (GPU via OpenCL).

Calls process_images_fft() in the C++ library, which runs the
`fft_magnitude` OpenCL kernel on the GPU: a radix-2 2D FFT in two passes, one
work-item per row and then one per column, for every image and channel,
producing the same magnitude spectrum as numpy.fft.fft2.

Output stored in context['fft_data'] so the subsequent filter_stage can
read it without overwriting the original input.
"""
import ctypes
from typing import Dict, Any

import numpy as np


def fft_frequency_stage(context: Dict[str, Any]) -> None:
    lib        = context["lib"]
    input_data: np.ndarray = context["input_data"]
    width:   int = context["width"]
    height:  int = context["height"]
    channels:int = context["channels"]

    batch_size = input_data.shape[0]
    if not input_data.flags["C_CONTIGUOUS"]:
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)

    fft_out = np.empty_like(input_data, dtype=np.float32)

    ip = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    op = fft_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Use the dedicated FFT function (Stage 2 only — no filtering here), so
    # Stage 2 and Stage 3 map to separate, individually timeable C++ calls.
    fn = getattr(lib, "process_images_fft", None)
    if fn is None:
        # backward compat: older builds expose only the combined function
        fn = lib.process_images_frequency_domain

    err = fn(ip, batch_size, width, height, channels, op)
    if err != 0:
        rank = context["comm"].Get_rank()
        print(f"Rank {rank}: Error in FFT stage (err={err})")

    # Store FFT result for filter_stage; also keep as processed_data for
    # pipelines that skip the filter stage.
    context["fft_data"] = fft_out
    context["processed_data"] = fft_out
