"""
Stage 3: High-Pass Frequency Filter (GPU via OpenCL).

Calls apply_frequency_filter_gpu() in the C++ library, which runs the
`highpass_filter` OpenCL kernel on the GPU. Each work-item applies the
element-wise high-pass mask H(u,v) = 1 - exp(-d²/50) to suppress low-
frequency (DC) components, emphasising edges and texture.

Reads context['fft_data'] (output of Stage 2) and writes filtered result
to context['processed_data'] for the feature extraction stage.
"""
import ctypes
from typing import Dict, Any

import numpy as np


def filter_stage(context: Dict[str, Any]) -> None:
    lib = context.get("lib")
    if lib is None:
        return  # no C++ library loaded — skip silently

    # Prefer fft_data (clean FFT output); fall back to processed_data so this
    # stage also works if the DAG config skips Stage 2 and feeds it raw input.
    spectrum: np.ndarray = context.get("fft_data")
    if spectrum is None:
        spectrum = context.get("processed_data")
    if spectrum is None:
        return

    width:    int = context["width"]
    height:   int = context["height"]
    channels: int = context["channels"]
    batch_size = spectrum.shape[0]

    if not spectrum.flags["C_CONTIGUOUS"]:
        spectrum = np.ascontiguousarray(spectrum, dtype=np.float32)

    filtered = np.empty_like(spectrum, dtype=np.float32)
    sp = spectrum.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    fp = filtered.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Use the dedicated GPU filter function (Stage 3 only)
    fn = getattr(lib, "apply_frequency_filter_gpu", None)
    if fn is None:
        # backward compat: if the library is old, pass spectrum through unchanged
        context["processed_data"] = spectrum
        return

    err = fn(sp, batch_size, width, height, channels, fp)
    if err != 0:
        rank = context["comm"].Get_rank()
        print(f"Rank {rank}: Error in filter stage (err={err})")

    context["processed_data"] = filtered
