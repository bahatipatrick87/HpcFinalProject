import time
from contextlib import contextmanager
from typing import Optional

from mpi4py import MPI


@contextmanager
def timed_block(label: str, comm: Optional[MPI.Comm] = None):
    """
    Simple context manager to time a code block.
    Only rank 0 prints the timing when an MPI communicator is provided, since
    printing from every rank would interleave P copies of the same line.
    Note this measures only the calling rank's own wall-clock time; the
    cross-rank "total time" reported in main_mpi.py is a separate
    comm.gather()+max() over each rank's locally measured duration.
    """
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        rank = 0
        if comm is not None:
            rank = comm.Get_rank()
        if rank == 0:
            print(f"[TIMING] {label}: {end - start:.4f} s")

