"""
Cost of the pipeline's MPI collectives at their real sizes (report Table 4.1).

Times Barrier, bcast of the real file list, gather of one float32 per image and
gather of one float per rank. Prints the slowest rank's mean over the repeats.

Usage:
    mpirun -n 8 python experiments/measure_mpi_collectives.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset_loader import _discover_files  # noqa: E402

REPEATS = 20


def timed(comm, fn):
    samples = []
    for _ in range(REPEATS):
        comm.Barrier()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return max(comm.gather(sum(samples) / REPEATS, root=0) or [0.0])


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    files = _discover_files(str(ROOT / "data" / "EuroSAT")) if rank == 0 else []
    features = np.zeros(len(files) // size if rank == 0 else 54597 // size, dtype=np.float32)
    results = {
        "Barrier": timed(comm, comm.Barrier),
        "bcast file list": timed(comm, lambda: comm.bcast(files if rank == 0 else None, root=0)),
        "gather features": timed(comm, lambda: comm.gather(features, root=0)),
        "gather rank times": timed(comm, lambda: comm.gather(1.0, root=0)),
    }
    if rank == 0:
        print(f"P={size}, mean of {REPEATS}, slowest rank:")
        for name, t in results.items():
            print(f"  {name:18s} {t * 1e3:8.3f} ms")


if __name__ == "__main__":
    main()
