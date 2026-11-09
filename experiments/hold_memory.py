"""
Occupies RAM for the duration of an experiment batch (report Section 5.2).

Reported timings depend on whether the OS file cache holds the 3.05 GB
dataset: with free RAM it does (P=1 about 6.4 s, almost no disk traffic); with
RAM in use by other applications it does not, and every run reads the images
from disk (P=1 about 16 s, about 3 GB read). Holding 7 GB of the testbed's
16 GB keeps the machine in the second state, the one seen in normal use, so
that results do not depend on what else happens to be open. 7 GB was chosen
as the smallest amount tested (7-10 GB) at which each run's disk traffic
equals the dataset size while swap does not grow: 8 GB and above push other
programs into swap, which slows the runs. The data is random because macOS
would compress uniform pages instead of giving up the cached files.

Do NOT run run_pipeline_onoff_comparison.py under this holder: sequential mode
keeps a rank's whole subset and its copies in memory, and the combination
exhausted RAM and swap on the 16 GB testbed (kernel panic: watchdog timeout,
low swap space). That experiment runs without the holder.

Idle held pages can be swapped out, which returns the images to the file cache
(report Section 5.2). --active reads one byte per page every 2 s, so they stay in
RAM; 9 GB --active gave a stable disk state on a freshly restarted testbed.

Usage (from project root):
    python experiments/hold_memory.py &      # holds until killed
    python experiments/hold_memory.py 9 --active &
    python experiments/run_scaling_experiments.py 5
    kill %1
"""
import sys
import time

import numpy as np

HOLD_GB = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "--active" else 7.0

if __name__ == "__main__":
    block = np.random.default_rng(0).bytes(64_000_000)
    held = [bytes(block) + bytes([i % 256]) for i in range(int(HOLD_GB * 1e9 / len(block)))]
    print(f"Holding {HOLD_GB:g} GB of RAM until killed.", flush=True)
    while True:
        if "--active" in sys.argv:
            for block_i in held:
                sum(block_i[::16384])
            time.sleep(2)
        else:
            time.sleep(60)
