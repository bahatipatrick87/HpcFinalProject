"""
Pipeline (task-parallel prefetch) ON vs OFF, end-to-end comparison.

Measures the real, end-to-end wall-clock payoff of the three-thread
intra-rank pipeline (Section 3.3 in the report) by running the full
production pipeline on the complete EuroSAT dataset with
task_parallel_prefetch toggled true/false, at P=1 and P=8, full dataset.

Do not run under experiments/hold_memory.py: sequential mode is memory-hungry
and the combination exhausted RAM and swap on the 16 GB testbed.

Sampling: 3 rounds in which ON and OFF alternate; every timed run is
immediately preceded by an untimed run of the same mode, so that it starts
from the same file-cache state. Reported as the mean.

Temporarily edits config/pipeline.yaml's task_parallel_prefetch value
for the OFF runs and always restores the original file afterward (even
on error), since this is the single shared config file every other
script and the live demo also read from.

Run from project root:
    python experiments/run_pipeline_onoff_comparison.py

Output: results/pipeline_onoff_comparison.csv
"""
import os
import re
import statistics as st
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"

N_TIMED_TRIALS = 3


def set_prefetch(enabled: bool, original_text: str) -> None:
    new_text = re.sub(
        r"task_parallel_prefetch:\s*(true|false)",
        f"task_parallel_prefetch: {'true' if enabled else 'false'}",
        original_text,
    )
    CONFIG_PATH.write_text(new_text)


def run_once(num_ranks: int) -> float:
    venv_python = PROJECT_ROOT / "venv" / "bin" / "python"
    python_exe = str(venv_python if venv_python.exists() else "python3")
    cmd = ["mpirun", "-n", str(num_ranks), python_exe, "src/main_mpi.py"]

    start = time.time()
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    elapsed = time.time() - start
    for line in proc.stdout.splitlines():
        if "Total Time:" in line:
            try:
                elapsed = float(line.split()[-2])
            except Exception:
                pass
            break
    return elapsed


def run_mode(num_ranks: int, enabled: bool, original_text: str) -> float:
    set_prefetch(enabled, original_text)
    return run_once(num_ranks)


def measure_interleaved(num_ranks: int, original_text: str):
    # Rounds alternate ON and OFF so both modes see the same machine state.
    # Each timed run is immediately preceded by an untimed run of the same
    # mode: an OFF run holds the rank's whole subset in memory (~4 GB at
    # P=1) and changes what the OS file cache holds for the run after it.
    on_trials, off_trials = [], []
    for i in range(N_TIMED_TRIALS):
        run_mode(num_ranks, True, original_text)
        on_trials.append(run_mode(num_ranks, True, original_text))
        run_mode(num_ranks, False, original_text)
        off_trials.append(run_mode(num_ranks, False, original_text))
        print(f"  [P={num_ranks}] trial {i + 1}/{N_TIMED_TRIALS}: "
              f"ON {on_trials[-1]:.3f}s  OFF {off_trials[-1]:.3f}s")
    sd = lambda xs: st.stdev(xs) if len(xs) > 1 else 0.0
    return (st.mean(on_trials), sd(on_trials), on_trials,
            st.mean(off_trials), sd(off_trials), off_trials)


def main():
    original_text = CONFIG_PATH.read_text()
    rows = []
    try:
        for p in (1, 8):
            print(f"=== P={p}, pipeline ON/OFF interleaved ===")
            (on_mean, on_std, on_trials,
             off_mean, off_std, off_trials) = measure_interleaved(p, original_text)

            speedup = off_mean / on_mean if on_mean > 0 else 0.0
            rows.append((p, on_mean, on_std, off_mean, off_std, speedup,
                         on_trials, off_trials))
            print(f"  P={p}: ON={on_mean:.3f}+-{on_std:.3f}s  "
                  f"OFF={off_mean:.3f}+-{off_std:.3f}s  speedup={speedup:.2f}x\n")
    finally:
        CONFIG_PATH.write_text(original_text)
        print("Restored original config/pipeline.yaml (task_parallel_prefetch: true).")

    csv_path = RESULTS_DIR / "pipeline_onoff_comparison.csv"
    with csv_path.open("w") as f:
        f.write("P,on_mean_s,on_std_s,off_mean_s,off_std_s,speedup\n")
        for p, on_mean, on_std, off_mean, off_std, speedup, _, _ in rows:
            f.write(f"{p},{on_mean:.4f},{on_std:.4f},{off_mean:.4f},{off_std:.4f},{speedup:.4f}\n")
    print(f"\nResults written to {csv_path}")

    raw_path = RESULTS_DIR / "pipeline_onoff_raw_trials.csv"
    with raw_path.open("w") as f:
        f.write("P,mode,trial,elapsed_seconds\n")
        for p, _, _, _, _, _, on_trials, off_trials in rows:
            for i, t in enumerate(on_trials):
                f.write(f"{p},on,{i + 1},{t:.4f}\n")
            for i, t in enumerate(off_trials):
                f.write(f"{p},off,{i + 1},{t:.4f}\n")
    print(f"Raw per-trial data written to {raw_path}")


if __name__ == "__main__":
    main()
