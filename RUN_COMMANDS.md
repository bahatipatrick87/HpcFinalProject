# How to Build, Run, and Evaluate the Project

All commands assume the project root:

```bash
cd /Users/bahatipatrick/HpcFinalProject
```

## 1. One-time build and environment setup

```bash
chmod +x scripts/build.sh
./scripts/build.sh
```

This:
- Builds the C++ OpenCL/OpenMP shared library into `lib/`.
- Creates/updates the Python virtual environment `venv/`.
- Installs all Python dependencies from `requirements.txt`.

## 2. Activate the virtual environment

```bash
cd /Users/bahatipatrick/HpcFinalProject
source venv/bin/activate
```

You should see a `(venv)` prefix in your shell prompt.

## 3. Run the hybrid MPI pipeline (EuroSAT)

From the project root with the venv active:

```bash
mpirun -n 4 ./venv/bin/python src/main_mpi.py
```

This:
- Starts 4 MPI processes.
- Loads the EuroSAT dataset under `data/EuroSAT`.
- Runs the full pipeline (I/O → frequency-domain processing → feature extraction).
- Prints total images, total time, and throughput.
- Writes:
  - `results/run_summary.txt`
  - `results/features.npy`

You can adjust `-n 4` to use a different number of MPI ranks.

## 3.1 Verify data and task parallelism

Code: report Listings 3.1–3.3 (Chapter 3). The commands below show the partition
across ranks and the per-batch stage overlap.

```bash
# Data parallelism: each MPI rank's disjoint slice of the dataset
mpirun -n 4 python3 -c "
import sys; sys.path.insert(0,'src')
from mpi4py import MPI
from dataset_loader import get_partitioned_paths
c=MPI.COMM_WORLD; r=c.Get_rank()
p=get_partitioned_paths('data/EuroSAT', r, c.Get_size(), comm=c)
print(f'Rank {r}: {len(p)} images  first={p[0].split(\"/\")[-1]}  last={p[-1].split(\"/\")[-1]}', flush=True)"

# Data parallelism: same work, more ranks, less time
mpirun -n 1 python3 src/main_mpi.py
mpirun -n 8 python3 src/main_mpi.py

# Task parallelism: timestamped proof that Load / GPU / CPU stages overlap
mpirun -n 1 python3 experiments/run_pipeline_timing.py

# Task parallelism: pipeline OFF vs ON (always restore to true afterwards)
sed -i '' 's/task_parallel_prefetch: true/task_parallel_prefetch: false/' config/pipeline.yaml
mpirun -n 4 python3 src/main_mpi.py | grep "Total Time"
sed -i '' 's/task_parallel_prefetch: false/task_parallel_prefetch: true/' config/pipeline.yaml
mpirun -n 4 python3 src/main_mpi.py > /dev/null   # untimed: OFF evicted the file cache
mpirun -n 4 python3 src/main_mpi.py | grep "Total Time"
```

## 4. Report experiments (Chapter 5)

With the venv active. Every script runs the full production pipeline under
`mpirun` and precedes each timed run with an untimed run of the same
configuration. The report's numbers were measured with 7 GB of RAM held, so that
each run reads the images from disk (report §5.2):

```bash
python experiments/hold_memory.py &      # start before the experiments
python experiments/hold_memory.py 9 --active &   # stricter: stays in RAM (use a freshly restarted machine)
kill %1                                  # stop afterwards
```

**Never run `run_pipeline_onoff_comparison.py` or `task_parallel_prefetch: false`
while the holder is running:** together they exhausted RAM and swap and restarted
the laptop (kernel panic). Keep the lid open and use `caffeinate -dimsu` for long
batches, because sleep invalidates timings. Keep at least 15 GB of disk free for swap.

```bash
python experiments/run_scaling_experiments.py 5        # Table 5.1, Figs 5.2/5.3  (~5 min)
python experiments/analyze_amdahl.py                   # Table 5.2: Amdahl fit from strong_scaling.csv
python experiments/run_sequential_baseline.py 3        # sequential stage split, §2.2 (~1 min)
python experiments/run_comparative_benchmark.py 3      # Table 5.10, Fig 5.7: sequential vs hybrid, same rounds (~4 min)
python experiments/run_stage_breakdown.py 8 5          # Table 5.7               (~1 min)
python experiments/run_partition_comparison.py 3       # cyclic vs contiguous, §5.6.2 (~2 min)
python experiments/run_pipeline_onoff_comparison.py    # Table 5.9               (~4 min)
python experiments/run_weak_scaling.py 4               # Table 5.3, Fig 5.4  (~2 min)
python experiments/run_openmp_scaling.py 3             # Table 5.6, Fig 5.6  (~3 min)
python experiments/run_gpu_scaling.py                  # Table 5.4, one forward sweep
python experiments/run_gpu_scaling.py --reverse        # one reverse sweep (8,4,2,1)
python experiments/run_gpu_stage_timing.py             # Table 5.5: GPU stages vs NumPy/OpenMP (no holder needed)
mpirun -n 1 python experiments/run_pipeline_timing.py  # Table 5.8: stage overlap, [VERIFIED] line
python experiments/verify_fft_against_numpy.py         # §2.1: every feature vs NumPy, P=1,2,4,8
python experiments/verify_dataset.py                   # §1.2: what the 54,597 files are
mpirun -n 8 python experiments/measure_mpi_collectives.py   # Table 4.1: cost of each collective
python experiments/run_openmp_fft_scaling.py 5         # §5.6.1: OpenMP on the CPU FFT path (no holder needed)
python scripts/generate_report_images.py               # regenerates the figures from results/*.csv
```

Outputs (in `results/`):

- `strong_scaling.csv`, `strong_scaling_raw_trials.csv`, `scaling_*ranks.log`
- `amdahl_fit.csv`
- `sequential_baseline.csv`, `comparative_benchmark.csv`
- `stage_breakdown.csv`
- `partition_comparison.csv`
- `pipeline_onoff_comparison.csv`, `pipeline_onoff_raw_trials.csv`
- `weak_scaling.csv`, `weak_scaling_raw_trials.csv`
- `openmp_scaling.csv`, `openmp_scaling_raw_trials.csv`
- `gpu_scaling.csv`, `gpu_scaling_raw_sweeps.csv`
- `gpu_stage_timing.csv`, `fft_verification.csv`, `openmp_fft_scaling.csv`
- `pipeline_timing_pipeline.csv`, `pipeline_timing_sequential.csv`

The exact CSVs behind the current report are frozen in `results/report_snapshot_final/`.

`results/adhoc_measurements.md` records the one-off measurements cited in the
report (file-cache state, run after sequential mode, efficiency-core decode speed, JPEG vs GeoTIFF load cost).

## 5. Inspect saved features

To quickly check the saved feature vector:

```bash
cd /Users/bahatipatrick/HpcFinalProject
source venv/bin/activate
python
```

Then in the Python prompt:

```python
import numpy as np

feats = np.load("results/features.npy")
print(feats.shape)
print(feats[:10])
```

Exit Python with:

```python
exit()
```

## 6. Weak-scaling experiments (proposal 0.7)

With the venv active, run:

```bash
python experiments/run_weak_scaling.py
```

By default each rank processes 1000 images (set `WEAK_SCALING_IMAGES_PER_RANK` to override). Results are written to `results/weak_scaling.csv`.

## 7. User-customizable workflow (proposal 0.5.5)

Edit `config/pipeline.yaml` and change `pipeline.stages` to enable, disable, or reorder stages. For example:

- Default: `[io, fft, filter, feature]`
- Without filtering: `[io, fft, feature]`
- A useless order, e.g. `[io, fft, feature, filter]`, runs but prints
  `[WARN] stage 'filter_stage' runs but none of its outputs is used by a later stage`
- An invalid order, e.g. `[fft, filter, feature]`, stops before loading anything:
  `[CONFIG ERROR] no earlier stage produces the required input: 'fft_stage' needs 'input_data'`

Then run the pipeline as in step 3; the task graph is built and validated from config at
runtime. Any stage list other than the default switches off the 3-thread pipeline. Restore
the default list afterwards. The DAG tests: `python -m pytest tests/test_task_graph.py -v`.

## 8. Compile the LaTeX report (if LaTeX is installed locally)

First, ensure a LaTeX distribution such as MacTeX is installed and `pdflatex` is available.

Then:

```bash
cd /Users/bahatipatrick/HpcFinalProject/report
pdflatex FINAL_COMPLETE_REPORT.tex
pdflatex FINAL_COMPLETE_REPORT.tex
```

This generates:

- `report/FINAL_COMPLETE_REPORT.pdf`

If you do not have LaTeX installed, upload `report/FINAL_COMPLETE_REPORT.tex` and `report/images/` to Overleaf and compile there.

