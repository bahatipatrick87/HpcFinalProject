# Usage

Run every command from the project root.

## 1. Setup

```bash
./scripts/build.sh                     # build the C++ library into lib/, create venv, install packages
source venv/bin/activate               # activate the Python environment
python -m pytest tests -q              # 27 tests: 19 kernels + 8 task graph
```

## 2. Dataset

```bash
ls data/EuroSAT/EuroSAT                # 10 class folders + allBands (GeoTIFFs)
python3 experiments/verify_dataset.py  # file counts and formats: 27,000 JPEG + 27,597 GeoTIFF
```

## 3. Sequential baseline

```bash
python3 experiments/run_sequential_baseline.py 1   # 1 thread, NumPy only; prints time and stage split
```

## 4. Hybrid parallel pipeline

```bash
mpirun -n 1 python3 src/main_mpi.py    # 1 rank: GPU + OpenMP + 3-thread pipeline
mpirun -n 2 python3 src/main_mpi.py    # 2 ranks
mpirun -n 4 python3 src/main_mpi.py    # 4 ranks
mpirun -n 8 python3 src/main_mpi.py    # 8 ranks: total time = slowest rank
```

## 5. Correctness

```bash
cp results/features.npy /tmp/f8.npy && mpirun -n 1 python3 src/main_mpi.py > /dev/null   # keep P=8 features, rerun with P=1
python3 -c "import numpy as np; a=np.load('/tmp/f8.npy'); b=np.load('results/features.npy'); print(a.shape, b.shape, 'match:', np.allclose(a, b, rtol=1e-5))"   # P=8 equals P=1
python3 experiments/verify_fft_against_numpy.py 8   # every feature vs a NumPy reference
```

## 6. Data parallelism: partition across ranks

```bash
mpirun -n 4 python3 -c "
import sys; sys.path.insert(0, 'src')
from mpi4py import MPI
from dataset_loader import get_partitioned_paths
c = MPI.COMM_WORLD; r = c.Get_rank()
p = get_partitioned_paths('data/EuroSAT', r, c.Get_size(), comm=c)
print(f'Rank {r}: {len(p)} images  first={p[0].split(\"/\")[-1]}  last={p[-1].split(\"/\")[-1]}', flush=True)"
# each rank prints its contiguous block of files
```

## 7. Task parallelism

```bash
mpirun -n 1 python3 experiments/run_pipeline_timing.py   # per-batch timestamps; prints [VERIFIED] Pipeline overlap

sed -i '' 's/task_parallel_prefetch: true/task_parallel_prefetch: false/' config/pipeline.yaml   # pipeline OFF
mpirun -n 4 python3 src/main_mpi.py | grep "Total Time"                                         # sequential mode
sed -i '' 's/task_parallel_prefetch: false/task_parallel_prefetch: true/' config/pipeline.yaml   # pipeline ON (restore)
mpirun -n 4 python3 src/main_mpi.py > /dev/null                                                 # untimed run: refills the file cache
mpirun -n 4 python3 src/main_mpi.py | grep "Total Time"                                         # pipeline mode
```

## 8. Configurable workflow (DAG validation)

```bash
cp config/pipeline.yaml /tmp/pipeline.yaml.bak       # back up the configuration
sed -i '' '/^    - io$/d' config/pipeline.yaml        # remove the io stage: invalid order
mpirun -n 1 python3 src/main_mpi.py                  # stops with [CONFIG ERROR] before loading data
cp /tmp/pipeline.yaml.bak config/pipeline.yaml       # restore the configuration
python -m pytest tests/test_task_graph.py -v         # 8 task-graph tests
```

## 9. Report experiments (Chapter 5)

```bash
python3 experiments/hold_memory.py &                       # hold 7 GB so runs read images from disk (§5.2)
python3 experiments/run_scaling_experiments.py 5           # strong scaling, Table 5.1, Figures 5.2–5.3
python3 experiments/analyze_amdahl.py                      # Amdahl fit and Karp–Flatt, Table 5.2
python3 experiments/run_comparative_benchmark.py 3         # sequential vs hybrid, Table 5.11, Figure 5.7
python3 experiments/run_sequential_baseline.py 3           # sequential stage split, §2.2
python3 experiments/run_stage_breakdown.py 8 5             # per-rank stage times, Table 5.7
python3 experiments/run_partition_comparison.py 3          # contiguous vs cyclic split, Table 5.8
python3 experiments/run_gpu_scaling.py                     # GPU concurrency, forward sweep, Table 5.4
python3 experiments/run_gpu_scaling.py --reverse           # GPU concurrency, reverse sweep
kill %1                                                    # stop the memory holder

python3 experiments/run_weak_scaling.py 4                  # weak scaling, Table 5.3, Figure 5.4
python3 experiments/run_openmp_scaling.py 3                # OpenMP threads in the pipeline, Table 5.6, Figure 5.6
python3 experiments/run_openmp_fft_scaling.py 5            # OpenMP threads on the CPU FFT path, §5.6.1
python3 experiments/run_gpu_stage_timing.py                # GPU FFT vs NumPy vs OpenMP, Table 5.5
python3 experiments/verify_gpu_profiling.py                # driver evidence of GPU execution, §5.5.3
python3 experiments/verify_gpu_breakdown.py                # driver timestamps vs host clock, §5.5.3
mpirun -n 1 python3 experiments/run_pipeline_timing.py     # stage overlap, Table 5.9
python3 experiments/run_pipeline_onoff_comparison.py       # pipeline ON vs OFF, Table 5.10 (never with hold_memory.py)
mpirun -n 8 python3 experiments/measure_mpi_collectives.py # cost of each MPI collective, Table 4.1
python3 experiments/verify_fft_against_numpy.py            # all features vs NumPy at P = 1, 2, 4, 8
```

## 10. Figures, report and results

```bash
python3 scripts/generate_report_images.py            # redraw report/images/*.png from results/*.csv
cd report && pdflatex FINAL_COMPLETE_REPORT.tex && pdflatex FINAL_COMPLETE_REPORT.tex && cd ..   # build the PDF
cp results/report_snapshot_final/*.csv results/      # restore the CSVs used in the report
```
