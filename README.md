# Hybrid Parallel Image Processing Pipeline

**Course:** High Performance Computing — Università degli Studi di Messina  
**Student:** Bahati Patrick (Matricola 573868)  
**Supervisor:** Prof. Salvatore Distefano  
**Academic Year:** 2025–2026

---

## Overview

A hybrid **MPI + OpenCL + OpenMP** parallel image processing pipeline for large-scale remote-sensing imagery (EuroSAT, 54,597 image files of 64×64 pixels: the benchmark's 27,000 patches as RGB JPEGs and again as 13-band GeoTIFFs, of which bands 4/3/2 are used, plus 597 further GeoTIFFs named Jakarta and Bombay; see report §1.2). The system implements **three levels of parallelism**:

| Level | Mechanism | Type |
|---|---|---|
| Across MPI ranks | Same 4-stage pipeline on disjoint image subsets | **Data parallelism** (primary) |
| Across successive batches | 3-thread producer–consumer queue | **Task / pipeline parallelism** |
| Inside each stage | OpenCL work-items (GPU) / OpenMP threads (CPU) | Fine-grained data parallelism |

### Four-Stage Pipeline

```
S1: Load & Preprocess  →  S2: FFT Magnitude  →  S3: High-Pass Filter  →  S4: Feature Extract
      CPU (I/O)               GPU (OpenCL)          GPU (OpenCL)             CPU (OpenMP)
```

- **S2** uses the `fft_magnitude` OpenCL kernel (real 2D radix-2 FFT, row pass then column pass, magnitude spectrum)
- **S3** uses the `highpass_filter` OpenCL kernel (H(u,v) = 1 − exp(−d²/50))
- **S4** uses `#pragma omp parallel for` reduction over batch elements

### Task / Pipeline Parallelism (Default)

With `task_parallel_prefetch: true` (the default), each MPI rank runs three threads:

```
Thread 1 (I/O):  Load batch k+1           |  Load batch k+2  |  ...
Thread 2 (GPU):               FFT+Filter k |  FFT+Filter k+1  |  ...
Thread 3 (CPU):                            |  Features k       |  ...
```

Successive batches overlap in wall-clock time — textbook pipeline/task parallelism.

---

## Project Structure

```
HpcFinalProject/
├── src/
│   ├── pipeline_kernels.cpp/.h   # C++ library: 2 OpenCL GPU kernels + OpenMP feature extraction
│   ├── main_mpi.py               # Primary MPI entry point
│   ├── dataset_loader.py         # EuroSAT loader + MPI block partitioning
│   ├── CMakeLists.txt            # Build configuration
│   ├── pipeline/
│   │   ├── batched_pipeline.py   # 3-thread queue pipeline (task parallelism)
│   │   ├── workflow_builder.py   # YAML → TaskGraph builder (stage inputs/outputs, validation)
│   │   ├── scheduler.py          # Sequential DAG runner
│   │   └── task_graph.py         # DAG: dependency edges + Kahn topological sort
│   ├── stages/
│   │   ├── io_stage.py           # S1: load + MPI partition
│   │   ├── fft_stage.py          # S2: calls process_images_fft (GPU kernel)
│   │   ├── filter_stage.py       # S3: calls apply_frequency_filter_gpu (GPU kernel)
│   │   └── feature_stage.py      # S4: calls extract_features_cpu (OpenMP)
│   └── utils/
│       ├── config_loader.py      # YAML config reader
│       ├── dataset_utils.py      # Dataset path resolution
│       └── timing.py             # Wall-clock timing context manager
├── config/
│   └── pipeline.yaml             # User-customizable workflow (YAML DAG)
├── experiments/
│   ├── run_scaling_experiments.py      # Strong scaling, P = 1,2,4,8 (Table 5.1)
│   ├── analyze_amdahl.py               # Amdahl fit from strong scaling (Table 5.2)
│   ├── run_weak_scaling.py             # Weak scaling (Table 5.3)
│   ├── run_gpu_scaling.py              # GPU concurrency, --reverse for reverse sweep (Table 5.4)
│   ├── run_gpu_stage_timing.py         # GPU stages vs NumPy/OpenMP FFT, host clock (Table 5.5)
│   ├── verify_gpu_breakdown.py         # Driver timestamps vs host clock, one dispatch (§5.5.3)
│   ├── verify_gpu_profiling.py         # Driver evidence of GPU execution, full dataset (§5.5.3)
│   ├── verify_fft_against_numpy.py     # Every feature vs a NumPy reference at each P (§2.1)
│   ├── verify_dataset.py               # What the 54,597 files are (§1.2)
│   ├── measure_mpi_collectives.py      # Cost of each MPI collective (Table 4.1)
│   ├── hold_memory.py                  # Holds 7 GB so runs read images from disk (§5.2)
│   ├── run_openmp_scaling.py           # OpenMP thread sweep, pipeline (Table 5.6)
│   ├── run_openmp_fft_scaling.py       # OpenMP thread sweep, CPU FFT path (§5.6.1)
│   ├── run_stage_breakdown.py          # Per-stage busy time per rank (Table 5.7)
│   ├── run_partition_comparison.py     # Contiguous vs cyclic split (§5.6.2)
│   ├── run_pipeline_timing.py          # Stage-overlap proof (Table 5.8)
│   ├── run_pipeline_onoff_comparison.py # Pipeline ON vs OFF (Table 5.9)
│   ├── run_comparative_benchmark.py    # Sequential vs hybrid, same rounds (Table 5.10)
│   └── run_sequential_baseline.py      # Single-threaded NumPy baseline (no MPI/GPU/OpenMP)
├── scripts/
│   ├── generate_report_images.py  # Generates all PNG figures for the report
│   └── build.sh                   # Build the C++ shared library
├── report/
│   ├── FINAL_COMPLETE_REPORT.tex  # LaTeX report (compile with pdflatex twice)
│   └── images/                    # PNG figures for the report
│       ├── system_architecture.png, parallel_patterns.png, mapping_hierarchy.png
│       ├── strong_scaling_optimized.png, amdahl_sensitivity.png, weak_scaling.png
│       └── gpu_concurrency.png, openmp_scaling.png, comparative_benchmark.png
├── lib/
│   └── libpipeline_kernels.dylib  # Compiled shared library (macOS .dylib / Linux .so)
├── results/                       # Experiment outputs (CSV data files)
│   └── report_snapshot_final/     # Exact CSVs behind the report's tables and figures
└── tests/
    ├── test_correctness.py        # Unit tests for C++ kernel correctness
    └── test_task_graph.py         # Unit tests for the pipeline DAG
```

---

## Build and Run

### 1. Build the C++ Library

```bash
./scripts/build.sh
# Configures CMake from src/ into build/, compiles, copies the library to
# lib/libpipeline_kernels.dylib (macOS) or .so (Linux) and code-signs it on macOS
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
# numpy, mpi4py, Pillow, tifffile (13-band GeoTIFFs), PyYAML, matplotlib, pyopencl (verify_gpu_*.py only)
```

### 3. Run the Pipeline

```bash
# Single process (pipeline parallelism active via 3-thread queue)
python3 src/main_mpi.py

# 4 MPI ranks (data + pipeline parallelism)
mpirun -n 4 python3 src/main_mpi.py

# Full 8-rank run (report Table 5.1)
mpirun -n 8 python3 src/main_mpi.py
```

### 4. Run Experiments

See `RUN_COMMANDS.md` §4 for every report experiment with its arguments and run time.
Each timed run follows an untimed run of the same configuration. For the report's numbers run
them under `experiments/hold_memory.py` (except pipeline ON/OFF: together they exhaust memory).

```bash
python3 experiments/run_scaling_experiments.py 5    # strong scaling
python3 experiments/analyze_amdahl.py               # Amdahl fit
python3 experiments/run_comparative_benchmark.py 3  # sequential vs hybrid
python3 experiments/run_pipeline_timing.py          # stage-overlap proof
python -m pytest tests -v                           # 27 tests (19 kernel + 8 DAG)
```

### 5. Regenerate Report Images

```bash
python3 scripts/generate_report_images.py
# → report/images/*.png
```

### 6. Compile Report

```bash
cd report
pdflatex FINAL_COMPLETE_REPORT.tex
pdflatex FINAL_COMPLETE_REPORT.tex   # second pass for cross-references
```

---

## Configuration (`config/pipeline.yaml`)

```yaml
pipeline:
  stages:                              # DAG nodes — validated at start-up
    - io
    - fft
    - filter
    - feature
  task_parallel_prefetch: true         # 3-thread pipeline (task parallelism) — default
  io_batch_size: 256                   # Images per batch chunk
```

| Option | Effect |
|---|---|
| `task_parallel_prefetch: true` | 3-thread queue pipeline overlapping Load \|\| GPU \|\| CPU |
| `task_parallel_prefetch: false` | Sequential stages over each rank's whole subset (data parallelism only) |
| Remove `filter` from `stages` | Skip GPU high-pass filter (features from the FFT output) |
| Invalid order, e.g. `[fft, filter, feature]` | Stops before loading: `[CONFIG ERROR] … 'fft_stage' needs 'input_data'` |
| `io_batch_size: N` | Tune for GPU occupancy / memory |

---

## Performance Results (EuroSAT, 54,597 images)

Average of 5 interleaved rounds, images read from disk (`experiments/run_scaling_experiments.py`
under `experiments/hold_memory.py`; `results/strong_scaling.csv`):

| MPI Ranks | Time [s] | Fastest rank [s] | Speedup | Efficiency | Throughput [img/s] |
|---|---|---|---|---|---|
| 1 | 17.03 | 17.03 | 1.00× | 100% | 3,206 |
| 2 | 9.17 | 8.19 | 1.86× | 92.9% | 5,956 |
| 4 | 7.06 | 3.16 | 2.41× | 60.3% | 7,732 |
| 8 | 5.37 | 2.60 | 3.17× | 39.7% | **10,170** |

When free RAM lets the OS cache the 3 GB dataset, P=1 falls to about 6.3 s (report §5.2).
At P=2 each rank gets an even JPEG/GeoTIFF mix and scaling is near-linear. From P=4 speedup is
limited mainly by load imbalance: the file list runs 14,000 JPEGs, 27,597 13-band GeoTIFFs,
13,000 JPEGs, so the contiguous split gives the middle ranks mainly GeoTIFFs, which take more
than twice as long to load. The shared GPU is a secondary factor. `experiments/run_partition_comparison.py` measures a cyclic split as the fix.  
MPI communication: `Gather` payload = **213 KB** (one `float32` per image vs 2.68 GB raw pixels).

Measured against a single-threaded sequential baseline on the full dataset
(pure NumPy, no MPI/GPU/OpenMP, same image loader; `experiments/run_sequential_baseline.py`,
timed in the same rounds as the hybrid runs by `experiments/run_comparative_benchmark.py`:
22.61 s), the hybrid pipeline is 1.43× faster at P=1 and 4.49× faster at P=8. Every feature
matches a NumPy reference to within 1e-6 (`experiments/verify_fft_against_numpy.py`). At the
production batch of 256 the GPU FFT is about 4.5× faster than NumPy and 2.1× faster than the
OpenMP path, but the Load stage is the bottleneck. See the report, Sections 5.5.3 and 5.7.

---

## Parallelism Architecture

| Pattern | Mechanism | Active |
|---|---|---|
| Data parallelism | MPI block partition across P ranks | Always |
| Task/pipeline parallelism | 3-thread queue: Load ∥ GPU ∥ CPU | Default (config) |
| Fine-grained (GPU) | OpenCL work-items over rows, columns, elements | Inside S2, S3 |
| Fine-grained (CPU) | OpenMP `parallel for` | Inside S4 |
| Functional decomposition | CPU↔GPU stage mapping | Always |
