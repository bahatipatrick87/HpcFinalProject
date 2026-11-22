# Hybrid MPI + OpenCL + OpenMP Image-Processing Pipeline

**Course:** High Performance Computing, Università degli Studi di Messina, A.Y. 2025–2026
**Student:** Bahati Patrick (Matricola 573868)
**Supervisor:** Prof. Salvatore Distefano
**Report:** [report/FINAL_COMPLETE_REPORT.pdf](report/FINAL_COMPLETE_REPORT.pdf)
**Commands:** [USAGE.md](USAGE.md)

## Overview

The project processes the EuroSAT Sentinel-2 dataset (54,597 files, 3.05 GB) through a
four-stage pipeline and parallelises it at three levels:

| Level | Mechanism | Pattern |
|---|---|---|
| Across processes | MPI ranks on disjoint contiguous blocks of files | Data parallelism |
| Inside each rank | Three threads linked by bounded queues, over batches of 256 images | Task (pipeline) parallelism |
| Inside each stage | OpenCL work-items on the GPU, OpenMP threads on the CPU | Fine-grained data parallelism |

```
S1 Load + preprocess  ->  S2 FFT magnitude  ->  S3 High-pass filter  ->  S4 Feature extraction
      CPU (I/O)             GPU (OpenCL)           GPU (OpenCL)             CPU (OpenMP)
```

- **S2** `fft_magnitude` kernel: radix-2 Cooley–Tukey 2D FFT, row pass then column pass.
- **S3** `highpass_filter` kernel: H(u,v) = 1 − exp(−d²/50), one work-item per element.
- **S4** `#pragma omp parallel for schedule(static)`: mean of each filtered image.
- **Workflow:** stages are declared in `config/pipeline.yaml`, built into a task graph and
  validated (topological sort) before any data is loaded.

## Dataset

EuroSAT ([Kaggle](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)), unzipped into
`data/EuroSAT/` (not included in the repository):

| Part | Files | Format |
|---|---|---|
| RGB patches, 10 land-cover classes | 27,000 | JPEG, 64×64×3 |
| The same patches, multispectral | 27,000 | GeoTIFF, 64×64×13 |
| Additional patches (Jakarta, Bombay) | 597 | GeoTIFF, 64×64×13 |
| **Total** | **54,597** | **3.05 GB** |

Every file becomes a 64×64×3 float32 array: JPEG ÷ 255; GeoTIFF bands 4/3/2 ÷ 2000, clipped to
[0, 1]. `experiments/verify_dataset.py` prints this audit.

## Repository structure

```
src/
  main_mpi.py              entry point: MPI set-up, run, gather, timing
  dataset_loader.py        file discovery, broadcast, block partition, image loading
  pipeline_kernels.cpp/.h  C++ library: OpenCL kernels and host code, OpenMP loops
  CMakeLists.txt           build of the C++ library
  pipeline/
    batched_pipeline.py    3-thread queue pipeline (task parallelism)
    workflow_builder.py    YAML -> task graph (stage inputs and outputs)
    task_graph.py          DAG with Kahn topological sort and validation
    scheduler.py           sequential execution of the task graph
  stages/                  io, fft, filter and feature stages (ctypes calls into the library)
  utils/                   configuration, dataset path and timing helpers
config/pipeline.yaml       stage list, pipeline on/off, batch size
experiments/               one script per table or figure of the report
scripts/                   build.sh (library + environment), generate_report_images.py
tests/                     19 kernel tests + 8 task-graph tests
results/                   experiment CSVs
  report_snapshot_final/   frozen CSVs behind the report's tables and figures
  reproduction_20261103/   raw outputs of the repeat sessions (report §5.2)
  adhoc_measurements.md    one-off measurements cited in the report
report/                    LaTeX source, PDF and figures
lib/                       compiled library (libpipeline_kernels.dylib / .so)
```

## Requirements

- macOS (Apple Silicon) or Linux, with an OpenCL device
- Open MPI, CMake, a C++14 compiler, OpenMP (`libomp` on macOS)
- Python 3.12 with the packages in `requirements.txt`

Testbed used for the report: MacBook Air, Apple M4 (4 performance + 6 efficiency cores, 8-core GPU), 16 GB RAM,
Open MPI 5.0.8, Python 3.12.

## Quick start

```bash
./scripts/build.sh                    # build lib/libpipeline_kernels.*, create venv, install packages
source venv/bin/activate
python -m pytest tests -q             # 27 tests
mpirun -n 8 python3 src/main_mpi.py   # full dataset on 8 MPI ranks
```

All commands, in order, are in [USAGE.md](USAGE.md).

## Configuration (`config/pipeline.yaml`)

| Setting | Effect |
|---|---|
| `stages: [io, fft, filter, feature]` | Stage order; validated as a DAG at start-up |
| `task_parallel_prefetch: true` | 3-thread pipeline (default) |
| `task_parallel_prefetch: false` | Sequential mode: each stage over the rank's whole block |
| `io_batch_size: 256` | Images per batch |

An invalid order, for example `[fft, filter, feature]`, stops every rank before loading:
`[CONFIG ERROR] no earlier stage produces the required input: 'fft_stage' needs 'input_data'`.

## Main results (report Chapter 5)

Strong scaling, full dataset, mean of 5 interleaved rounds, images read from disk
(`results/report_snapshot_final/strong_scaling.csv`):

| P | T_P [s] | Speedup | Efficiency | Throughput [img/s] |
|---|---|---|---|---|
| 1 | 17.03 | 1.00× | 100 % | 3,206 |
| 2 | 9.17 | 1.86× | 92.9 % | 5,956 |
| 4 | 7.06 | 2.41× | 60.3 % | 7,732 |
| 8 | 5.37 | 3.17× | 39.7 % | 10,170 |

- Against the single-threaded NumPy baseline (22.61 s): 1.43× at P = 1 and 4.49× at P = 8.
- Amdahl fit: f = 0.217, speedup ceiling about 4.6×. The main limit is load imbalance
  (GeoTIFFs load more slowly than JPEGs and are concentrated in the middle ranks); the shared
  GPU is secondary.
- Communication: one broadcast of the file list and one gather of one float32 per image
  (213 KB instead of 2.68 GB of pixels).
- Correctness: every feature matches a NumPy reference to within 1e-6 at P = 1, 2, 4, 8.

Absolute times depend on the operating-system file cache (report §5.2): with the dataset cached,
a P = 1 run takes about 6.3 s instead of about 16 s.

## Reproducibility

Each experiment script overwrites its CSV in `results/`. The values used in the report are kept
in `results/report_snapshot_final/`; restore them with:

```bash
cp results/report_snapshot_final/*.csv results/
```
