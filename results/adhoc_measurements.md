# One-off measurements cited in the report

These measurements back statements in report §1.2, §4.1, §4.3, §5.1.2, §5.2, §5.6.2 and §5.6.4. They were
taken on the testbed (MacBook Air, Apple M4) and are not produced by an
experiment script. Run from the project
root with the venv active.

## 0. File-cache state: disk traffic per run (2026-11-01, §5.2)

`iostat -d -I disk0` before and after `mpirun -n 1 python3 src/main_mpi.py`:

| Condition | Time per run | Disk traffic per run |
|---|---|---|
| Free RAM, images cached | 6.40-6.42 s | 2-15 MB |
| 11 GB of RAM held, 5 consecutive runs | 14.61-14.74 s | 2,658-2,671 MB |

Holding amount (incompressible data, P=1/2/4/8): 7 GB gives about 3.1 GB of disk traffic per
run (the dataset is 3.05 GB) with swap growing 68 MB; 8 and 9 GB grow swap by more than 550 MB.
Running pipeline-OFF mode under the holder caused kernel panics on 2026-11-01 at 11:11
and 11:17 (`watchdog timeout`, "LOW swap space"), so that experiment runs without it.

## 1. Run directly after a sequential-mode run (2026-11-01, real FFT kernel; §5.6.4)

A pipeline run directly after a sequential-mode run is slow, because the
sequential run evicts the dataset from the OS file cache. The runs that
follow it are fast again. No memory holder is running.

```bash
sed -i '' 's/task_parallel_prefetch: true/task_parallel_prefetch: false/' config/pipeline.yaml
/usr/bin/time -l mpirun -n 1 python3 src/main_mpi.py | grep -E 'Total Time|maximum resident'
sed -i '' 's/task_parallel_prefetch: false/task_parallel_prefetch: true/' config/pipeline.yaml
for i in 1 2 3 4 5 6; do mpirun -n 1 python3 src/main_mpi.py | grep 'Total Time'; done
```

Measured:
- Pipeline (ON) runs before it: 6.21 and 6.26 s, peak resident memory 140-153 MB.
- Sequential-mode (OFF) run, starting with the images cached: 10.99 s, peak resident memory 4,833,820,672 B (4.8 GB).
- Pipeline (ON) runs after it: 17.40 s, then 6.32, 6.28, 6.30 and 6.31 s.
- A sequential-mode run inside the ON/OFF experiment reached 5.47 GB resident (15.43 s in that run), so the report says "about 5 GB".

## 2. Efficiency-core decode speed (§5.1.2, Level 1)

`taskpolicy -c background` confines a process to the efficiency cores.
The benchmark loads 3,000 JPEGs and 3,000 GeoTIFFs with
`dataset_loader.load_images_from_paths` and prints the cost per image.

| Run | JPEG (µs/image) | GeoTIFF (µs/image) |
|---|---|---|
| Default scheduling, run 1 | 115 | 224 |
| Default scheduling, run 2 | 77 | 157 |
| `taskpolicy -c background`, run 1 | 363 | 705 |
| `taskpolicy -c background`, run 2 | 233 | 840 |

Loading on the efficiency cores is about 3x slower.

## 3. GeoTIFF vs JPEG load cost (§3.3, §5.6.2)

The loader was run on 2,000 random files of each type, single process:

| Type | µs/image |
|---|---|
| JPEG | 204 |
| GeoTIFF | 439 |

In this single-process test a GeoTIFF cost about 2.15x as much to load as a JPEG. Read from
disk at P=8: in `stage_breakdown.csv` Load takes 2.41 s on JPEG ranks and 5.65 s on GeoTIFF ranks.

## 4. Size on disk (2026-11-01, §5.6.2)

Mean file size: JPEG 3,402 B, GeoTIFF 107,240 B (31.5x). Total 3.05 GB.

## 5. Stage 4 share (§5.6.1)

`extract_features_cpu` on the first 20,000 files: 45 ms, about 2 % of the 2.1 s run.


## 6. What the 54,597 files are (2026-11-03, §1.2)

`python experiments/verify_dataset.py`:

```
Files: 27000 JPEG + 27597 GeoTIFF = 54597
  AnnualCrop              3000 JPEG  3000 GeoTIFF
  Forest                  3000 JPEG  3000 GeoTIFF
  HerbaceousVegetation    3000 JPEG  3000 GeoTIFF
  Highway                 2500 JPEG  2500 GeoTIFF
  Industrial              2500 JPEG  2500 GeoTIFF
  Pasture                 2000 JPEG  2000 GeoTIFF
  PermanentCrop           2500 JPEG  2500 GeoTIFF
  Residential             3000 JPEG  3000 GeoTIFF
  River                   2500 JPEG  2500 GeoTIFF
  SeaLake                 3000 JPEG  3597 GeoTIFF
Extra GeoTIFFs: 597 {'Bombay': 85, 'Jakarta': 512}, all in ['SeaLake']
Dataset split lists: 597 of the extras listed; files not listed: ['SeaLake/SeaLake_2335.tif', 'SeaLake/SeaLake_509.tif']
Readable shapes: {'GeoTIFF (64, 64, 13) uint16': 27597, 'JPEG (64, 64) RGB': 27000}
Files with identical pixels: 0 groups
Same-name JPEG/GeoTIFF pairs: 27000, correlation median 0.997, 95.8% above 0.8, minimum 0.404
Control, random pairs: median |r| 0.071, 0.0% above 0.8
Best match in class is the same-name GeoTIFF: 500 of 500 random JPEGs
UTM zones, benchmark GeoTIFFs: {'27N': 50, '28N': 45, '29N': 988, '30N': 4975, '31N': 6318, '32N': 5620, '33N': 4391, '34N': 2284, '35N': 1905, '36N': 424}
UTM zones, extra GeoTIFFs: {'43N': 85, '48S': 512}
```

The published benchmark is 27,000 patches in RGB and multispectral versions. The origin of the 597 extra files in this copy was not established; they are listed in the copy's own split CSVs.

## 7. Disk traffic per experiment under 7 GB held (2026-11-03, §5.2)

`iostat -d -I disk0` around the second of two identical runs. Traffic includes any swap, and swap grew to about 5.4 GB in this session, so this shows the state of each experiment rather than the exact dataset read.

```
full P=1: time 18.0449s, disk read+write 3408 MB
full P=8: time 5.8482s, disk read+write 3648 MB
weak-1000 P=1: time 0.1640s, disk read+write 44 MB
weak-1000 P=8: time 0.4132s, disk read+write 145 MB
gpu-5000 P=1: time 0.4537s, disk read+write 36 MB
gpu-5000 P=2: time 0.4830s, disk read+write 9 MB
gpu-5000 P=8: time 3.5899s, disk read+write 2808 MB
openmp-20000 P=1: time 1.9905s, disk read+write 20 MB
```

The full-dataset and P=8 GPU runs read from disk; the subsets stay mostly cached. P=4 of the GPU experiment was not measured. The rerun of the partition experiment with P=2 was abandoned: free disk had fallen to 2.3 GB with 5 GB of swap on battery, so §5.6.2 keeps the 2026-11-01 data (P=4, P=8).

## 8. Host-clock kernel time at batch 64, alternating methods (2026-11-03, §5.5.3)

Three alternations in one session, no held memory:
- `verify_gpu_breakdown.py` (PyOpenCL): 2.47, 2.07, 2.28 ms.
- `run_gpu_stage_timing.py` (production C++ path, FFT kernel): 4.29, 1.73, 1.74 ms.

## 9. Testbed identification (2026-11-03, §5.1.1)

`system_profiler`: MacBook Air, Mac16,12, MC6T4T/A, Apple M4, 10 CPU cores (4P + 6E), 8 GPU cores, 16 GB, APPLE SSD AP0256Z. macOS 26.6.2, Open MPI 5.0.8, mpi4py 4.1.1, Python 3.12.12, Apple clang 21.0.0. The 120 GB/s is Apple's specification (https://support.apple.com/en-us/122209), not measured.

## 10. Re-run of the project (2026-11-03)

Raw outputs: `results/reproduction_20261103/` (`cached_*` = without held memory). The report's own data stays in `results/report_snapshot_final/`.

| Experiment | Report | Re-run | State of re-run |
|---|---|---|---|
| Tests | 27 pass | 27 pass | - |
| FFT verification | max 1.00e-6, mean 4.66e-8 | identical | - |
| Sequential vs pipeline features, 3,000 files | - | max 3.96e-7 | - |
| Pipeline ON/OFF | 3.49x, 2.32x | 3.37x, 2.51x | same |
| Weak scaling [s] | 0.180, 0.188, 0.258, 0.383 | 0.168, 0.180, 0.234, 0.366 | same (cached) |
| OpenMP pipeline [s] | 2.131-2.159 | 2.132-2.154 | same (cached) |
| GPU concurrency P=1 [img/s] | 10,325 | 10,222 | same (cached) |
| OpenMP FFT at 10 threads | 5.03x | 4.64x | same |
| GPU FFT call at 256 [ms] | 5.61 | 5.83 | same |
| Strong scaling T [s] | 17.03, 9.17, 7.06, 5.37 | 6.80, 3.80, 3.35, 2.64 | cached, not comparable |
| Sequential [s] | 22.61 | 13.48 | cached, not comparable |
| GPU concurrency P=8 [img/s] | 8,489 | 18,860 | cached, shows the disk effect |

Not re-run in the read-from-disk state in this session: strong scaling, sequential comparison, stage breakdown, partition. A held-memory run needs more free disk: the watchdog stopped the ON/OFF run when free disk fell below 3 GB. A single full run under the holder gave 18.04 s at P=1 and 5.85 s at P=8 (section 7).

Sequential baseline: `process_one` against production features on 3,000 random files, relative difference max 3.96e-7, mean 4.49e-8; same file order; `np.fft.fft2` single precision (complex64, NumPy 2.4.1).

## 11. Read-from-disk re-run after a restart (2026-11-03, §5.2)

Files in `results/reproduction_20261103/`: `disk_*` = 7 GB idle holder, discarded because the state drifted (T1 by round: 11.89, 12.91, 13.44, 7.35, 6.34 s; swap reached 2.3 GB). `disk9_*` = 9 GB held and kept active; 5 rounds, T1 12.90-13.23 s.

| Quantity | Report | 9 GB active | Cached (`cached_*`) |
|---|---|---|---|
| T1, T2, T4, T8 [s] | 17.03, 9.17, 7.06, 5.37 | 13.05, 7.36, 6.49, 5.96 | 6.80, 3.80, 3.35, 2.64 |
| S2, S4, S8 | 1.86, 2.41, 3.17 | 1.77, 2.01, 2.19 | 1.79, 2.03, 2.58 |
| f, ceiling | 0.217, 4.60x | 0.379, 2.64x | 0.300, 3.33x |
| Karp-Flatt P=2,4,8 | 0.076, 0.220, 0.217 | 0.128, 0.330, 0.379 | 0.116, 0.323, 0.300 |
| Sequential [s] | 22.61 | 19.76 | 13.48 |
| Load, FFT share | 69.8, 26.2 % | 66.5, 29.0 % | 49.9, 43.3 % |
| Hybrid vs sequential, P=1, P=8 | 1.43x, 4.49x | 1.50x, 3.31x | 1.98x, 5.11x |
| Weak efficiency P=2,4,8 | 95.5, 69.7, 46.9 % | 89.0, 68.0, 36.7 % | 93.4, 71.7, 45.8 % |
| OpenMP pipeline [s] | 2.13-2.16 | 1.98-1.99 | 2.13-2.15 |
| GeoTIFF rank Load, GPU share | 98, 12 % | 98, 19 % | 97, 12 % (critical rank) |
| Cyclic gain P=4, P=8 | 35, 28 % | 38, 35 % | 23, 12 % |
| GPU concurrency [img/s] | 10,325, 14,606, 10,800, 8,489 | 11,136, 20,120, 13,722, 9,002 | 10,222, 17,677, 14,257, 18,860 |

The report keeps its 2026-11-01 tables and states the range in §5.2. Disk traffic with 9 GB active:

```
full P=1: time 16.3651s, disk 3055 MB
full P=8: time 4.8270s, disk 2987 MB
weak-1000 P=8: time 0.7164s, disk 15 MB
openmp-20000 P=1: time 2.0059s, disk 59 MB
gpu-5000 P=1: time 0.4453s, disk 0 MB
gpu-5000 P=2: time 0.4761s, disk 1 MB
gpu-5000 P=4: time 1.1556s, disk 7 MB
gpu-5000 P=8: time 4.6173s, disk 2194 MB
```
The strong-scaling rounds gave T1 about 13 s although a check just before gave 16.4 s with 3,055 MB read, so the cache state still moved between phases. Not resolved.

## 12. MPI collectives and in-flight memory (2026-11-03, §3.2.2, §4.1, §4.3)

Code: one `comm.Barrier` (`main_mpi.py`), one `comm.bcast` (`dataset_loader.py`), two `comm.gather` (`main_mpi.py`: features, per-rank times); no point-to-point calls.

`mpirun -n 8 python experiments/measure_mpi_collectives.py`, three runs, slowest rank, mean of 20: Barrier 0.006 ms; bcast of the file list 8.67-9.10 ms; gather of features 0.055-0.089 ms; gather of rank times 0.004-0.006 ms.

Dispatches per stage: 214 at P=1 and P=2, 216 at P=4 and P=8; at batch 64, 854 at P=1 and 856 at P=8.

In-flight image data estimate: q1 3 x 12.58 MB + q2 3 x 37.75 MB + I/O thread 12.58 MB + GPU thread 37.75 MB + CPU thread 37.75 MB = 239 MB. Peak resident memory of P=1 pipeline runs: 116-153 MB (section 1).
