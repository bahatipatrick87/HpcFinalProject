# Results Directory

- `*.csv`: outputs of the experiments in `experiments/` (see `RUN_COMMANDS.md`, section 4).
  `scripts/generate_report_images.py` reads these to draw the report's plots.
- `report_snapshot_final/`: frozen copy of the CSVs behind the report. Running an experiment
  overwrites the CSV above it; restore with `cp results/report_snapshot_final/*.csv results/`.
- `adhoc_measurements.md`: one-off measurements cited in the report.
- `reproduction_20261103/`: raw outputs of the re-runs summarised in `adhoc_measurements.md`, sections 10 and 11.

Created when you run things, and safe to delete: `features.npy` and `run_summary.txt`
(from `src/main_mpi.py`), `logs/`, `gpu_scaling_logs/` and `scaling_*ranks.log` (from the experiments).
