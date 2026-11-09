"""
Naive sequential baseline (Algorithm 1 in the report): single Python thread,
no MPI, no OpenMP, no GPU/OpenCL. Pure NumPy per-image loop implementing
Equations (2.1)-(2.4): load -> FFT magnitude -> high-pass filter -> mean.

Used to produce a *measured* (not guessed) reference point for the
"Sequential (1 CPU thread, no GPU)" row of the comparative-benchmarking
table. Stage 1 uses the project's own loader (load_images_from_paths), so
both EuroSAT formats are read exactly as the hybrid pipeline reads them:
the 27,000 RGB JPEGs and the 27,597 13-band GeoTIFFs. By default the whole
dataset is processed, so no extrapolation is involved.

Usage:
    python3 experiments/run_sequential_baseline.py [n_trials]
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_loader import _discover_files, load_images_from_paths  # noqa: E402

RESULTS_DIR = ROOT / "results"


def highpass_mask(h, w):
    # Distance from the DC term, which np.fft.fft2 places at (0, 0) with
    # wrap-around; same mask as the OpenCL highpass_filter kernel.
    yy, xx = np.mgrid[0:h, 0:w]
    dy = np.minimum(yy, h - yy)
    dx = np.minimum(xx, w - xx)
    return 1.0 - np.exp(-(dx ** 2 + dy ** 2) / 50.0)


def process_one(path, H, timings):
    t0 = time.perf_counter()
    arr = load_images_from_paths([path])[0]                   # Eq. 2.1
    t1 = time.perf_counter()
    spec = np.abs(np.fft.fft2(arr, axes=(0, 1)))              # Eq. 2.2
    t2 = time.perf_counter()
    filtered = spec * H[:, :, None]                           # Eq. 2.3
    t3 = time.perf_counter()
    feat = float(np.mean(filtered))                           # Eq. 2.4
    t4 = time.perf_counter()
    timings["load"] += t1 - t0
    timings["fft"] += t2 - t1
    timings["filter"] += t3 - t2
    timings["feature"] += t4 - t3
    return feat


def run_once(files, H):
    timings = {"load": 0.0, "fft": 0.0, "filter": 0.0, "feature": 0.0}
    t0 = time.perf_counter()
    for p in files:
        process_one(p, H, timings)
    return time.perf_counter() - t0, timings


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    files = _discover_files(str(ROOT / "data" / "EuroSAT"))
    if not files:
        print("No images found under data/EuroSAT")
        return
    n_tif = sum(f.lower().endswith((".tif", ".tiff")) for f in files)
    print(f"Sequential baseline: {len(files)} images "
          f"({len(files) - n_tif} JPEG, {n_tif} GeoTIFF), {n_trials} trial(s)")

    H = highpass_mask(64, 64)
    print("  untimed pass (settles the OS file cache)...")
    run_once(files, H)
    rows = []
    for trial in range(1, n_trials + 1):
        elapsed, timings = run_once(files, H)
        rows.append((trial, elapsed, timings))
        print(f"  trial {trial}: {elapsed:.2f} s ({len(files)/elapsed:.1f} img/s)")

    total = sum(sum(t.values()) for _, _, t in rows)
    print("\nStage breakdown (all trials):")
    for stage in ("load", "fft", "filter", "feature"):
        s = sum(t[stage] for _, _, t in rows)
        print(f"  {stage:8s}: {100 * s / total:5.1f} %")

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / "sequential_baseline.csv").open("w") as f:
        f.write("trial,elapsed_seconds,load_s,fft_s,filter_s,feature_s\n")
        for trial, elapsed, t in rows:
            f.write(f"{trial},{elapsed:.4f},{t['load']:.4f},{t['fft']:.4f},"
                    f"{t['filter']:.4f},{t['feature']:.4f}\n")


if __name__ == "__main__":
    main()
