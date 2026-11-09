"""
End-to-end correctness check of the hybrid pipeline.

Runs the production pipeline under mpirun at each P and compares every image's
feature with one computed by plain NumPy (the sequential baseline's
np.fft.fft2, mask and mean) on the same image. They must agree to single
precision, since the pipeline's GPU kernels compute the same transform.

Usage:
    python experiments/verify_fft_against_numpy.py [P ...]

Output: results/fft_verification.csv
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from dataset_loader import _discover_files, load_images_from_paths  # noqa: E402
from run_sequential_baseline import highpass_mask  # noqa: E402


def numpy_features(files):
    H = highpass_mask(64, 64)[:, :, None]
    out = np.empty(len(files), dtype=np.float64)
    for i in range(0, len(files), 256):
        img = load_images_from_paths(files[i:i + 256])
        spec = np.abs(np.fft.fft2(img, axes=(1, 2)))
        out[i:i + len(img)] = (spec * H).mean(axis=(1, 2, 3))
    return out


def main():
    ranks = [int(a) for a in sys.argv[1:]] or [1, 2, 4, 8]
    files = _discover_files(str(ROOT / "data" / "EuroSAT"))
    print(f"{len(files)} images; computing NumPy reference...")
    ref = numpy_features(files)
    py = str(ROOT / "venv" / "bin" / "python")
    rows = []
    for p in ranks:
        subprocess.run(["mpirun", "-n", str(p), py, "src/main_mpi.py"], cwd=ROOT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        got = np.load(ROOT / "results" / "features.npy")
        if got.shape != ref.shape:
            raise RuntimeError(f"P={p}: {got.shape} features, expected {ref.shape}")
        rel = np.abs(got - ref) / np.abs(ref)
        rows.append((p, len(got), float(rel.max()), float(rel.mean())))
        print(f"P={p}: {len(got)} features, relative difference to NumPy: "
              f"max {rel.max():.2e}, mean {rel.mean():.2e}")
    with (ROOT / "results" / "fft_verification.csv").open("w") as f:
        f.write("ranks,images,max_relative_difference,mean_relative_difference\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.3e},{r[3]:.3e}\n")


if __name__ == "__main__":
    main()
