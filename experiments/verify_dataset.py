"""
Audit of what the 54,597 dataset files are (report Section 1.2).

Counts the files per format and class, checks that every image is readable and
that no file is duplicated, tests whether a JPEG and the GeoTIFF of the same name
show the same patch, and reports the map zones of the GeoTIFFs.

Usage:
    python experiments/verify_dataset.py

Prints the audit; the 2026-11-03 output is in results/adhoc_measurements.md.
"""
import collections
import csv
import glob
import hashlib
import os
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "data" / "EuroSAT" / "EuroSAT"
BANDS = ROOT / "allBands"
SAMPLE = 500


def jpeg_gray(path):
    return np.array(Image.open(path).convert("RGB"), float).mean(2).ravel()


def tif_gray(path):
    return tifffile.imread(path)[:, :, [3, 2, 1]].astype(float).mean(2).ravel()


def utm_zone(path):
    with tifffile.TiffFile(path) as tif:
        epsg = tif.geotiff_metadata["ProjectedCSTypeGeoKey"]
    return f"{int(epsg) % 100}{'N' if int(epsg) // 100 == 326 else 'S'}"


def is_extra(path):
    return "Jakarta" in path or "Bombay" in path


def main():
    jpgs = sorted(glob.glob(str(ROOT / "*" / "*.jpg")))
    tifs = sorted(glob.glob(str(BANDS / "*" / "*.tif")))
    extras = [p for p in tifs if is_extra(os.path.basename(p))]
    print(f"Files: {len(jpgs)} JPEG + {len(tifs)} GeoTIFF = {len(jpgs) + len(tifs)}")

    per_class = collections.defaultdict(lambda: [0, 0])
    for p in jpgs:
        per_class[Path(p).parent.name][0] += 1
    for p in tifs:
        per_class[Path(p).parent.name][1] += 1
    for name, (j, t) in sorted(per_class.items()):
        print(f"  {name:22s} {j:5d} JPEG {t:5d} GeoTIFF")
    prefixes = collections.Counter(os.path.basename(p).split("_")[0] for p in extras)
    print(f"Extra GeoTIFFs: {len(extras)} {dict(prefixes)}, all in "
          f"{sorted({Path(p).parent.name for p in extras})}")

    listed = set()
    for split in ("train", "validation", "test"):
        with open(BANDS / f"{split}.csv") as f:
            listed |= {row["Filename"] for row in csv.DictReader(f)}
    on_disk = {os.path.relpath(p, BANDS) for p in tifs}
    print(f"Dataset split lists: {sum(os.path.relpath(p, BANDS) in listed for p in extras)} of the "
          f"extras listed; files not listed: {sorted(on_disk - listed)}")

    shapes, hashes = collections.Counter(), collections.Counter()
    for p in tifs:
        a = tifffile.imread(p)
        shapes[f"GeoTIFF {a.shape} {a.dtype}"] += 1
        hashes[hashlib.sha1(a.tobytes()).hexdigest()] += 1
    for p in jpgs:
        with Image.open(p) as im:
            im.load()
            shapes[f"JPEG {im.size} {im.mode}"] += 1
            hashes[hashlib.sha1(im.tobytes()).hexdigest()] += 1
    print(f"Readable shapes: {dict(shapes)}")
    print(f"Files with identical pixels: {sum(n > 1 for n in hashes.values())} groups")

    def partner(p):
        return str(BANDS / Path(p).parent.name / (Path(p).stem + ".tif"))

    r = np.array([np.corrcoef(jpeg_gray(p), tif_gray(partner(p)))[0, 1] for p in jpgs])
    print(f"Same-name JPEG/GeoTIFF pairs: {len(r)}, correlation median {np.median(r):.3f}, "
          f"{np.mean(r > 0.8):.1%} above 0.8, minimum {r.min():.3f}")

    rng = np.random.default_rng(0)
    plain = [p for p in tifs if not is_extra(os.path.basename(p))]
    ctrl = [abs(np.corrcoef(jpeg_gray(jpgs[i]), tif_gray(plain[j]))[0, 1])
            for i, j in zip(rng.integers(len(jpgs), size=2000), rng.integers(len(plain), size=2000))]
    print(f"Control, random pairs: median |r| {np.median(ctrl):.3f}, "
          f"{np.mean(np.array(ctrl) > 0.8):.1%} above 0.8")

    # The harder test: is the same-name GeoTIFF the best match in its whole class?
    matrices = {}
    hits = 0
    for i in rng.choice(len(jpgs), SAMPLE, replace=False):
        p = jpgs[i]
        name = Path(p).parent.name
        if name not in matrices:
            files = [f for f in tifs if Path(f).parent.name == name]
            m = np.stack([tif_gray(f) for f in files])
            matrices[name] = (files, (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9))
        files, m = matrices[name]
        g = jpeg_gray(p)
        hits += files[int(np.argmax(m @ ((g - g.mean()) / (g.std() + 1e-9))))] == partner(p)
    print(f"Best match in class is the same-name GeoTIFF: {hits} of {SAMPLE} random JPEGs")

    print(f"UTM zones, benchmark GeoTIFFs: {dict(sorted(collections.Counter(map(utm_zone, plain)).items()))}")
    print(f"UTM zones, extra GeoTIFFs: {dict(collections.Counter(map(utm_zone, extras)))}")


if __name__ == "__main__":
    main()
