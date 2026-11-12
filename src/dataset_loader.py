import os
import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None


def _discover_files(data_dir: str) -> list:
    """
    Recursively walk `data_dir` and return every image file path, in the order
    os.walk visits them.

    On the testbed this order is stable from run to run and gives 14,000 RGB
    JPEGs, then the 27,597 13-band GeoTIFFs (``allBands/``), then 13,000 JPEGs.
    Rank 0 alone calls this and broadcasts the list, so all ranks always
    partition the same list.
    """
    all_files = []
    if os.path.exists(data_dir):
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    all_files.append(os.path.join(root, file))
    return all_files


def get_partitioned_paths(
    data_dir, rank, size, comm=None, max_images=None
):
    """
    Return the list of image file paths assigned to this MPI rank (no loading).

    Data-parallel partitioning (report Eq. 3.1): only rank 0 scans the dataset
    directory and broadcasts the resulting file list, rather than every rank
    scanning independently. This avoids N ranks hitting the filesystem with a
    redundant full directory walk, and guarantees every rank partitions from
    the exact same list.

    Each rank then takes a contiguous slice of that list, not a round-robin
    (cyclic) one: a simple closed-form split with equal image counts. Image
    counts are equal but per-image cost is not: a GeoTIFF takes more than
    twice as long to load from disk as a JPEG, so ranks holding mostly
    GeoTIFFs finish last (report §5.6.2). A cyclic split (all_files[rank::size])
    would give every rank the same JPEG/GeoTIFF mix.
    """
    all_files = []
    if rank == 0:
        all_files = _discover_files(data_dir)
        total_images = len(all_files)
        if total_images == 0:
            print(f"[Warning] No images found in '{data_dir}'. Will use simulated data.")

    # MPI_Bcast: the one and only inter-rank communication needed before compute
    # begins. After this call every rank independently computes its own slice
    # below, with no further coordination required.
    if comm is not None:
        all_files = comm.bcast(all_files, root=0)

    total_images = len(all_files)
    if total_images == 0:
        return []

    if max_images is not None:
        # Weak-scaling mode: fixed images per rank regardless of dataset size,
        # so total work grows with rank count instead of staying constant.
        start_idx = rank * max_images
        end_idx = start_idx + max_images
        if start_idx >= total_images:
            return []
        my_files = all_files[start_idx : min(end_idx, total_images)]
    else:
        # Block distribution (report Eq. 3.1): rank r starts at r*Q + min(r, R),
        # where Q = N // P images per rank and R = N % P left-over images.
        # Strong-scaling mode: fixed total dataset split across ranks.
        # remainder images are distributed one-per-rank to the first
        # `remainder` ranks, so no rank differs from another by more than 1
        # image (balanced split, not just floor division).
        chunk_size = total_images // size
        remainder = total_images % size
        start_idx = rank * chunk_size + min(rank, remainder)
        end_idx = start_idx + chunk_size + (1 if rank < remainder else 0)
        my_files = all_files[start_idx:end_idx]

    if rank == 0:
        print(f"Dataset Found: {total_images} images.")
        if max_images is None:
            cs, rem = total_images // size, total_images % size
            print(
                f"Partitioning: contiguous slices (~{cs + (1 if rem else 0)} max per rank)."
            )
        else:
            print(f"Partitioning: max_images={max_images} per rank.")

    return my_files


def load_images_from_paths(file_paths, target_size=(64, 64), max_pad_to=None):
    """
    Stage 1 (Load and Preprocess, CPU): read each file and resize it to a
    64x64x3 array. This is the slowest stage of the pipeline (report Table 5.7).

    Load and normalize images from explicit paths into a float32 array
    shaped (N, H, W, C), values in [0, 1].

    Handles two source formats because EuroSAT ships both: the standard
    8-bit RGB JPEG release, and the raw 13-band Sentinel-2 GeoTIFF release
    (used for the larger multispectral workload).
    """
    if not file_paths:
        n = max_pad_to if max_pad_to else 1
        return generate_simulated_data(max(n, 1), target_size)

    valid_images = []
    for i, file_path in enumerate(file_paths):
        try:
            arr = None
            ext = os.path.splitext(file_path)[1].lower()

            if ext in {".tif", ".tiff"} and tifffile:
                img_data = tifffile.imread(file_path)
                if img_data.ndim == 3:
                    if img_data.shape[0] == 13:
                        # Sentinel-2 bands ship channel-first; move to HWC to
                        # match the JPEG path's layout.
                        img_data = np.transpose(img_data, (1, 2, 0))
                    if img_data.shape[2] >= 4:
                        # Bands 4/3/2 (indices 3,2,1) are the true-colour
                        # Red/Green/Blue bands in the Sentinel-2 spec — select
                        # them so the 13-band scan reduces to an RGB image
                        # comparable to the JPEG release, not a false-colour one.
                        img_data = img_data[:, :, [3, 2, 1]]
                if img_data.dtype == np.uint16:
                    # Sentinel-2 reflectance is scaled by 10000 and typically
                    # capped well below that for valid surface returns; 2000 is
                    # the empirical normalisation constant that maps the usable
                    # range into [0, 1] without clipping most real pixels.
                    arr = img_data.astype(np.float32) / 2000.0
                    arr = np.clip(arr, 0.0, 1.0)
                else:
                    arr = img_data.astype(np.float32) / 255.0
                if arr.shape[0] != target_size[1] or arr.shape[1] != target_size[0]:
                    # Resize via PIL for consistency with the JPEG path below,
                    # round-tripping through uint8 since PIL expects that range.
                    temp_img = Image.fromarray((arr * 255).astype(np.uint8))
                    temp_img = temp_img.resize(target_size)
                    arr = np.array(temp_img, dtype=np.float32) / 255.0
            else:
                with Image.open(file_path) as img:
                    img = img.convert("RGB")
                    if img.size != target_size:
                        img = img.resize(target_size)
                    arr = np.array(img, dtype=np.float32) / 255.0

            if arr is not None:
                valid_images.append(arr)
        except Exception as e:
            # Corrupt/unreadable files are skipped rather than aborting the
            # whole rank's load; only the first few are logged to avoid
            # flooding stdout across many MPI ranks on a bad batch.
            if i < 5:
                print(f"Error reading {file_path}: {e}")
            continue

    if len(valid_images) == 0:
        n = len(file_paths) if file_paths else (max_pad_to or 1)
        return generate_simulated_data(max(n, max_pad_to or 0), target_size)

    arr = np.array(valid_images)
    if max_pad_to is not None and len(arr) < max_pad_to:
        # Pad up to the requested batch size with simulated data so every
        # rank reports the same batch shape even if some files failed to load
        # (keeps weak-scaling comparisons apples-to-apples across ranks).
        pad = generate_simulated_data(max_pad_to - len(arr), target_size)
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def load_eurosat_dataset(data_dir, rank, size, comm=None, target_size=(64, 64), max_images=None):
    """
    Top-level entry point: partition the dataset for this rank, then load
    and normalize its share of images.

    Kept separate from get_partitioned_paths/load_images_from_paths because
    the task-parallel pipeline (batched_pipeline.py) needs the partitioning
    and per-batch loading steps independently, while the simple sequential
    path just wants "give me this rank's whole dataset" in one call.
    """
    my_files = get_partitioned_paths(
        data_dir, rank, size, comm=comm, max_images=max_images
    )
    if not my_files:
        # No real files on disk for this rank (e.g. running without the
        # dataset present): fall back to simulated data so the pipeline is
        # still exercisable for development/testing without EuroSAT installed.
        n = max_images if max_images else max(128 // max(size, 1), 1)
        return generate_simulated_data(n, target_size)

    arr = load_images_from_paths(my_files, target_size, max_pad_to=max_images)
    if len(arr) == 0:
        n = len(my_files) if not max_images else max_images
        return generate_simulated_data(max(n, 1), target_size)
    return arr


def generate_simulated_data(count, size):
    """Random placeholder batch, used only when real dataset files are unavailable."""
    return np.random.rand(count, size[1], size[0], 3).astype(np.float32)
