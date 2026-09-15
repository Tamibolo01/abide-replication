"""
vlm_slices.py -- pipeline B, step 1: 4D scans -> small 2D pictures.

Each scan is a 3D movie. Take 16 frames, cut each 7 ways from the top, front
and side, keep the 76x76 pictures as uint8 in data/slices/<subject>.npz
(~0.6 MB per subject). Download the 100 MB scan, slice it, delete it; there
is no disk for 871 of them.
Remember these volumes are residuals, so a single frame is not anatomy but a
map of where the signal is above/below its mean at that instant.
extract_slices() does one subject; build_slices() loops with download+delete.
`python src/vlm_slices.py --sites PITT` (~15 min); no --sites = everyone (hours).

Technical notes
---------------
Build a bank of 2D slices, each taken at a single time point of the fMRI.

This is the image side of the VLM (vlm_model.py): every training image is one
slice of one volume of a subject's resting-state recording, so the model
never sees a time series, only a single instant.

For each subject the ABIDE PCP ``func_preproc`` derivative (C-PAC output on
the MNI 3 mm grid, 61 x 73 x 61 voxels x T time points, about 100 MB
compressed) is reduced to ``n_timepoints`` x 3 planes x ``n_slices`` images
of CANVAS x CANVAS pixels, stored as uint8 in ``data/slices/<FILE_ID>.npz``
(about 1.5 MB with the defaults). Volumes are downloaded one subject at a
time and, unless ``--keep-volumes`` is given, deleted again once sliced:
all 871 quality-checked volumes would take about 87 GB. Volumes that were
already on disk before the run are never deleted.

What a single time point looks like. C-PAC's func_preproc is the residual
after nuisance regression, so every voxel's time series has zero mean. A
single volume is therefore not an anatomical picture but a map of where
the BOLD signal is above or below its temporal mean at that instant,
inside the brain mask (voxels that are constant over time are outside the
brain). Each volume is scaled by its own in-mask standard deviation and
clipped to +/- CLIP_SD standard deviations before quantisation, so slices
from different subjects and sites share one intensity scale; the
background (zero) maps to mid-grey.

Slice positions are chosen per subject from the brain mask's bounding box
along each axis: ``n_slices`` evenly spaced cuts through the central
(1 - 2 * MARGIN) fraction of the box, so slices land on comparable
anatomy across subjects. Time points are evenly spaced over the scan.
"""

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np

import download

SLICES_DIR = download.DATA_DIR / "slices"
PLANES = ("axial", "coronal", "sagittal")
PLANE_AXIS = {"axial": 2, "coronal": 1, "sagittal": 0}
CANVAS = 76  # pixels; every slice of the 61 x 73 x 61 grid fits, centred
CLIP_SD = 3.0
MARGIN = 0.2
BACKGROUND = 128  # uint8 value of zero signal


def brain_mask(volumes):
    """Voxels that vary over time; the rest is outside the brain (all zero)."""
    return volumes.std(axis=-1) > 0


def pick_timepoints(n_total, n_timepoints):
    """Evenly spaced time indices over the scan (all of them if n_timepoints >= n_total)."""
    if n_timepoints >= n_total:
        return np.arange(n_total)
    return np.unique(np.linspace(0, n_total - 1, n_timepoints).round().astype(int))


def pick_slices(mask, axis, n_slices, margin=MARGIN):
    """Evenly spaced slice indices through the central part of the brain.

    Returns
    -------
    indices : numpy.ndarray of int
    positions : numpy.ndarray of float
        Relative position of each slice in the brain's bounding box along
        ``axis``, in [0, 1] and in index order (used by vlm_reports.position_words).
    """
    extent = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    lo, hi = extent.min(), extent.max()
    positions = np.linspace(margin, 1 - margin, n_slices)
    indices = np.round(lo + positions * (hi - lo)).astype(int)
    return indices, positions


def to_uint8(volume, mask, clip_sd=CLIP_SD):
    """Scale one volume by its in-mask std, clip to +/- clip_sd, quantise."""
    scale = volume[mask].std()
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    scaled = np.clip(volume / scale, -clip_sd, clip_sd) / clip_sd  # [-1, 1]
    return np.round((scaled + 1) * 127.5).astype(np.uint8)


def take_slice(volume, plane, index):
    """A 2D slice with 'up' on top, padded to a CANVAS x CANVAS square."""
    axis = PLANE_AXIS[plane]
    image = np.rot90(np.take(volume, index, axis=axis))
    canvas = np.full((CANVAS, CANVAS), BACKGROUND, dtype=volume.dtype)
    top = (CANVAS - image.shape[0]) // 2
    left = (CANVAS - image.shape[1]) // 2
    canvas[top : top + image.shape[0], left : left + image.shape[1]] = image
    return canvas


def extract_slices(volumes, n_timepoints=16, n_slices=7):
    """Slice bank of one subject from its 4D array.

    Parameters
    ----------
    volumes : numpy.ndarray, shape (x, y, z, t)
    n_timepoints, n_slices : int

    Returns
    -------
    dict
        For each plane P in PLANES: ``P`` (uint8, shape (n_timepoints,
        n_slices, CANVAS, CANVAS)), ``P_index`` (slice indices) and
        ``P_position`` (relative positions); plus ``timepoints`` and
        ``n_timepoints_total``.
    """
    mask = brain_mask(volumes)
    if mask.sum() == 0:
        raise ValueError("empty brain mask: the volume is constant over time")
    timepoints = pick_timepoints(volumes.shape[-1], n_timepoints)
    bank = {"timepoints": timepoints, "n_timepoints_total": np.int64(volumes.shape[-1])}
    cuts = {plane: pick_slices(mask, PLANE_AXIS[plane], n_slices) for plane in PLANES}
    stacks = {plane: [] for plane in PLANES}
    for t in timepoints:
        volume = to_uint8(volumes[..., t], mask)
        for plane in PLANES:
            stacks[plane].append([take_slice(volume, plane, i) for i in cuts[plane][0]])
    for plane in PLANES:
        bank[plane] = np.asarray(stacks[plane], dtype=np.uint8)
        bank[f"{plane}_index"] = cuts[plane][0]
        bank[f"{plane}_position"] = cuts[plane][1]
    return bank


def slices_path(file_id, slices_dir=SLICES_DIR):
    return Path(slices_dir) / f"{file_id}.npz"


def volume_path(file_id, pipeline="cpac", strategy="nofilt_noglobal"):
    """Where nilearn caches a subject's func_preproc volume."""
    return download.DATA_DIR / "ABIDE_pcp" / pipeline / strategy / f"{file_id}_func_preproc.nii.gz"


def build_slices(phenotypic, slices_dir=SLICES_DIR, n_timepoints=16, n_slices=7, keep_volumes=False, verbose=1):
    """Download, slice and (optionally) discard the volume of every subject.

    Subjects whose .npz already exists are skipped, so the command can be
    re-run to resume.

    Parameters
    ----------
    phenotypic : pandas.DataFrame
        Rows of the subjects to process (needs FILE_ID).
    slices_dir : path
    n_timepoints, n_slices : int
    keep_volumes : bool
        Keep the downloaded 4D volumes on disk (about 100 MB each).
    verbose : int

    Returns
    -------
    list of Path
        The .npz file of every subject in ``phenotypic``.
    """
    slices_dir = Path(slices_dir)
    slices_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, file_id in enumerate(phenotypic["FILE_ID"], start=1):
        out = slices_path(file_id, slices_dir)
        paths.append(out)
        if out.exists():
            continue
        start = time.time()
        volume_file = volume_path(file_id)
        downloaded = not volume_file.exists()
        dataset = download.fetch_abide(derivatives=("func_preproc",), FILE_ID=file_id, verbose=0)
        volume_file = Path(dataset.func_preproc[0])
        volumes = np.asarray(nib.load(volume_file).dataobj, dtype=np.float32)
        bank = extract_slices(volumes, n_timepoints=n_timepoints, n_slices=n_slices)
        np.savez_compressed(out, **bank)
        if downloaded and not keep_volumes:
            volume_file.unlink()
        if verbose:
            print(f"[{n}/{len(phenotypic)}] {file_id}: {volumes.shape} -> {out.name} "
                  f"({'downloaded' if downloaded else 'cached'}, {time.time() - start:.0f} s)")
    return paths


def load_slices(file_ids, slices_dir=SLICES_DIR):
    """Load the banks of the given subjects into memory.

    Returns
    -------
    dict
        FILE_ID -> dict of arrays as produced by extract_slices. Subjects
        without a bank file are omitted, so check the length.
    """
    banks = {}
    for file_id in file_ids:
        path = slices_path(file_id, slices_dir)
        if path.exists():
            with np.load(path) as data:
                banks[file_id] = {key: data[key] for key in data.files}
    return banks


def main():
    parser = argparse.ArgumentParser(description="Build the single-time-point slice bank from ABIDE func_preproc volumes.")
    parser.add_argument("--sites", nargs="+", default=None, metavar="SITE_ID",
                        help="Restrict to these sites, e.g. --sites PITT NYU (default: all sites).")
    parser.add_argument("--n-subjects", type=int, default=None,
                        help="Process only the first N matching subjects (in ID order).")
    parser.add_argument("--n-timepoints", type=int, default=16, help="Time points per subject (default: 16).")
    parser.add_argument("--n-slices", type=int, default=7, help="Slices per plane (default: 7).")
    parser.add_argument("--keep-volumes", action="store_true",
                        help="Keep downloaded 4D volumes (100 MB each) instead of deleting them after slicing.")
    parser.add_argument("--slices-dir", type=Path, default=SLICES_DIR)
    args = parser.parse_args()

    # The phenotypic table of the quality-checked sample, via the cached ROI
    # derivative (no volume download here).
    filters = {"SITE_ID": args.sites} if args.sites else {}
    _, phenotypic = download.fetch_roi_timeseries("ho", n_subjects=args.n_subjects, verbose=0, **filters)
    y, sites = download.phenotypic_targets(phenotypic)
    print(f"{len(phenotypic)} subjects from {len(np.unique(sites))} sites; {int(y.sum())} ASD / {int((y == 0).sum())} TC")
    print(f"Slices -> {args.slices_dir} ({args.n_timepoints} time points x {len(PLANES)} planes x {args.n_slices} slices per subject)")
    build_slices(phenotypic, args.slices_dir, args.n_timepoints, args.n_slices, args.keep_volumes)


if __name__ == "__main__":
    main()
