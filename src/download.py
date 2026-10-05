"""
download.py -- getting the data. Used by both pipelines.

Only file that touches the internet. nilearn pulls the ABIDE files into data/
the first time and reads from disk after that, so re-running is cheap.
Two things every experiment needs come out of here: each subject's diagnosis
(DX_GROUP 1 = autism, 2 = control in the raw table; phenotypic_targets turns
that into 1/0) and the site that scanned them.
fetch_roi_timeseries() -> the small per-region signal tables (pipeline A).
fetch_abide(derivatives=("func_preproc",)) -> the full 4D scans, 100 MB each
(pipeline B; vlm_slices.py handles those one subject at a time).
`python src/download.py` downloads the region signals for all 871 subjects.

Technical notes
---------------
Fetch ABIDE (Autism Brain Imaging Data Exchange) resting-state data via nilearn.

Replication target: Abraham et al. 2017, "Deriving reproducible biomarkers
from multi-site resting-state data: The ABIDE autism dataset".

Responsibilities:
- Download the ABIDE Preprocessed Connectomes Project (PCP) release using
  nilearn.datasets.fetch_abide_pcp.
- Store downloaded files under data/ (gitignored).
- Return phenotypic data (site, diagnosis, age, ...) alongside the
  derivatives needed for connectivity estimation.

Two kinds of derivative are useful here:

1. ``rois_<atlas>`` -- ROI time series that the PCP already extracted with a
   reference atlas (Harvard-Oxford, Craddock 200, ...). These are ~200 KB per
   subject, so the full quality-checked sample (871 subjects, the same N as
   the paper) is ~200 MB. This is what the replication in connectivity_experiment.py uses:
   it corresponds to the paper's pipeline with a *reference* atlas, i.e. steps
   1-2 (region definition, time-series extraction) done once by the PCP.
2. ``func_preproc`` -- the preprocessed 4D volumes (~100 MB per subject).
   Needed only if you want to redo steps 1-2 yourself, e.g. with a different
   or data-driven atlas (see connectivity.extract_timeseries_from_func).

Defaults match the paper:
- pipeline="cpac": Abraham et al. used data preprocessed with the
  Configurable Pipeline for the Analysis of Connectomes (C-PAC).
- quality_checked=True: keep only subjects that passed visual quality
  assessment. This yields 871 of the original 1112 subjects, as in the paper.
"""

from pathlib import Path

from nilearn import datasets

# Project-local data directory (gitignored) rather than nilearn's default
# ~/nilearn_data, so all downloaded files stay inside this project.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Precomputed ROI time-series derivatives offered by the PCP, keyed by the
# short atlas name used throughout this project. The first two are reference
# atlases that the paper compared against its data-driven atlases.
ROI_DERIVATIVES = {
    "ho": "rois_ho",  # Harvard-Oxford, 111 regions (structural atlas)
    "cc200": "rois_cc200",  # Craddock 200 (spectral clustering)
    "cc400": "rois_cc400",  # Craddock 400
    "aal": "rois_aal",  # Automated Anatomical Labeling
    "ez": "rois_ez",  # Eickhoff-Zilles
    "tt": "rois_tt",  # Talairach-Tournoux
    "dosenbach160": "rois_dosenbach160",  # Dosenbach 160 spheres
}


def fetch_abide(
    n_subjects=None,
    pipeline="cpac",
    derivatives=("rois_ho",),
    band_pass_filtering=False,
    global_signal_regression=False,
    quality_checked=True,
    data_dir=DATA_DIR,
    verbose=1,
    **kwargs,
):
    """Download a subset of the ABIDE preprocessed (PCP) dataset.

    Thin, documented wrapper around nilearn.datasets.fetch_abide_pcp that
    pins defaults to match Abraham et al. 2017 and caches into this
    project's data/ directory instead of the user's home directory.
    Already-downloaded files are reused, not re-downloaded.

    Parameters
    ----------
    n_subjects : int or None, default=None
        Number of subjects to load, taken in subject-ID order (so small
        values land on a single site). None loads every subject matching the
        filters: 871 with quality_checked=True.
    pipeline : {"cpac", "ccs", "dparsf", "niak"}, default="cpac"
        Preprocessing pipeline used to generate the derivatives. The paper
        used C-PAC.
    derivatives : sequence of str, default=("rois_ho",)
        Which derivative(s) to download. See the module docstring and
        ROI_DERIVATIVES for the options.
    band_pass_filtering : bool, default=False
        Use the 0.01-0.1 Hz band-pass filtered variant of the derivatives.
    global_signal_regression : bool, default=False
        Use the variant with global signal regression applied.
    quality_checked : bool, default=True
        Restrict to subjects that passed visual quality assessment, as in
        the paper.
    data_dir : str or pathlib.Path, default=<project_root>/data
        Where downloaded files are cached.
    verbose : int, default=1
        nilearn verbosity (0 silences per-file progress messages).
    **kwargs
        Extra phenotypic filters forwarded to fetch_abide_pcp, e.g.
        SITE_ID="NYU" or DX_GROUP=1 (1=autism, 2=control).

    Returns
    -------
    sklearn.utils.Bunch
        - phenotypic: pandas.DataFrame, one row per subject, with SITE_ID,
          DX_GROUP (1=autism, 2=control), AGE_AT_SCAN, SEX, FILE_ID, ...
        - one attribute per requested derivative, aligned with phenotypic:
          for "func_preproc" a list of NIfTI file paths; for "rois_*" a list
          of numpy arrays of shape (n_timepoints, n_rois), already loaded.
        - description: text description of the dataset
    """
    return datasets.fetch_abide_pcp(
        data_dir=str(data_dir),
        n_subjects=n_subjects,
        pipeline=pipeline,
        derivatives=list(derivatives),
        band_pass_filtering=band_pass_filtering,
        global_signal_regression=global_signal_regression,
        quality_checked=quality_checked,
        verbose=verbose,
        **kwargs,
    )


def fetch_roi_timeseries(atlas="ho", n_subjects=None, **kwargs):
    """Fetch precomputed ROI time series for one reference atlas.

    Parameters
    ----------
    atlas : str, default="ho"
        Key of ROI_DERIVATIVES.
    n_subjects, **kwargs
        Forwarded to fetch_abide.

    Returns
    -------
    timeseries : list of numpy.ndarray
        One (n_timepoints, n_rois) array per subject. Scan length varies by
        site, so the arrays have different numbers of rows.
    phenotypic : pandas.DataFrame
        Phenotypic table aligned with ``timeseries``.
    """
    if atlas not in ROI_DERIVATIVES:
        raise ValueError(f"atlas must be one of {sorted(ROI_DERIVATIVES)}, got {atlas!r}")
    derivative = ROI_DERIVATIVES[atlas]
    dataset = fetch_abide(n_subjects=n_subjects, derivatives=(derivative,), **kwargs)
    return list(dataset[derivative]), dataset.phenotypic


def phenotypic_targets(phenotypic):
    """Extract the classification targets used throughout the project.

    Parameters
    ----------
    phenotypic : pandas.DataFrame
        As returned by fetch_abide / fetch_roi_timeseries.

    Returns
    -------
    y : numpy.ndarray of int
        1 for autism spectrum disorder (ABIDE DX_GROUP == 1), 0 for typical
        control (DX_GROUP == 2). ASD is the "positive" class, so sensitivity
        in connectivity_experiment.py means the fraction of ASD subjects detected.
    sites : numpy.ndarray of str
        Acquisition site of each subject (SITE_ID), used for site-stratified
        and leave-one-site-out cross-validation.
    """
    y = (phenotypic["DX_GROUP"].to_numpy() == 1).astype(int)
    sites = phenotypic["SITE_ID"].to_numpy().astype(str)
    return y, sites


def main():
    """CLI entry point: download derivatives and print a per-site summary."""
    import argparse

    parser = argparse.ArgumentParser(description="Download ABIDE preprocessed data via nilearn into data/.")
    parser.add_argument(
        "--derivative",
        default="rois_ho",
        choices=sorted(ROI_DERIVATIVES.values()) + ["func_preproc"],
        help="What to download (default: rois_ho, Harvard-Oxford ROI time series, "
        "~200 MB for all subjects; func_preproc is ~100 MB PER subject).",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=None,
        help="Number of subjects to fetch (default: all 871 quality-checked subjects).",
    )
    parser.add_argument(
        "--pipeline",
        default="cpac",
        choices=["cpac", "ccs", "dparsf", "niak"],
        help="Preprocessing pipeline (default: cpac, as used in the paper).",
    )
    args = parser.parse_args()

    dataset = fetch_abide(
        n_subjects=args.n_subjects,
        pipeline=args.pipeline,
        derivatives=(args.derivative,),
    )
    phenotypic = dataset.phenotypic
    print(f"\nLoaded {len(phenotypic)} subjects ({args.derivative}) from {DATA_DIR}")
    per_site = (
        phenotypic.groupby("SITE_ID")["DX_GROUP"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={1: "ASD", 2: "TC"})
    )
    print(per_site.to_string())


if __name__ == "__main__":
    main()
