"""
Fetch ABIDE (Autism Brain Imaging Data Exchange) resting-state data via nilearn.

Replication target: Abraham et al. 2017, "Deriving reproducible biomarkers
from multi-site resting-state data: The ABIDE autism dataset".

Responsibilities:
- Download the ABIDE preprocessed (PCP) dataset subset needed for this
  replication using nilearn.datasets.fetch_abide_pcp.
- Store downloaded files under data/ (gitignored).
- Return/organise phenotypic data (site, diagnosis, age, etc.) alongside
  the functional derivatives needed for connectivity estimation.

Notes on defaults, matching the paper's methodology:
- pipeline="cpac": Abraham et al. used data preprocessed with the
  Configurable Pipeline for the Analysis of Connectomes (C-PAC), which is
  also nilearn's default.
- derivatives=("func_preproc",): we fetch the preprocessed 4D functional
  volumes rather than any of nilearn's precomputed ROI time series, because
  connectivity.py performs its own region definition + time-series
  extraction (pipeline steps 1-2 in the paper), mirroring the paper rather
  than skipping straight to a fixed atlas's precomputed signals.
- quality_checked=True: restrict to subjects that passed the paper's visual
  quality assessment (871 of the original 1112 subjects passed).
"""

from pathlib import Path

from nilearn import datasets

# Project-local data directory (gitignored) rather than nilearn's default
# ~/nilearn_data, so all downloaded files stay inside this project.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_abide(
    n_subjects=None,
    pipeline="cpac",
    derivatives=("func_preproc",),
    band_pass_filtering=False,
    global_signal_regression=False,
    quality_checked=True,
    data_dir=DATA_DIR,
    **kwargs,
):
    """Download a subset of the ABIDE preprocessed (PCP) dataset.

    Thin, documented wrapper around nilearn.datasets.fetch_abide_pcp that
    pins defaults to match Abraham et al. 2017 and caches into this
    project's data/ directory instead of the user's home directory.

    Parameters
    ----------
    n_subjects : int or None, default=None
        Number of subjects to download. None fetches every subject matching
        the filters (871 with quality_checked=True) -- several GB. Pass a
        small number (e.g. 20) for local development before scaling up.
    pipeline : {"cpac", "ccs", "dparsf", "niak"}, default="cpac"
        Preprocessing pipeline used to generate the derivatives. The paper
        used C-PAC.
    derivatives : sequence of str, default=("func_preproc",)
        Which derivative(s) to download. "func_preproc" is the preprocessed
        4D functional volume, the input connectivity.py needs to perform its
        own region definition and time-series extraction.
    band_pass_filtering : bool, default=False
        Whether to band-pass filter the signal (0.01-0.1 Hz) before serving
        it. Off by default, matching nilearn's default.
    global_signal_regression : bool, default=False
        Whether global signal regression was applied. Off by default,
        matching nilearn's default.
    quality_checked : bool, default=True
        Restrict to subjects that passed visual quality assessment, as in
        the paper.
    data_dir : str or pathlib.Path, default=<project_root>/data
        Where downloaded files are cached.
    **kwargs
        Extra phenotypic filters forwarded to fetch_abide_pcp, e.g.
        SITE_ID="NYU" or DX_GROUP=1 (1=autism, 2=control).

    Returns
    -------
    sklearn.utils.Bunch
        Bunch with, among other fields:
        - func_preproc: list of paths to preprocessed 4D NIfTI files
        - phenotypic: numpy structured array with SITE_ID, DX_GROUP,
          AGE_AT_SCAN, SEX, etc.
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
        **kwargs,
    )


def main():
    """CLI entry point: download a small subset for local development."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download a subset of ABIDE preprocessed data via nilearn."
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=20,
        help="Number of subjects to fetch (default: 20, for quick local testing).",
    )
    parser.add_argument(
        "--pipeline",
        default="cpac",
        choices=["cpac", "ccs", "dparsf", "niak"],
        help="Preprocessing pipeline to fetch derivatives from (default: cpac, as used in the paper).",
    )
    args = parser.parse_args()

    dataset = fetch_abide(n_subjects=args.n_subjects, pipeline=args.pipeline)

    print(f"Downloaded {len(dataset.func_preproc)} subjects to {DATA_DIR}")
    print(dataset.phenotypic[["SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX"]][:5])


if __name__ == "__main__":
    main()
