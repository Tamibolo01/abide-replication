"""
connectivity.py -- pipeline A, paper steps 1-3: region signals -> features.

Input is one table per subject (rows = time points, columns = ~100 regions).
Output is one row of 5,050 numbers per subject: for every pair of regions,
how much the two signals move together. Three ways to measure that
(correlation / partial correlation / tangent) because the paper compares them.
Not run on its own; connectivity_experiment.py calls connectivity_features()
once per fold. That function is the one to read: it fits the measure on the
training subjects only, which is what keeps the tangent version leak-free.
standardize_timeseries() is the bug fix from the audit in the README.

Technical notes
---------------
Build functional connectivity matrices from ROI time series.

Replication target: Abraham et al. 2017, "Deriving reproducible biomarkers
from multi-site resting-state data: The ABIDE autism dataset".

This module covers the paper's pipeline steps 1-3:

1. Region definition. The replication uses reference atlases whose ROI
   time series the PCP already extracted (download.fetch_roi_timeseries).
   extract_timeseries_from_func shows how to redo steps 1-2 yourself from
   the preprocessed 4D volumes with an atlas from nilearn.
2. Time-series extraction (as above).
3. Connectivity estimation. Each subject's ROI time series become one
   connectivity matrix. As in the paper, the covariance is estimated with
   the Ledoit-Wolf shrinkage estimator (nilearn's default) and three
   connectivity measures are compared:
     - "correlation": Pearson correlation between region signals,
     - "partial correlation": from the inverse covariance (precision)
       matrix, i.e. direct coupling with other regions factored out,
     - "tangent": tangent-space embedding, which maps every subject's
       covariance matrix into the tangent space at the *group* mean
       covariance. This depends on the group, so it must be fitted on
       training subjects only (see connectivity_features).
   One weight per pair of regions is kept, giving n_rois*(n_rois-1)/2
   features per subject for the classifier in connectivity_experiment.py.

Before any covariance is estimated, every ROI signal is detrended and
z-scored (standardize_timeseries), as in the paper. This is done here
explicitly because nilearn's ConnectivityMeasure only standardizes for
kind="correlation"; for "tangent" and "partial correlation" it uses the
raw signals, whose amplitude varies across regions and scanners.
"""

import numpy as np
from nilearn import datasets, signal
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

KINDS = ("correlation", "partial correlation", "tangent")


def drop_constant_rois(timeseries):
    """Remove ROIs that are constant (or non-finite) in *any* subject.

    A region can have a flat signal for a subject when it falls outside
    that site's field of view. Its variance is zero, so its correlation with
    anything is 0/0 = NaN, which breaks every downstream step. Dropping the
    region from every subject, not just the affected one, keeps the feature
    space identical across subjects and sites.

    Parameters
    ----------
    timeseries : list of numpy.ndarray
        One (n_timepoints, n_rois) array per subject.

    Returns
    -------
    cleaned : list of numpy.ndarray
        Same arrays restricted to the kept ROIs.
    keep : numpy.ndarray of bool
        Mask over the original ROIs (True = kept).
    """
    n_rois = timeseries[0].shape[1]
    keep = np.ones(n_rois, dtype=bool)
    for ts in timeseries:
        keep &= np.isfinite(ts).all(axis=0)
        keep &= ts.std(axis=0) > 0
    return [ts[:, keep] for ts in timeseries], keep


def standardize_timeseries(timeseries):
    """Detrend and z-score every ROI signal of every subject.

    Removes a linear trend, then scales each region's signal to zero mean
    and unit variance, so that covariance estimates reflect co-fluctuation
    rather than signal amplitude. Region amplitude in raw BOLD data depends
    on tissue, coil sensitivity and scanner, i.e. mostly on the site.
    """
    return [signal.clean(ts, detrend=True, standardize="zscore_sample") for ts in timeseries]


def make_connectivity_measure(kind="tangent", vectorize=True):
    """Configure nilearn's ConnectivityMeasure as in the paper.

    Parameters
    ----------
    kind : {"correlation", "partial correlation", "tangent"}
    vectorize : bool, default=True
        If True, transform() returns one feature vector per subject holding
        the strictly-upper-triangular part of the symmetric matrix (one
        value per pair of regions). If False, it returns full square
        matrices, which is handy for plotting.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    # cov_estimator defaults to LedoitWolf, matching the paper.
    return ConnectivityMeasure(
        kind=kind,
        vectorize=vectorize,
        discard_diagonal=True,
        standardize="zscore_sample",
    )


def connectivity_features(kind, train_timeseries, test_timeseries=None):
    """Turn ROI time series into classifier features, without leakage.

    The measure is fitted on the training subjects only and then applied to
    both sets. For "tangent" this matters: the embedding is taken around the
    training group's mean covariance, so fitting on all subjects would let
    test subjects shape the features they are later scored on. For the
    other kinds fit() learns nothing from the group, but the same code path
    keeps every cross-validation loop honest.

    Parameters
    ----------
    kind : str
        One of KINDS.
    train_timeseries, test_timeseries : list of numpy.ndarray
        (n_timepoints, n_rois) arrays; test_timeseries may be None.

    Returns
    -------
    X_train : numpy.ndarray, shape (n_train, n_rois*(n_rois-1)/2)
    X_test : numpy.ndarray or None, shape (n_test, n_features)
    measure : fitted ConnectivityMeasure
    """
    measure = make_connectivity_measure(kind, vectorize=True)
    X_train = measure.fit_transform(standardize_timeseries(train_timeseries))
    X_test = None
    if test_timeseries is not None:
        X_test = measure.transform(standardize_timeseries(test_timeseries))
    return X_train, X_test, measure


def connectivity_matrices(kind, timeseries):
    """Full square connectivity matrices for all subjects (for inspection).

    Returns an array of shape (n_subjects, n_rois, n_rois). Not for use
    inside cross-validation; see connectivity_features for that.
    """
    measure = make_connectivity_measure(kind, vectorize=False)
    return measure.fit_transform(standardize_timeseries(timeseries))


def extract_timeseries_from_func(
    func_files, atlas_name="cort-maxprob-thr25-2mm", data_dir=None, verbose=0
):
    """Paper steps 1-2 from scratch: ROI time series from 4D volumes.

    Uses nilearn's Harvard-Oxford atlas and a NiftiLabelsMasker to average
    the preprocessed BOLD signal within each labelled region. Signals are
    detrended and z-scored per region. This is a simplified version of the
    paper's step 2, which additionally regressed out CompCor components and
    motion parameters at the region level.

    Parameters
    ----------
    func_files : list of str
        Paths to preprocessed 4D NIfTI files (download.fetch_abide with
        derivatives=("func_preproc",)).
    atlas_name : str
        Harvard-Oxford atlas variant, see nilearn.datasets.fetch_atlas_harvard_oxford.
    data_dir : str or Path or None
        Where nilearn caches the atlas (defaults to ~/nilearn_data).
    verbose : int

    Returns
    -------
    list of numpy.ndarray
        One (n_timepoints, n_rois) array per subject.
    """
    atlas = datasets.fetch_atlas_harvard_oxford(atlas_name, data_dir=data_dir)
    masker = NiftiLabelsMasker(
        labels_img=atlas.maps,
        standardize="zscore_sample",
        detrend=True,
        verbose=verbose,
    )
    return [masker.fit_transform(f) for f in func_files]
