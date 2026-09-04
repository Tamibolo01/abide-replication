"""
Cross-validation and scoring for the ASD vs. control classification task.

Replication target: Abraham et al. 2017, "Deriving reproducible biomarkers
from multi-site resting-state data: The ABIDE autism dataset".

This module covers the paper's pipeline step 4 (supervised learning) and
its validation scheme:

- Classifiers, as in the paper: l2-penalised linear SVC, l1-penalised
  (sparse) linear SVC, and ridge. Regularisation strength is chosen by
  nested cross-validation on the training fold only (GridSearchCV for the
  SVCs, RidgeClassifierCV's efficient leave-one-out for ridge), so no test
  subject influences a hyperparameter. A "dummy" classifier that always
  predicts the majority class (typical control) gives the chance level;
  on the full sample that is 53.7%, the figure the paper reports.
- Two cross-validation schemes:
    "intra": 10-fold, stratified by site x diagnosis, so every fold has the
             same mix of sites and conditions (the paper's intra-site CV);
    "inter": leave-one-site-out, i.e. train on all other sites and test on
             a site the classifier never saw (the paper's inter-site CV,
             the realistic clinical scenario).
- Scores per fold: accuracy, sensitivity (ASD recall) and specificity (TC
  recall), the three numbers the paper reports.

Connectivity features are recomputed inside every fold from training
subjects only (see connectivity.connectivity_features), which is what makes
the tangent-space results leak-free.

Run ``python src/experiment.py --help`` for the command-line options.
Results (per-fold CSV, summary CSV, figure) land in results/ (gitignored).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold
from sklearn.svm import LinearSVC

import connectivity
import download

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SCHEMES = ("intra", "inter")
CLASSIFIER_NAMES = ("svc_l2", "svc_l1", "ridge", "dummy")
SVC_C_GRID = np.logspace(-5, 1, 7)  # 1e-5 ... 10; connectivity features are small numbers
RIDGE_ALPHA_GRID = np.logspace(-5, 5, 21)


def make_classifier(name, random_state=0, inner_cv=3):
    """Return a fresh, unfitted classifier by name (see CLASSIFIER_NAMES).

    inner_cv is the number of stratified folds used, inside the training
    set, to pick the SVC penalty C. Ridge picks alpha by leave-one-out.
    """
    inner = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
    if name == "svc_l2":
        svc = LinearSVC(penalty="l2", max_iter=20000, random_state=random_state)
        return GridSearchCV(svc, {"C": SVC_C_GRID}, cv=inner, n_jobs=1)
    if name == "svc_l1":
        # l1 needs the primal formulation (dual=False).
        svc = LinearSVC(penalty="l1", dual=False, max_iter=20000, random_state=random_state)
        return GridSearchCV(svc, {"C": SVC_C_GRID}, cv=inner, n_jobs=1)
    if name == "ridge":
        return RidgeClassifierCV(alphas=RIDGE_ALPHA_GRID)
    if name == "dummy":
        return DummyClassifier(strategy="most_frequent")
    raise ValueError(f"unknown classifier {name!r}; choose from {CLASSIFIER_NAMES}")


def chosen_hyperparameter(clf):
    """The regularisation value a fitted classifier settled on, or NaN."""
    if hasattr(clf, "best_params_"):
        return clf.best_params_["C"]
    if hasattr(clf, "alpha_"):
        return float(clf.alpha_)
    return np.nan


def cv_splits(scheme, y, sites, n_splits=10, random_state=0):
    """Yield (fold_name, train_indices, test_indices) for a CV scheme.

    "intra" stratifies on the combination of site and diagnosis, mirroring
    the paper: "randomly splitting the participants into training and
    testing sets while preserving the ratio of samples for each site and
    condition". "inter" holds out one whole site per fold.
    """
    placeholder = np.zeros(len(y))  # sklearn splitters only look at its length
    if scheme == "intra":
        strata = np.array([f"{site}|{label}" for site, label in zip(sites, y)])
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for i, (train, test) in enumerate(splitter.split(placeholder, strata)):
            yield f"fold{i:02d}", train, test
    elif scheme == "inter":
        if len(np.unique(sites)) < 2:
            raise ValueError("leave-one-site-out needs at least two sites")
        for train, test in LeaveOneGroupOut().split(placeholder, y, groups=sites):
            yield str(sites[test[0]]), train, test
    else:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")


def score(y_true, y_pred):
    """Accuracy, sensitivity (ASD recall) and specificity (TC recall)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


def evaluate_fold(scheme, fold, train, test, timeseries, y, kind, classifiers, random_state=0, inner_cv=3):
    """Compute features for one fold and score every classifier on it.

    Features are computed once per fold and kind and shared by all
    classifiers, because feature extraction (not classification) dominates
    the run time.
    """
    X_train, X_test, _ = connectivity.connectivity_features(
        kind,
        [timeseries[i] for i in train],
        [timeseries[i] for i in test],
    )
    rows = []
    for name in classifiers:
        clf = make_classifier(name, random_state=random_state, inner_cv=inner_cv)
        clf.fit(X_train, y[train])
        metrics = score(y[test], clf.predict(X_test))
        rows.append(
            {
                "cv_scheme": scheme,
                "fold": fold,
                "connectivity": kind,
                "classifier": name,
                "n_train": len(train),
                "n_test": len(test),
                "hyperparameter": chosen_hyperparameter(clf),
                **metrics,
            }
        )
    return rows


def run_experiment(
    timeseries,
    y,
    sites,
    kinds=connectivity.KINDS,
    classifiers=CLASSIFIER_NAMES,
    schemes=SCHEMES,
    n_splits=10,
    inner_cv=3,
    n_jobs=1,
    random_state=0,
    verbose=1,
):
    """Run every (scheme, fold, connectivity kind) job and collect scores.

    Returns
    -------
    pandas.DataFrame
        One row per (scheme, fold, connectivity kind, classifier).
    """
    jobs = []
    for scheme in schemes:
        if scheme == "inter" and len(np.unique(sites)) < 2:
            print("Skipping inter-site CV: only one site in the sample.")
            continue
        for fold, train, test in cv_splits(scheme, y, sites, n_splits, random_state):
            for kind in kinds:
                jobs.append((scheme, fold, train, test, kind))
    if verbose:
        print(f"Running {len(jobs)} feature-extraction jobs x {len(classifiers)} classifiers ...")
    start = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(evaluate_fold)(scheme, fold, train, test, timeseries, y, kind, classifiers, random_state, inner_cv)
        for scheme, fold, train, test, kind in jobs
    )
    if verbose:
        print(f"Done in {time.time() - start:.0f} s")
    return pd.DataFrame([row for rows in results for row in rows])


def summarize(results):
    """Mean and std of each score over folds, per pipeline.

    Also reports accuracy pooled over test subjects ("accuracy_pooled"): with
    leave-one-site-out, sites differ in size, so the plain mean over folds
    weights a 10-subject site as much as a 170-subject one.
    """
    keys = ["cv_scheme", "connectivity", "classifier"]
    stats = results.groupby(keys)[["accuracy", "sensitivity", "specificity"]].agg(["mean", "std"])
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    pooled = results.assign(correct=results["accuracy"] * results["n_test"]).groupby(keys)
    stats["accuracy_pooled"] = pooled["correct"].sum() / pooled["n_test"].sum()
    stats["n_folds"] = results.groupby(keys).size()
    stats["hyperparameter_median"] = results.groupby(keys)["hyperparameter"].median()
    return stats.reset_index()


# Colours: first three categorical slots of the project's validated palette.
KIND_COLORS = {
    "correlation": "#2a78d6",
    "partial correlation": "#eb6834",
    "tangent": "#1baf7a",
}
INK, INK_MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def plot_results(summary, path, title="ASD vs. control classification accuracy"):
    """Dot-and-error-bar chart of mean accuracy (+/- std over folds).

    One panel per CV scheme, classifiers on the x-axis, one colour per
    connectivity kind, dashed line at the dummy classifier's chance level.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schemes = [s for s in SCHEMES if s in set(summary["cv_scheme"])]
    kinds = [k for k in KIND_COLORS if k in set(summary["connectivity"])]
    classifiers = [c for c in CLASSIFIER_NAMES if c != "dummy" and c in set(summary["classifier"])]
    fig, axes = plt.subplots(1, len(schemes), figsize=(4.2 * len(schemes), 4.4), sharey=True, squeeze=False)
    n_folds = summary.groupby("cv_scheme")["n_folds"].max()
    labels = {
        "intra": f"Intra-site ({n_folds.get('intra', 0)}-fold, site-stratified)",
        "inter": f"Inter-site (leave-one-site-out, {n_folds.get('inter', 0)} sites)",
    }
    offsets = np.linspace(-0.22, 0.22, len(kinds)) if len(kinds) > 1 else [0.0]

    for ax, scheme in zip(axes[0], schemes):
        panel = summary[summary["cv_scheme"] == scheme]
        for offset, kind in zip(offsets, kinds):
            rows = panel[panel["connectivity"] == kind].set_index("classifier")
            xs = [i + offset for i, c in enumerate(classifiers) if c in rows.index]
            means = [rows.loc[c, "accuracy_mean"] for c in classifiers if c in rows.index]
            stds = [rows.loc[c, "accuracy_std"] for c in classifiers if c in rows.index]
            ax.errorbar(xs, means, yerr=stds, fmt="o", ms=7, color=KIND_COLORS[kind],
                        ecolor=KIND_COLORS[kind], elinewidth=1.5, capsize=3, label=kind)
        chance = panel[panel["classifier"] == "dummy"]["accuracy_mean"]
        if len(chance):
            ax.axhline(chance.mean(), color=INK_MUTED, ls="--", lw=1, label="chance (majority class)")
        ax.set_title(labels.get(scheme, scheme), color=INK, fontsize=11)
        ax.set_xticks(range(len(classifiers)))
        ax.set_xticklabels(classifiers, color=INK)
        ax.set_ylim(0.35, 0.85)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_MUTED)
        ax.tick_params(colors=INK_MUTED, labelcolor=INK)
    axes[0][0].set_ylabel("Accuracy (mean ± std over folds)", color=INK)
    handles, names = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc="lower center", ncol=len(names))
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # leave room for the legend below the panels
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Replicate the ABIDE classification experiment.")
    parser.add_argument("--atlas", default="ho", choices=sorted(download.ROI_DERIVATIVES),
                        help="Reference atlas whose PCP ROI time series to use (default: ho).")
    parser.add_argument("--n-subjects", type=int, default=None,
                        help="Use only the first N subjects (default: all 871).")
    parser.add_argument("--sites", nargs="+", default=None, metavar="SITE_ID",
                        help="Restrict to these sites, e.g. --sites PITT OLIN (default: all sites).")
    parser.add_argument("--kinds", nargs="+", default=list(connectivity.KINDS), choices=connectivity.KINDS,
                        help="Connectivity measures to compare.")
    parser.add_argument("--classifiers", nargs="+", default=list(CLASSIFIER_NAMES), choices=CLASSIFIER_NAMES)
    parser.add_argument("--schemes", nargs="+", default=list(SCHEMES), choices=SCHEMES)
    parser.add_argument("--n-splits", type=int, default=10, help="Folds for intra-site CV.")
    parser.add_argument("--inner-cv", type=int, default=3,
                        help="Inner folds for choosing the SVC penalty on the training set (default: 3).")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers (-1 = all cores).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    filters = {"SITE_ID": args.sites} if args.sites else {}
    timeseries, phenotypic = download.fetch_roi_timeseries(
        args.atlas, n_subjects=args.n_subjects, verbose=0, **filters
    )
    y, sites = download.phenotypic_targets(phenotypic)
    if len(np.unique(y)) < 2:
        raise SystemExit(
            "Only one diagnostic group in the selected subjects (subjects are ordered by ID, "
            "so a small --n-subjects can be all ASD). Use more subjects or --sites."
        )
    timeseries, keep = connectivity.drop_constant_rois(timeseries)
    print(f"{len(timeseries)} subjects from {len(np.unique(sites))} sites; "
          f"{int(y.sum())} ASD / {int((y == 0).sum())} TC; "
          f"{keep.sum()} of {len(keep)} ROIs kept -> {keep.sum() * (keep.sum() - 1) // 2} features")

    results = run_experiment(timeseries, y, sites, kinds=args.kinds, classifiers=args.classifiers,
                             schemes=args.schemes, n_splits=args.n_splits, inner_cv=args.inner_cv,
                             n_jobs=args.n_jobs, random_state=args.seed)
    summary = summarize(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.atlas}_n{len(timeseries)}"
    results.to_csv(args.output_dir / f"{stem}_folds.csv", index=False)
    summary.to_csv(args.output_dir / f"{stem}_summary.csv", index=False)
    plot_results(summary, args.output_dir / f"{stem}_accuracy.png",
                 title=f"ABIDE, {args.atlas} atlas, n={len(timeseries)}")
    print(f"\nWrote {stem}_folds.csv, {stem}_summary.csv, {stem}_accuracy.png to {args.output_dir}\n")

    pd.set_option("display.width", 160)
    show = summary[["cv_scheme", "connectivity", "classifier", "accuracy_mean", "accuracy_std",
                    "accuracy_pooled", "sensitivity_mean", "specificity_mean", "hyperparameter_median"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
