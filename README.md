# ABIDE Replication

A partial replication of:

> Abraham, A., Milham, M. P., Di Martino, A., Craddock, R. C., Samaras, D.,
> Thirion, B., & Varoquaux, G. (2017). Deriving reproducible biomarkers from
> multi-site resting-state data: An Autism-based example. *NeuroImage*, 147,
> 736–745. https://doi.org/10.1016/j.neuroimage.2016.10.045

## What the paper does

The paper asks whether resting-state fMRI connectivity can predict autism
spectrum disorder (ASD) vs. typical control (TC) status on the large,
heterogeneous, multi-site ABIDE dataset, and in particular whether a
classifier trained on some acquisition sites generalises to sites it has
never seen. It compares choices at each step of a four-step pipeline:

1. **Region definition** – a reference atlas (Harvard-Oxford, Yeo,
   Craddock) or a data-driven one (ICA, K-Means, Ward, MSDL).
2. **Time-series extraction** – one signal per region per subject.
3. **Connectivity estimation** – correlation, partial correlation, or
   tangent-space embedding of the Ledoit-Wolf covariance matrix.
4. **Supervised learning** – l2-SVC, l1-SVC or ridge on the pairwise
   connectivity weights.

Validation uses 10-fold CV stratified by site and diagnosis (intra-site) and
leave-one-site-out CV (inter-site). Headline results: the best pipeline
(MSDL atlas, tangent embedding, l2-regularised classifier) reaches 67%
accuracy on the full 871-subject sample; inter-site accuracy tops out at
66.8% against a dummy-classifier chance level of 53.7%; reference atlases
come close to the best data-driven atlas; full correlations beat partial
correlations; sparse (l1) models and feature selection do worse.

## What this project replicates

The pipeline with a **reference atlas**, run on the **same 871
quality-checked subjects** as the paper, with **both validation schemes**,
**all three connectivity measures** and **all three classifiers**, plus the
dummy chance level. Steps 1–2 use the ROI time series that the ABIDE
Preprocessed Connectomes Project (PCP) already extracted with the
Harvard-Oxford atlas from C-PAC-preprocessed data (the paper's preprocessing).
The data-driven atlases (MSDL etc.) are out of scope.

## Project structure

```
data/               # gitignored, ABIDE downloads land here (~2 GB currently)
src/
  download.py       # fetch ABIDE via nilearn; phenotypic -> labels and sites
  connectivity.py   # ROI time series -> connectivity features (steps 1-3)
  experiment.py     # cross-validation, scoring, results (step 4 + validation)
results/            # gitignored: per-fold CSV, summary CSV, figure
notebooks/          # exploratory work (empty so far)
requirements.txt    # pinned package versions
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
# 1. Download Harvard-Oxford ROI time series for all 871 subjects (~200 MB)
python src/download.py

# 2. Run the full experiment (about 40 minutes on 8 cores; --inner-cv 2 roughly halves it)
python src/experiment.py --n-jobs -1

# Quick two-site test (under a minute)
python src/experiment.py --sites PITT OLIN --n-splits 5
```

Useful options: `--atlas cc200` (Craddock 200 instead of Harvard-Oxford;
downloads its time series first), `--kinds tangent`, `--classifiers svc_l2
ridge`, `--schemes inter`, `--seed`. See `--help`.

Outputs in `results/`: `<atlas>_n<N>_folds.csv` (one row per fold, pipeline
and classifier), `<atlas>_n<N>_summary.csv` (mean/std over folds, plus
accuracy pooled over test subjects) and `<atlas>_n<N>_accuracy.png`.

## Results

Full run on all 871 subjects, Harvard-Oxford atlas (101 regions kept,
5050 pairwise features), seed 0, 3 inner folds, about 42 minutes on 8
cores. Accuracy is the mean over folds (± std); "pooled" counts correct
predictions over all test subjects, which matters for leave-one-site-out
because sites differ in size. Chance is the majority-class rate, 53.7%.

| Validation | Connectivity | Classifier | Accuracy | Pooled | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Inter-site (leave-one-site-out) | tangent | ridge | **0.651 ± 0.112** | **0.679** | 0.591 | 0.706 |
| Inter-site | tangent | svc_l2 | 0.645 ± 0.108 | 0.673 | 0.578 | 0.706 |
| Inter-site | correlation | ridge | 0.639 ± 0.110 | 0.666 | 0.609 | 0.671 |
| Inter-site | correlation | svc_l2 | 0.632 ± 0.092 | 0.657 | 0.605 | 0.664 |
| Inter-site | partial correlation | ridge | 0.609 ± 0.091 | 0.619 | 0.511 | 0.701 |
| Inter-site | tangent | svc_l1 | 0.603 ± 0.090 | 0.615 | 0.539 | 0.668 |
| Inter-site | any | chance (majority) | 0.540 ± 0.085 | 0.537 | 0.000 | 1.000 |
| Intra-site (10-fold) | correlation | svc_l2 | **0.680 ± 0.050** | 0.680 | 0.609 | 0.741 |
| Intra-site | tangent | ridge | 0.679 ± 0.047 | 0.679 | 0.589 | 0.756 |
| Intra-site | tangent | svc_l2 | 0.675 ± 0.055 | 0.675 | 0.591 | 0.748 |
| Intra-site | correlation | ridge | 0.670 ± 0.050 | 0.670 | 0.606 | 0.726 |
| Intra-site | correlation | svc_l1 | 0.628 ± 0.050 | 0.628 | 0.554 | 0.690 |
| Intra-site | partial correlation | svc_l2 | 0.594 ± 0.032 | 0.594 | 0.424 | 0.740 |
| Intra-site | any | chance (majority) | 0.537 ± 0.016 | 0.537 | 0.000 | 1.000 |

All 24 pipeline/classifier combinations, with the chosen regularisation
values, are in `results/ho_n871_summary.csv`; the figure is
`results/ho_n871_accuracy.png` (regenerate with the commands above;
`results/` is not versioned).

How this compares with the paper:

- **Inter-site prediction works.** Every real classifier beats chance on
  unseen sites. The best pooled inter-site accuracy, 67.9%, matches the
  paper's best of 66.8% (chance 53.7%), which used its data-driven MSDL
  atlas; the paper reports reference atlases performing near that level.
- **Tangent embedding with an l2-regularised classifier (ridge or SVC) is
  the best inter-site pipeline**, the paper's headline methodological
  result. Intra-site, tangent and plain correlation are within a point of
  each other.
- **Ridge and l2-SVC beat l1-SVC** everywhere, as in the paper, where sparse
  models and feature selection hurt.
- **Partial correlation is the worst connectivity measure**, as in the
  paper, which attributes this to the short ABIDE scans.
- **Inter-site scores vary more across folds than intra-site ones**, as the
  paper notes; the std roughly doubles.

## Departures from the paper

- **Atlas.** Only the Harvard-Oxford reference atlas via the PCP's
  precomputed time series (111 regions, of which 10 are dropped because they
  are flat at some sites). The paper also derived atlases from the data and
  found a hand-picked 84-region subset of Harvard-Oxford slightly better than
  the full atlas.
- **Region-level nuisance regression.** The paper regressed CompCor and
  motion components out of the ROI signals; the PCP time series come without
  that step. Signals are detrended and z-scored here, as in the paper.
- **Hyperparameters.** Both tune regularisation by nested cross-validation
  on the training fold; the grids and inner-fold count (3 by default,
  `--inner-cv`) are this project's choices.
- **Sites.** Leave-one-site-out uses the 20 site identifiers in the
  phenotypic file (some institutions appear as two sub-sites), not 17.
- **Software.** Python 3.12, nilearn 0.14, scikit-learn 1.8, versus the
  paper's Python 2.7, nilearn 0.1.5, scikit-learn 0.17.

## Checking the pipeline against the paper

A first version of this pipeline found tangent embedding and plain
correlation tied, whereas the paper found tangent best. Before blaming the
paper departures, the pipeline itself was audited on the full sample, same
folds, one change at a time (script not kept; results summarised here):

| Suspect | Finding | Verdict |
|---|---|---|
| Signals not standardized before covariance | nilearn's `ConnectivityMeasure` applies `standardize` only for `kind="correlation"`; tangent and partial correlation were fed raw BOLD amplitudes. Explicit detrend + z-score raised tangent accuracy by 1–2 points in both schemes and made it the best measure. | Bug, fixed |
| Fixed SVC penalty C=1 | C=0.01 beat C=1 by about 3 points for tangent features, which are small numbers. | Bug, fixed with nested CV |
| Chance level | A random-label dummy gave about 51%; the paper's 53.7% is the majority-class rate (468/871). | Wrong baseline, fixed |
| Diagonal discarded from tangent matrices | Keeping the diagonal lowered accuracy slightly. | Kept as is |
| Ridge alpha grid too narrow | Chosen alphas (about 30–100) sat well inside the grid. | Not an issue |

The results above are from the corrected pipeline.

