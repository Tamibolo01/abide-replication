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

# 2. Run the full experiment (about 10 minutes on 8 cores)
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
5050 pairwise features), seed 0, about 10 minutes on 8 cores. Accuracy is
the mean over folds (± std); "pooled" counts correct predictions over all
test subjects, which matters for leave-one-site-out because sites differ in
size.

| Validation | Connectivity | Classifier | Accuracy | Pooled | Sensitivity | Specificity |
|---|---|---|---|---|---|---|
| Intra-site (10-fold) | correlation | ridge | **0.673 ± 0.054** | 0.673 | 0.601 | 0.734 |
| Intra-site | tangent | ridge | 0.662 ± 0.043 | 0.662 | 0.576 | 0.737 |
| Intra-site | correlation | svc_l2 | 0.647 ± 0.048 | 0.648 | 0.607 | 0.681 |
| Intra-site | tangent | svc_l2 | 0.645 ± 0.037 | 0.645 | 0.566 | 0.713 |
| Intra-site | partial correlation | ridge | 0.605 ± 0.030 | 0.605 | 0.495 | 0.702 |
| Intra-site | any | dummy (chance) | 0.576 ± 0.058 | 0.576 | 0.538 | 0.610 |
| Inter-site (leave-one-site-out) | tangent | ridge | **0.637 ± 0.075** | 0.658 | 0.581 | 0.691 |
| Inter-site | correlation | ridge | 0.631 ± 0.125 | 0.658 | 0.596 | 0.668 |
| Inter-site | tangent | svc_l2 | 0.610 ± 0.098 | 0.639 | 0.566 | 0.655 |
| Inter-site | correlation | svc_l2 | 0.610 ± 0.100 | 0.631 | 0.573 | 0.649 |
| Inter-site | partial correlation | ridge | 0.596 ± 0.069 | 0.611 | 0.477 | 0.705 |
| Inter-site | any | dummy (chance) | 0.518 ± 0.060 | 0.513 | 0.635 | 0.423 |

All 24 pipeline/classifier combinations are in `results/ho_n871_summary.csv`
and the figure is `results/ho_n871_accuracy.png` (regenerate with the
commands above; `results/` is not versioned).

How this compares with the paper:

- **Inter-site prediction works.** Every real classifier beats chance on
  unseen sites. The best pooled inter-site accuracy, 65.8%, is close to the
  paper's best of 66.8% (chance 53.7%), obtained with its data-driven MSDL
  atlas; the paper reports reference atlases performing near that level.
- **Ridge and l2-SVC are the best classifiers; l1-SVC is worse**, as in the
  paper, where sparse models and feature selection hurt.
- **Partial correlation is the worst connectivity measure**, as in the
  paper, which attributes this to the short ABIDE scans.
- **Tangent embedding and plain correlation are roughly tied here**, whereas
  the paper found tangent best. The gap likely comes from the departures
  listed below (no nested hyperparameter search, no region-level nuisance
  regression, full rather than 84-region Harvard-Oxford).
- **Inter-site scores vary more across folds than intra-site ones**, as the
  paper notes; the std doubles for most pipelines.

## Departures from the paper

- **Atlas.** Only the Harvard-Oxford reference atlas via the PCP's
  precomputed time series (111 regions, of which 10 are dropped because they
  are flat at some sites). The paper also derived atlases from the data and
  found a hand-picked 84-region subset of Harvard-Oxford slightly better than
  the full atlas.
- **Region-level nuisance regression.** The paper regressed CompCor and
  motion components out of the ROI signals; the PCP time series come without
  that step.
- **Hyperparameters.** The paper used nested cross-validation. Here the SVCs
  use C=1 and ridge picks its penalty by internal leave-one-out CV.
- **Sites.** Leave-one-site-out uses the 20 site identifiers in the
  phenotypic file (some institutions appear as two sub-sites), not 17.
- **Software.** Python 3.12, nilearn 0.14, scikit-learn 1.8, versus the
  paper's Python 2.7, nilearn 0.1.5, scikit-learn 0.17.
