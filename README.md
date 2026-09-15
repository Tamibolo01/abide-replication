# ABIDE Replication

New to the code? Start with [GUIDE.md](GUIDE.md), a plain-language tour of
the two approaches, the files and the vocabulary.

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
  connectivity_experiment.py  # cross-validation, scoring, results (step 4 + validation)
  vlm_slices.py     # VLM: 2D slices at single time points of the 4D volumes
  vlm_reports.py    # VLM: template reports from the phenotypic table
  vlm_model.py      # VLM: MaMA-style model, losses, training loop
  vlm_experiment.py # VLM: cross-validated ASD vs. control classification
results/            # gitignored: per-fold CSV, summary CSV, figure
cluster/            # Slurm scripts and guide for running on Yale's Bouchet cluster
notebooks/          # exploratory work (empty so far)
requirements.txt    # pinned package versions
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run on Yale's Bouchet cluster instead, see [cluster/README.md](cluster/README.md).

## Running

```bash
# 1. Download Harvard-Oxford ROI time series for all 871 subjects (~200 MB)
python src/download.py

# 2. Run the full experiment (about 40 minutes on 8 cores; --inner-cv 2 roughly halves it)
python src/connectivity_experiment.py --n-jobs -1

# Quick two-site test (under a minute)
python src/connectivity_experiment.py --sites PITT OLIN --n-splits 5
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

## A second model: vision-language model on single-time-point slices

`src/vlm_slices.py`, `src/vlm_reports.py`, `src/vlm_model.py` and `src/vlm_experiment.py`
implement a different approach to the same ASD vs. control task, modelled
on:

> Du, Y., Onofrey, J., & Dvornek, N. C. (2024). Multi-view and multi-scale
> alignment for contrastive language-image pre-training in mammography
> (MaMA). *IPMI 2025*. https://arxiv.org/abs/2409.18119

Instead of connectivity between regions, the visual input is a **2D slice
of a single time point** of the preprocessed 4D fMRI, and the model is a
CLIP-style vision-language model (VLM) trained to match such slices with a
text report about the subject.

### What the model sees

C-PAC's `func_preproc` volumes are residuals after nuisance regression, so
every voxel has zero mean over time. One volume is therefore not anatomy
but a map of where the BOLD signal is above or below its mean at that
instant, inside the brain mask. `vlm_slices.py` scales each volume by its
in-mask standard deviation, clips at ±3, and stores 16 evenly spaced time
points × 3 planes (axial, coronal, sagittal) × 7 slices through the central
60% of the brain as 76×76 uint8 images (about 0.6 MB per subject). The
100 MB volumes are downloaded one at a time and deleted after slicing,
because all 871 would need 87 GB.

### How MaMA's ingredients map onto ABIDE

| MaMA (mammography) | Here (resting-state fMRI) |
|---|---|
| Image: one mammogram (518 px) | One slice of one time point (76 px, fed at 140 px) |
| Report generated from tabular fields by a clinical template; meta keywords masked with p=0.8 | Report generated from the phenotypic table by the same segment structure (procedure, patient, image, cognition, findings, impression, assessment); site, age, sex, handedness, eye status and IQ masked with p=0.8; findings (diagnosis, DSM-IV-TR subtype, ADOS) never masked |
| Multi-view: CC and MLO of the same breast are positives; a single-image study uses an augmented copy | One slice per example; the second view is an augmented copy of the same slice (MaMA's single-image case). `--pair-mode time/plane/any` pairs another slice of the same recording instead; `--vv-weight 0` drops the loss |
| Multi-scale: global CLIP loss + symmetric local alignment (patches ↔ sentences) | Same, with sentence embeddings read at each `[SEP]` |
| DINOv2 ViT-B/14, fully fine-tuned | DINOv2 ViT-S/14 by default (`--image-model facebook/dinov2-base` for the paper's size), with gradient checkpointing |
| BioMedLM 2.7B with LoRA (BioClinicalBERT as smaller baseline) | Bio_ClinicalBERT with LoRA (rank 8 on query/value), base frozen |
| Loss = L_VV + L_VT(v,t) + L_VT(ṽ,t) + w·L_local, w=0 for the first 8k of 40k steps | Same, w=0 for the first 20% of steps |
| AdamW, lr 4e-5, wd 0.1, cosine schedule, bf16 | Same (bf16 on CUDA only; float32 on Apple MPS) |
| Zero-shot with meta information in the prompts; linear probe; full fine-tuning | Same three protocols |

Because the training reports contain the diagnosis, pre-training is
supervised. What makes it honest is that everything happens inside the
cross-validation of `connectivity_experiment.py` (10-fold site-stratified or
leave-one-site-out), split **by subject**: the model is trained on the
training subjects' slices and reports and then scored on the held-out
subjects, one score per subject obtained by averaging over that subject's
slices. Zero-shot classification compares the image embedding with two
candidate reports that share the subject's meta information and differ
only in the findings.

### Running

```bash
pip install -r requirements.txt          # now includes torch, transformers, peft

# 1. Build the slice bank (downloads 100 MB per subject, ~30 s each; deletes the volume afterwards)
python src/vlm_slices.py --sites PITT        # 50 subjects, ~15 minutes
python src/vlm_slices.py                     # all 871 subjects, several hours and 87 GB of transfer

# 2. Train and evaluate (per fold: pre-training, then zero-shot and linear probe on the test subjects)
python src/vlm_experiment.py --sites PITT --n-splits 5 --steps 300
python src/vlm_experiment.py --schemes intra inter --steps 1000 --eval zeroshot linear finetune

# Baseline without any pre-training: off-the-shelf DINOv2 features + linear probe
python src/vlm_experiment.py --steps 0 --eval linear
```

On an M1 Pro (16 GB) a pre-training step at 140 px and batch 32 takes about
1.5 s, so 1000 steps are about 25 minutes per fold. `--folds` runs a subset
of folds; results are written after every fold. On a GPU with memory to
spare, `--no-grad-checkpointing` is about 30% faster. See `--help` for the
model, loss and schedule options.

Outputs in `results/`: `vlm_<tag>_folds.csv` (one row per fold and
protocol, with accuracy, balanced accuracy, AUC, sensitivity, specificity),
`_summary.csv` (mean/std over folds and accuracy pooled over test
subjects), `_scores.csv` (one score per test subject) and `_history.csv`
(training losses).

### First results (smoke scale: one site, 300 steps)

Both runs use the 50 quality-checked PITT subjects (24 ASD / 26 TC),
5-fold site-and-diagnosis-stratified CV, 4 evaluation time points × 21
slices per subject, seed 0, on an M1 Pro (about 10 minutes per fold for the
pre-trained model). Chance is 0.52 (majority class).

| Model | Protocol | Accuracy | Pooled | Balanced acc. | AUC | Sens. | Spec. |
|---|---|---|---|---|---|---|---|
| DINOv2-S features, no pre-training | linear probe | 0.54 ± 0.21 | 0.54 | 0.55 | 0.52 | 0.54 | 0.55 |
| MaMA-style, 300 steps | linear probe | 0.48 ± 0.19 | 0.48 | 0.48 | 0.50 | 0.53 | 0.43 |
| MaMA-style, 300 steps | zero-shot | 0.42 ± 0.08 | 0.42 | 0.40 | 0.33 | 0.08 | 0.73 |

Nothing beats chance at this scale, which is what 40 training subjects
from one site and 300 steps should give; the run is a functional check,
not an evaluation. Two observations from `results/pitt_mama300_history.csv`
matter for the next, larger run:

- **The multi-view loss learns, the image-text loss does not.** L_VV
  falls from 3.4 to about 2.6 (ln 32 = 3.47 is the random level) while
  L_VT stays at 3.42 and the local loss at 3.43. With template reports,
  every subject of the same class produces a nearly identical report once
  the meta keywords are masked, so most texts in a batch of 32 are
  duplicates of each other and the contrastive objective has a floor near
  ln 16 = 2.8; 300 steps at 4e-5 with warm-up and cosine decay do not get
  there. Longer training, a smaller `--mask-prob`, or richer report
  fields (e.g. more phenotypic scores) would give the text tower something
  to separate.
- **Zero-shot is biased to "control".** The test prompts carry unmasked
  meta information and no ADOS sentence, so they sit slightly outside the
  training distribution; both classes get negative scores and the sign
  test picks control most of the time. Calibrating the decision threshold
  on training subjects, or masking less, would remove the bias.

To evaluate the approach properly, build the full slice bank
(`python src/vlm_slices.py`, all 871 subjects) and run with
`--schemes intra inter --steps 2000` or more; the connectivity pipeline
above is the reference to beat (67.9% pooled inter-site accuracy).

### Departures from MaMA

- **Smaller everything.** ViT-S instead of ViT-B, Bio_ClinicalBERT instead
  of BioMedLM, 140 px instead of 518 px, hundreds or a few thousand steps
  instead of 40k, so that a fold trains on a laptop. All are flags.
- **No real reports.** ABIDE has no radiology reports; the reports are
  templated from the phenotypic table, as MaMA's are from its tabular
  fields, but the vocabulary is much narrower.
- **One image per example.** MaMA pairs the two X-ray projections of a
  breast; a single fMRI time point has no second acquisition, so the
  default follows MaMA's single-image case, an augmented copy of the same
  slice. Pairing another frame or plane of the same recording is
  available (`--pair-mode`), but then the report describes only the
  first picture.
- **Validation** follows this project's subject-level CV rather than
  MaMA's fixed train/test split.

