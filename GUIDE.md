# Reading this project without a coding background

This guide is the entry point if you want to understand what the code
does without reading it line by line. The [README](README.md) has the
technical details and results.

## The question

Can a computer tell, from a resting-state brain scan (fMRI), whether a
person has autism? The dataset is ABIDE: 871 people scanned at 20
hospitals, about half with an autism diagnosis. The project tries two
approaches to answering that question and measures how often each is
right on people it has never seen.

## Two approaches, two pipelines

```
                   download.py  (get the data, once)
                        |
        +---------------+----------------------+
        |                                      |
  PIPELINE A: connectivity            PIPELINE B: vision-language model
  (replicates Abraham et al. 2017)    (adapts MaMA, Du et al. 2024)
        |                                      |
  connectivity.py                     vlm_slices.py    scans -> pictures
  signals -> "connectivity" numbers   vlm_reports.py   records -> text
        |                             vlm_model.py     the model
  connectivity_experiment.py          vlm_experiment.py
  train, test, report                 train, test, report
        |                                      |
     results/                               results/
```

**Pipeline A** summarises each scan as "how much does every pair of brain
regions rise and fall together", giving about 5,000 numbers per person,
and feeds them to a simple classifier. It works: about 68% correct on
scans from a hospital the model never saw, against a 54% chance level.

**Pipeline B** looks at one picture at a time: a single slice of the
brain at a single instant of the scan. A model with two halves learns to
match such pictures to short written reports about the person, and then
diagnoses a new person by asking which of two reports ("autism" /
"typical development") their pictures match better. It runs end to end
but has only been tested at a very small scale so far, where it is no
better than guessing.

## What each file does

| File | Pipeline | What it does | Run it yourself? |
|---|---|---|---|
| `src/download.py` | both | Downloads the data; extracts each subject's diagnosis and hospital | `python src/download.py` |
| `src/connectivity.py` | A | Region signals to connectivity numbers | No, used by the next file |
| `src/connectivity_experiment.py` | A | Splits subjects, trains classifiers, scores, saves tables and a figure | `python src/connectivity_experiment.py --n-jobs -1` |
| `src/vlm_slices.py` | B | Cuts 4D scans into small pictures, deletes the big files | `python src/vlm_slices.py --sites PITT` |
| `src/vlm_reports.py` | B | Writes a text report per subject from their record | `python src/vlm_reports.py` (prints examples) |
| `src/vlm_model.py` | B | The model, its training objectives and training loop | No, used by the next file |
| `src/vlm_experiment.py` | B | Splits subjects, trains the model, scores, saves tables | `python src/vlm_experiment.py --sites PITT --n-splits 5 --steps 300` |
| `README.md` | | Technical description, commands, results | |
| `requirements.txt` | | Exact software versions to install | |
| `data/` | | Downloads (not in git, too big) | |
| `results/` | | Output tables and figures (not in git) | |

## How to read one of the code files

Every file in `src/` starts with a few lines of notes: what the file is
for, what goes in and what comes out, how to run it, and which function
is worth reading first. Then come "technical notes" for people who know
the field, and then the code. Some things to know about the code:

- A **function** is a named recipe: `def name(ingredients):` followed by
  the steps. The text in triple quotes right under it explains what it
  does, what goes in and what comes out. You can read only those
  explanations and skip the steps.
- Lines starting with `#` are comments for humans; the computer ignores
  them. The experiment scripts have `# Step 1 of 4 ...` comments marking
  the flow.
- `main()` at the bottom of a runnable file is where execution starts.
  Read it top to bottom to see the whole story of that file.
- Options like `--sites PITT` or `--steps 300` are typed after the
  command to change what a script does. `python src/<file>.py --help`
  lists every option with a one-line explanation.

## How to run something small

```bash
source venv/bin/activate                          # switch on the project's Python
python src/connectivity_experiment.py --sites PITT OLIN --n-splits 5   # under a minute
python src/vlm_slices.py --sites PITT             # ~15 min, downloads 50 scans
python src/vlm_experiment.py --sites PITT --n-splits 5 --steps 300     # ~1 hour on a laptop
```

Each experiment prints a summary table and writes files into `results/`.

## How to read a results table

One row per way of doing things (for example "tangent connectivity +
ridge classifier", or "vision-language model, zero-shot"). Columns:

- **accuracy**: fraction of test subjects classified correctly, averaged
  over folds, with its spread (std) across folds.
- **accuracy_pooled**: the same but counting all test subjects together,
  which matters when folds differ in size (whole hospitals).
- **sensitivity**: fraction of autism subjects correctly detected.
- **specificity**: fraction of control subjects correctly left alone.
- **balanced_accuracy**: the average of those two.
- **auc**: how well the model ranks autism above control regardless of
  the threshold; 0.5 is guessing, 1.0 is perfect.
- **chance**: what you get by always predicting the larger group
  (control): 54%. Anything must beat this to mean something.

## Glossary

- **fMRI**: a scan that records brain activity over a few minutes by
  measuring blood oxygen (the "BOLD" signal) in every small cube of
  tissue (a **voxel**), taken every couple of seconds. The result is a
  3D movie.
- **Resting state**: the person lies still and does nothing in
  particular.
- **Atlas / region (ROI)**: a map dividing the brain into named areas.
  Averaging the voxels inside an area gives one activity signal per
  region, a **time series**.
- **Connectivity**: how strongly two regions' signals move together.
  **Correlation** measures this directly; **partial correlation**
  removes the influence of all other regions first; **tangent
  embedding** compares each person's whole pattern to the group average
  in a mathematically careful way.
- **Classifier**: a rule that turns a list of numbers into a yes/no.
  The ones here (**SVM**, **ridge**) are weighted sums with a threshold.
  **Regularisation** keeps the weights small so the rule does not
  memorise the training people.
- **Cross-validation, fold**: to measure honest accuracy, subjects are
  split into groups; each group takes a turn as the test set while the
  rest train the model. **Intra-site** mixes hospitals in every fold;
  **inter-site** holds out one entire hospital, the harder and more
  realistic test.
- **Leakage**: any way information about test subjects sneaks into
  training. The code is arranged so that it cannot (for example the
  tangent step is set up on training subjects only).
- **Chance level**: the accuracy of the dumbest possible rule.
- **Embedding**: a list of numbers that summarises a picture or a text
  so that similar things get similar lists.
- **Transformer / ViT / BERT**: the neural-network families used for
  the image tower (DINOv2, a vision transformer) and the text tower
  (Bio_ClinicalBERT, a medical language model).
- **Contrastive training (CLIP)**: teaching two towers by rewarding them
  when a matching picture and text get similar embeddings and
  non-matching ones do not.
- **LoRA**: a way to adjust a big language model by training a few
  small added pieces instead of the whole thing.
- **Zero-shot**: classifying without a dedicated classifier, by asking
  which of two written descriptions the picture matches better.
- **Linear probe**: freezing the model and training only a simple
  classifier on its embeddings. **Fine-tuning**: training the whole
  model for the task.
- **Step, batch, learning rate**: training happens in small steps; each
  step looks at a batch of 32 examples and nudges the model a little,
  with the learning rate setting how big the nudge is.

## Where the papers' ideas live in the code

| Idea | Where |
|---|---|
| Abraham et al. step 1-2, regions and signals | `download.fetch_roi_timeseries` (precomputed by ABIDE) |
| Step 3, connectivity measures | `connectivity.connectivity_features` |
| Step 4, classifiers with tuned regularisation | `connectivity_experiment.make_classifier` |
| Intra-site and inter-site validation | `connectivity_experiment.cv_splits` |
| MaMA's generated reports with masked details | `vlm_reports.report_from_fields` |
| MaMA's two views of one study | `vlm_experiment.PairDataset` |
| MaMA's global and local matching objectives | `vlm_model.MaMA.forward`, `vlm_model.local_alignment_loss` |
| MaMA's zero-shot, linear-probe and fine-tuning tests | `vlm_experiment.zeroshot_scores`, `linear_probe_scores`, `finetune_scores` |
