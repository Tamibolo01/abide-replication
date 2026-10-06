# Research journal: can a computer tell autism from a resting brain scan?

*Written 6 October 2026 for readers without a machine-learning background. The plain-language tour of the code is GUIDE.md; the technical details and results tables are in README.md.*

## The question

ABIDE is a public dataset of resting-state fMRI scans from 871 people at 20 sites (hospitals and universities), about half with an autism diagnosis. Resting-state fMRI records how blood-oxygen signal rises and falls in every part of the brain while the person lies still for a few minutes. The question of this project is whether a computer can learn, from those recordings alone, who has an autism diagnosis, and whether what it learns still works at a hospital it has never seen.

## Two approaches, two pipelines

**Pipeline A: connectivity (a replication of Abraham et al., 2017).** The brain is divided into about 100 regions using a standard atlas. For every pair of regions we ask: how much do their signals rise and fall together? That gives about 5,000 numbers per person, a "connectivity fingerprint". A simple classifier (a weighted sum of those numbers) is trained to separate the two groups. This replicates a published paper step by step so that we have a known reference point.

**Pipeline B: a vision-language model (adapted from MaMA, a mammography model).** Here the computer never sees a time series. It sees single pictures: one slice of the brain at one instant of the scan, which is a map of where the signal is above or below its usual level at that moment. Each picture is paired with a short written report generated from the person's record ("a 14-year-old right-handed male ... Impression: autism."). The model has two halves, one that reads pictures and one that reads text, and it is trained to match each picture to its report. To diagnose a new person, we show the model their pictures and two candidate reports, "autism" and "typical development", and ask which one the pictures resemble more.

## What we have found so far

| | Unseen-site accuracy | Chance level |
|---|---|---|
| Pipeline A, best setting (tangent connectivity, ridge classifier) | 68% of 871 people | 54% |
| Pipeline B, small trial (50 people, one site) | no better than chance | 52% |

**Pipeline A works and matches the paper.** Trained on 19 sites and tested on the 20th, it identifies the right group for about two in three people, well above the 54% you would get by always guessing "control". The best way to measure connectivity (called tangent embedding) came out on top, as in the paper. Getting there required auditing our own code: a first version had three mistakes (signals were not standardised before computing connectivity, one setting was fixed instead of tuned, and the wrong definition of chance was used). Fixing them changed the conclusion, which is the main lesson of the project so far: before blaming a disagreement with a paper on the paper, check your own pipeline one change at a time.

**Pipeline B has only been run at toy scale** (50 people from one site, a few minutes of training), where it is no better than guessing. That is expected at that scale and not yet a verdict on the approach.

**This week: a "does the machinery work?" check for pipeline B.** Before spending days of cluster time, we took 10 people from one site (5 autistic, 5 controls) and asked the model to learn them by heart. A model that cannot even memorise ten people it has seen has a broken pipeline, not a hard problem. The check passed: after a few hundred training steps the model classified all 10 correctly and 89% of their individual pictures; every component of the training signal fell below its "random guessing" level; and the saved model, reloaded from disk and given records with the diagnosis blanked out, produced exactly the same predictions. The notebook (`notebooks/pitt10_pipeline_check.ipynb`) also displays everything the model receives, pictures and text, so that nothing is hidden.

## How we make sure the results can be trusted

- **No peeking.** Every number comes from people the model never saw during training. Anything learned from data (even the averaging step in connectivity) is learned from the training group only. Automated tests now check this on synthetic data every time the code changes.
- **Quality gates.** The code has a test suite and style checks that must pass before any change is kept (`make check`), and every reported number is tied to the exact command and code version that produced it.
- **Honest chance levels.** "Chance" is what you get by always guessing the larger group, which is 54%, not 50%.

## Next steps

1. **Run pipeline B at real scale on the Yale cluster**: all 50 PITT participants with proper cross-validation, then all 20 sites, with longer training and the larger picture model. This is the first real test of whether single-instant pictures carry diagnostic information.
2. **Compare the two pipelines fairly**, on the same people and the same splits, including the leave-one-site-out test that matters clinically.
3. **Give the text half more to work with.** The reports are templated, so once personal details are hidden most of them read the same. Hiding less, or adding more clinical fields, should help the model learn the picture-to-text link, which was the slowest part of training.
4. **Return to the biology.** For pipeline A, find which region pairs carry the most weight in the classifier and check whether they match networks reported in the autism literature.
5. **Write-up and poster** once step 1 has run. A first A0 poster is in `poster/`.

## A few terms

- **Cross-validation**: training on most people and testing on the rest, repeated so everyone is tested once.
- **Leave-one-site-out**: train on 19 sites, test on the 20th; the realistic clinical scenario.
- **Leakage**: any way test information can influence training; it inflates results and is the main thing the tests guard against.
- **Accuracy vs chance**: a result only means something when compared with the accuracy of always guessing the larger group.
- **Loss**: the model's running error score during training; falling is learning.
