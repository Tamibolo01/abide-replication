# ABIDE Replication

A partial replication of:

> Abraham, A., Milham, M. P., Di Martino, A., Craddock, R. C., Samaras, D.,
> Thirion, B., & Varoquaux, G. (2017). Deriving reproducible biomarkers from
> multi-site resting-state data: The ABIDE autism dataset. *NeuroImage*, 147,
> 736–745.

The paper studies functional connectivity biomarkers for autism spectrum
disorder (ASD) classification using resting-state fMRI from the multi-site
ABIDE dataset. This project replicates a subset of that pipeline: fetching
ABIDE data, deriving ROI-based functional connectivity matrices, and
evaluating ASD vs. control classification performance under cross-validation.

## Project structure

```
data/               # gitignored, ABIDE downloads land here
src/
  download.py       # fetch ABIDE via nilearn
  connectivity.py   # ROI time series -> connectivity matrices
  experiment.py      # cross-validation and scoring
results/            # gitignored
notebooks/
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Status

Project scaffolding only — no pipeline logic implemented yet.
