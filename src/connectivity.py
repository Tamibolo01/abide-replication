"""
Build functional connectivity matrices from ROI time series.

Replication target: Abraham et al. 2017, "Deriving reproducible biomarkers
from multi-site resting-state data: The ABIDE autism dataset".

Responsibilities:
- Extract region-of-interest (ROI) time series from preprocessed fMRI data
  using a brain atlas (e.g. via nilearn NiftiMapsMasker/NiftiLabelsMasker).
- Compute connectivity matrices from the time series (e.g. correlation,
  partial correlation, tangent space embedding via
  nilearn.connectome.ConnectivityMeasure).
- Return connectivity features in a form suitable for use as inputs to the
  classifier in experiment.py.
"""
