"""
Cross-validation and scoring for the ASD vs. control classification task.

Replication target: Abraham et al. 2017, "Deriving reproducible biomarkers
from multi-site resting-state data: The ABIDE autism dataset".

Responsibilities:
- Take connectivity features (from connectivity.py) and phenotypic labels
  (from download.py) as input.
- Run cross-validated classification (e.g. leave-site-out or stratified
  k-fold via scikit-learn) to evaluate diagnostic classification
  performance across sites.
- Score and report results (e.g. accuracy, ROC-AUC) and persist them to
  results/ (gitignored).
"""
