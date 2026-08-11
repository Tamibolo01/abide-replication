"""
Fetch ABIDE (Autism Brain Imaging Data Exchange) resting-state data via nilearn.

Replication target: Abraham et al. 2017, "Deriving reproducible biomarkers
from multi-site resting-state data: The ABIDE autism dataset".

Responsibilities:
- Download the ABIDE preprocessed (PCP) dataset subset needed for this
  replication using nilearn.datasets.fetch_abide_pcp.
- Store downloaded files under data/ (gitignored).
- Return/organise phenotypic data (site, diagnosis, age, etc.) alongside
  the functional derivatives needed for connectivity estimation.
"""
