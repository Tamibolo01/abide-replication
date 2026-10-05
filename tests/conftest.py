"""Shared synthetic fixtures for the test suite.

No network, no reads of data/ or results/, and every random draw is seeded,
so the suite runs anywhere in seconds (see CLAUDE.md). The sizes are tiny on
purpose. ``targets`` gives 2 sites x 2 labels x 6 subjects, so every site|label
stratum has at least n_splits=3 members and StratifiedKFold never warns.
"""

import numpy as np
import pandas as pd
import pytest

N_SUBJECTS, N_TIMEPOINTS, N_ROIS = 24, 40, 6


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def timeseries(rng):
    """24 subjects x (40 time points, 6 regions) of standard normal noise."""
    return [rng.standard_normal((N_TIMEPOINTS, N_ROIS)) for _ in range(N_SUBJECTS)]


@pytest.fixture
def targets():
    """(y, sites): labels alternate 1/0; the first 12 subjects are at site A, the rest at site B."""
    y = np.tile([1, 0], N_SUBJECTS // 2)
    sites = np.array(["A"] * (N_SUBJECTS // 2) + ["B"] * (N_SUBJECTS // 2))
    return y, sites


@pytest.fixture
def phenotypic():
    """Three rows shaped like the ABIDE phenotypic table.

    Row 0: ASD with every field present. Row 1: control with a missing IQ and
    the file's -9999 code in the ADOS columns. Row 2: control with missing
    handedness, sex and eye status. Text columns use the "-9999" sentinel
    rather than NaN, as the real file does.
    """
    return pd.DataFrame(
        {
            "FILE_ID": ["SITEA_0050001", "SITEB_0050002", "SITEA_0050003"],
            "SITE_ID": ["SITEA", "SITEB", "SITEA"],
            "DX_GROUP": [1, 2, 2],
            "DSM_IV_TR": [1, 0, 0],
            "AGE_AT_SCAN": [14.2, 30.0, 9.4],
            "SEX": [1.0, 2.0, np.nan],
            "HANDEDNESS_CATEGORY": ["R", "L", "-9999"],
            "EYE_STATUS_AT_SCAN": [1.0, 2.0, np.nan],
            "FIQ": [110.0, np.nan, 90.0],
            "ADOS_TOTAL": [12.0, -9999.0, -9999.0],
            "ADOS_GOTHAM_SEVERITY": [7.0, -9999.0, -9999.0],
        }
    )


@pytest.fixture
def volume(rng):
    """A (20, 24, 20, 8) float32 '4D scan': zero outside a box, noise inside it."""
    vol = np.zeros((20, 24, 20, 8), dtype=np.float32)
    vol[5:15, 6:18, 4:16, :] = rng.standard_normal((10, 12, 12, 8)).astype(np.float32)
    return vol
