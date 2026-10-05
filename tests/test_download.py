"""download.py: only the pure function. Everything else in the module touches the network."""

import numpy as np

import download


def test_phenotypic_targets_maps_dx_group_and_sites(phenotypic):
    y, sites = download.phenotypic_targets(phenotypic)
    assert y.tolist() == [1, 0, 0]
    assert np.issubdtype(y.dtype, np.integer)
    assert sites.dtype.kind == "U"
    assert sites.tolist() == ["SITEA", "SITEB", "SITEA"]
