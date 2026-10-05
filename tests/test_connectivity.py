"""connectivity.py: standardisation and leak-free feature extraction, on synthetic signals."""

import numpy as np
import pytest

import connectivity
from connectivity import KINDS


def test_drop_constant_rois_removes_bad_columns_from_every_subject(timeseries):
    ts = [t.copy() for t in timeseries]
    ts[0][:, 2] = 1.0  # flat in one subject
    ts[1][3, 0] = np.nan  # non-finite in another
    cleaned, keep = connectivity.drop_constant_rois(ts)
    assert keep.tolist() == [False, True, False, True, True, True]
    assert len(cleaned) == len(ts)
    assert all(c.shape == (40, 4) for c in cleaned)


def test_standardize_timeseries_gives_zero_mean_unit_variance_and_removes_trend(timeseries):
    ts = [t.copy() for t in timeseries]
    trend = np.arange(40, dtype=float)
    ts[0][:, 1] += 5.0 * trend  # strong linear drift in one region
    before = [t.copy() for t in ts]
    out = connectivity.standardize_timeseries(ts)
    for t in out:
        assert t.shape == (40, 6)
        np.testing.assert_allclose(t.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(t.std(axis=0, ddof=1), 1.0, atol=1e-6)
    assert abs(np.corrcoef(out[0][:, 1], trend)[0, 1]) < 1e-6
    for original, untouched in zip(ts, before):
        np.testing.assert_array_equal(original, untouched)  # no in-place edit


def test_make_connectivity_measure_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        connectivity.make_connectivity_measure("foo")


@pytest.mark.parametrize("kind", KINDS)
def test_training_features_do_not_depend_on_test_subjects(timeseries, kind):
    """The guarantee behind the cross-validation: fit() never sees a test subject."""
    train, test = timeseries[:16], timeseries[16:]
    X_alone, none, _ = connectivity.connectivity_features(kind, train)
    X_with, X_test, _ = connectivity.connectivity_features(kind, train, test)
    assert none is None
    assert X_alone.shape == (16, 15) and X_test.shape == (8, 15)
    assert np.isfinite(X_with).all() and np.isfinite(X_test).all()
    np.testing.assert_allclose(X_alone, X_with)


def test_tangent_features_change_when_test_subjects_leak_into_the_fit(timeseries):
    """Shows the previous test discriminates: fitting on everybody moves the tangent features."""
    train, test = timeseries[:16], timeseries[16:]
    X_train, _, _ = connectivity.connectivity_features("tangent", train, test)
    measure = connectivity.make_connectivity_measure("tangent")
    X_leaky = measure.fit_transform(connectivity.standardize_timeseries(train + test))[:16]
    assert not np.allclose(X_train, X_leaky)


def test_connectivity_matrices_are_square_and_symmetric(timeseries):
    mats = connectivity.connectivity_matrices("correlation", timeseries)
    assert mats.shape == (24, 6, 6)
    assert np.isfinite(mats).all()
    np.testing.assert_allclose(mats, np.transpose(mats, (0, 2, 1)), atol=1e-10)
