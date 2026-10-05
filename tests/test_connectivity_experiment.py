"""connectivity_experiment.py: cross-validation guarantees, scoring and the end-to-end loop on synthetic data."""

import warnings

import numpy as np
import pandas as pd
import pytest

import connectivity_experiment as ce


def _check_partition(train, test, n):
    assert set(train).isdisjoint(test)
    assert sorted(np.concatenate([train, test]).tolist()) == list(range(n))


def test_intra_splits_are_disjoint_exhaustive_and_stratified(targets):
    y, sites = targets
    folds = list(ce.cv_splits("intra", y, sites, n_splits=3, random_state=0))
    assert [name for name, _, _ in folds] == ["fold00", "fold01", "fold02"]
    for _, train, test in folds:
        _check_partition(train, test, 24)
        assert len(test) == 8
        strata = [f"{sites[i]}|{y[i]}" for i in test]
        assert sorted(set(strata)) == ["A|0", "A|1", "B|0", "B|1"]
        assert all(strata.count(s) == 2 for s in set(strata))


def test_intra_splits_are_reproducible_from_the_seed(targets):
    y, sites = targets
    a = list(ce.cv_splits("intra", y, sites, n_splits=3, random_state=0))
    b = list(ce.cv_splits("intra", y, sites, n_splits=3, random_state=0))
    c = list(ce.cv_splits("intra", y, sites, n_splits=3, random_state=1))
    for (_, train_a, test_a), (_, train_b, test_b) in zip(a, b):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
    assert sorted(a[0][2].tolist()) != sorted(c[0][2].tolist())


def test_inter_splits_hold_out_exactly_one_whole_site(targets):
    y, sites = targets
    folds = list(ce.cv_splits("inter", y, sites))
    assert len(folds) == 2
    covered = []
    for name, train, test in folds:
        _check_partition(train, test, 24)
        assert set(sites[test]) == {name}
        assert name not in set(sites[train])
        covered.extend(test.tolist())
    assert sorted(covered) == list(range(24))


def test_cv_splits_rejects_one_site_and_unknown_scheme(targets):
    y, sites = targets
    with pytest.raises(ValueError, match="at least two sites"):
        list(ce.cv_splits("inter", y, np.array(["A"] * 24)))
    with pytest.raises(ValueError, match="scheme must be one of"):
        list(ce.cv_splits("bogus", y, sites))


def test_score_on_a_hand_computed_example():
    metrics = ce.score(np.array([1, 1, 1, 0, 0]), np.array([1, 0, 1, 0, 1]))
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["sensitivity"] == pytest.approx(2 / 3)
    assert metrics["specificity"] == pytest.approx(0.5)


def test_score_all_negative_predictions_gives_zero_sensitivity_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = ce.score(np.array([1, 1, 0, 0]), np.zeros(4, dtype=int))
    assert metrics["sensitivity"] == 0.0
    assert metrics["specificity"] == 1.0


def test_make_classifier_names_and_dummy_majority():
    for name in ce.CLASSIFIER_NAMES:
        clf = ce.make_classifier(name)
        assert hasattr(clf, "fit") and hasattr(clf, "predict")
    with pytest.raises(ValueError, match="unknown classifier"):
        ce.make_classifier("forest")
    dummy = ce.make_classifier("dummy").fit(np.zeros((3, 2)), np.array([1, 1, 0]))
    assert dummy.predict(np.zeros((2, 2))).tolist() == [1, 1]


def test_chosen_hyperparameter_comes_from_the_grids(rng):
    X = rng.standard_normal((30, 5))
    y = (X[:, 0] > 0).astype(int)
    assert np.isnan(ce.chosen_hyperparameter(ce.make_classifier("dummy").fit(X, y)))
    svc = ce.make_classifier("svc_l2", inner_cv=2).fit(X, y)
    assert ce.chosen_hyperparameter(svc) in ce.SVC_C_GRID
    ridge = ce.make_classifier("ridge").fit(X, y)
    assert ce.chosen_hyperparameter(ridge) in ce.RIDGE_ALPHA_GRID


def _fold_row(fold, n_test, accuracy):
    return {
        "cv_scheme": "inter",
        "fold": fold,
        "connectivity": "tangent",
        "classifier": "ridge",
        "n_train": 100 - n_test,
        "n_test": n_test,
        "hyperparameter": 1.0,
        "accuracy": accuracy,
        "sensitivity": accuracy,
        "specificity": accuracy,
    }


def test_summarize_pools_accuracy_by_test_set_size():
    """Leave-one-site-out: a 10-subject site must not count as much as a 90-subject one."""
    results = pd.DataFrame([_fold_row("A", 10, 1.0), _fold_row("B", 90, 0.0)])
    summary = ce.summarize(results)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["accuracy_mean"] == pytest.approx(0.5)
    assert row["accuracy_pooled"] == pytest.approx(0.1)
    assert row["n_folds"] == 2


def test_run_experiment_end_to_end_on_synthetic_data(timeseries, targets, tmp_path):
    y, sites = targets
    results = ce.run_experiment(
        timeseries,
        y,
        sites,
        kinds=("correlation", "tangent"),
        classifiers=ce.CLASSIFIER_NAMES,
        schemes=ce.SCHEMES,
        n_splits=3,
        inner_cv=2,
        n_jobs=1,
        random_state=0,
        verbose=0,
    )
    assert len(results) == 40  # (3 intra + 2 inter folds) x 2 kinds x 4 classifiers
    expected = {
        "cv_scheme",
        "fold",
        "connectivity",
        "classifier",
        "n_train",
        "n_test",
        "hyperparameter",
        "accuracy",
        "sensitivity",
        "specificity",
    }
    assert expected <= set(results.columns)
    assert (results["n_train"] + results["n_test"] == 24).all()
    for column in ("accuracy", "sensitivity", "specificity"):
        assert results[column].between(0, 1).all()
    dummy = results[results["classifier"] == "dummy"]  # always one class, so exactly one recall is 1
    np.testing.assert_allclose(dummy["sensitivity"] + dummy["specificity"], 1.0)

    summary = ce.summarize(results)
    assert len(summary) == 2 * 2 * 4
    path = tmp_path / "accuracy.png"
    ce.plot_results(summary, path)
    assert path.stat().st_size > 0
