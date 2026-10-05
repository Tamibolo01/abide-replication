"""vlm_experiment.py: fold selection, scoring, pooling and the datasets, without a model."""

import numpy as np
import pandas as pd
import pytest

import vlm_experiment as ve
import vlm_slices as vs

pytestmark = pytest.mark.vlm

FOLDS = [
    ("PITT", np.array([1, 2]), np.array([0])),
    ("OLIN", np.array([0, 2]), np.array([1])),
    ("NYU", np.array([0, 1]), np.array([2])),
]


def test_select_folds_by_name_index_or_default_name():
    assert ve.select_folds(FOLDS, None) == FOLDS
    assert [name for name, _, _ in ve.select_folds(FOLDS, ["OLIN"])] == ["OLIN"]
    assert [name for name, _, _ in ve.select_folds(FOLDS, ["2"])] == ["NYU"]
    assert [name for name, _, _ in ve.select_folds(FOLDS, ["fold00"])] == ["PITT"]
    with pytest.raises(SystemExit):
        ve.select_folds(FOLDS, ["nope"])


def test_score_subjects_perfect_scores_and_single_class_auc():
    metrics = ve.score_subjects(np.array([1, 0, 1, 0]), np.array([2.0, -1.0, 1.0, -2.0]))
    for key in ("accuracy", "sensitivity", "specificity", "balanced_accuracy", "auc"):
        assert metrics[key] == pytest.approx(1.0)
    # One class only (never the case for an ABIDE site): AUC is undefined and sklearn says so.
    with pytest.warns(UserWarning, match="single label"):
        single = ve.score_subjects(np.array([1, 1]), np.array([2.0, 1.0]))
    assert np.isnan(single["auc"])
    assert single["accuracy"] == 1.0


def test_summarize_pools_accuracy_by_test_set_size():
    rows = []
    for fold, n_test, accuracy in (("A", 10, 1.0), ("B", 90, 0.0)):
        rows.append(
            {
                "cv_scheme": "inter",
                "method": "zeroshot",
                "fold": fold,
                "n_test": n_test,
                "accuracy": accuracy,
                "balanced_accuracy": accuracy,
                "auc": accuracy,
                "sensitivity": accuracy,
                "specificity": accuracy,
            }
        )
    summary = ve.summarize(pd.DataFrame(rows))
    assert len(summary) == 1
    assert summary.loc[0, "accuracy_mean"] == pytest.approx(0.5)
    assert summary.loc[0, "accuracy_pooled"] == pytest.approx(0.1)
    assert summary.loc[0, "n_folds"] == 2


@pytest.fixture
def banks_and_rows(volume, phenotypic):
    file_ids = phenotypic["FILE_ID"].tolist()[:2]
    banks = {fid: vs.extract_slices(volume, n_timepoints=4, n_slices=3) for fid in file_ids}
    rows = {fid: phenotypic.iloc[i] for i, fid in enumerate(file_ids)}
    return banks, rows, file_ids


def test_pair_dataset_self_mode_items(banks_and_rows):
    banks, rows, file_ids = banks_and_rows
    dataset = ve.PairDataset(banks, rows, file_ids, planes=("axial", "coronal"), pair_mode="self", seed=0)
    assert len(dataset) == 2 * 2 * 4 * 3  # subjects x planes x time points x slices
    item = dataset[0]
    np.testing.assert_array_equal(item["view1"], item["view2"])
    assert item["view1"].dtype == np.uint8 and item["view1"].shape == (vs.CANVAS, vs.CANVAS)
    assert isinstance(item["report"], list) and all(isinstance(s, str) for s in item["report"])
    with pytest.raises(ValueError, match="pair_mode must be one of"):
        ve.PairDataset(banks, rows, file_ids, planes=("axial",), pair_mode="random")


def test_labelled_slices_items(banks_and_rows):
    banks, _, file_ids = banks_and_rows
    dataset = ve.LabelledSlices(banks, file_ids, labels=[1, 0], planes=("axial",), n_timepoints=2)
    assert len(dataset) == 2 * 1 * 2 * 3
    image, label = dataset[0]
    assert image.dtype == np.uint8 and image.shape == (vs.CANVAS, vs.CANVAS)
    assert label == 1 and isinstance(label, int)
