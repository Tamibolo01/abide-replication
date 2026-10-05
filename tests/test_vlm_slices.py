"""vlm_slices.py: 4D volume -> uint8 slice bank, on a synthetic volume."""

import numpy as np
import pytest

import vlm_slices as vs


def test_brain_mask_is_the_box_that_varies_over_time(volume):
    expected = np.zeros(volume.shape[:3], dtype=bool)
    expected[5:15, 6:18, 4:16] = True
    np.testing.assert_array_equal(vs.brain_mask(volume), expected)


def test_pick_timepoints_is_evenly_spaced_or_everything():
    assert vs.pick_timepoints(100, 4).tolist() == [0, 33, 66, 99]
    assert vs.pick_timepoints(3, 16).tolist() == [0, 1, 2]


def test_pick_slices_stays_inside_the_brain(volume):
    indices, positions = vs.pick_slices(vs.brain_mask(volume), axis=2, n_slices=3)
    np.testing.assert_allclose(positions, [0.2, 0.5, 0.8])
    assert all(4 <= i <= 15 for i in indices)
    assert indices.tolist() == sorted(indices.tolist())


def test_to_uint8_maps_zero_to_mid_grey_and_clips():
    vol = np.zeros(100, dtype=np.float32)
    vol[:2] = [1.0, -1.0]  # far beyond +/- CLIP_SD once scaled by the tiny std
    mask = np.ones(100, dtype=bool)
    out = vs.to_uint8(vol, mask)
    assert out.dtype == np.uint8
    assert out[:3].tolist() == [255, 0, 128]
    assert (out[2:] == vs.BACKGROUND).all()
    flat = vs.to_uint8(np.zeros(100, dtype=np.float32), mask)  # zero std must not divide by zero
    assert (flat == vs.BACKGROUND).all()


def test_take_slice_centres_the_cut_on_the_canvas():
    volume = np.full((20, 24, 20), 7, dtype=np.uint8)
    image = vs.take_slice(volume, "axial", 3)
    assert image.shape == (vs.CANVAS, vs.CANVAS) and image.dtype == np.uint8
    assert (image == 7).sum() == 20 * 24
    assert (image[26:50, 28:48] == 7).all()
    assert image[0, 0] == image[-1, -1] == vs.BACKGROUND


def test_extract_slices_shapes_dtype_and_background(volume):
    bank = vs.extract_slices(volume, n_timepoints=4, n_slices=3)
    assert bank["timepoints"].tolist() == [0, 2, 5, 7]
    assert bank["n_timepoints_total"] == 8
    for plane in vs.PLANES:
        stack = bank[plane]
        assert stack.shape == (4, 3, vs.CANVAS, vs.CANVAS) and stack.dtype == np.uint8
        assert (stack[:, :, 0, 0] == vs.BACKGROUND).all()
        assert stack.min() < vs.BACKGROUND < stack.max()
        assert len(bank[f"{plane}_index"]) == 3
        np.testing.assert_allclose(bank[f"{plane}_position"], [0.2, 0.5, 0.8])


def test_extract_slices_rejects_a_constant_volume():
    with pytest.raises(ValueError, match="empty brain mask"):
        vs.extract_slices(np.zeros((4, 4, 4, 3), dtype=np.float32))


def test_slices_round_trip_and_missing_subjects_are_skipped(volume, tmp_path):
    bank = vs.extract_slices(volume, n_timepoints=2, n_slices=2)
    path = vs.slices_path("X", tmp_path)
    assert path == tmp_path / "X.npz"
    np.savez_compressed(path, **bank)
    loaded = vs.load_slices(["X", "missing"], tmp_path)
    assert list(loaded) == ["X"]
    for key, value in bank.items():
        np.testing.assert_array_equal(loaded["X"][key], value)
