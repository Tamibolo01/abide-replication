"""vlm_reports.py: the text side of the VLM. Masking, the zero-shot prompts, and what must never leak into them."""

import numpy as np
import pytest

import vlm_reports as vr


def test_subject_fields_complete_row(phenotypic):
    assert vr.subject_fields(phenotypic.iloc[0]) == {
        "site": "SITEA",
        "age": "14",
        "sex": "male",
        "handedness": "right-handed",
        "eyes": "open",
        "fiq": "110",
        "fiq_band": "in the average range",
        "asd": True,
        "dsm_subtype": "autism",
        "ados_total": "12",
        "ados_severity": "7",
    }


def test_subject_fields_treats_nan_and_minus_9999_as_missing(phenotypic):
    row1 = vr.subject_fields(phenotypic.iloc[1])
    assert row1["asd"] is False
    assert row1["fiq"] is None and row1["fiq_band"] is None
    assert row1["ados_total"] is None and row1["ados_severity"] is None
    assert row1["handedness"] == "left-handed" and row1["eyes"] == "closed"
    row2 = vr.subject_fields(phenotypic.iloc[2])
    assert row2["handedness"] is None and row2["sex"] is None and row2["eyes"] is None


@pytest.mark.parametrize(
    "fiq, band",
    [
        (None, None),
        (84, "below the average range"),
        (85, "in the average range"),
        (115, "in the average range"),
        (116, "above the average range"),
    ],
)
def test_iq_band_boundaries(fiq, band):
    assert vr.iq_band(fiq) == band


def test_position_words_and_bad_plane():
    assert [vr.position_words("axial", p) for p in (0.0, 0.5, 1.0)] == ["inferior", "mid-level", "superior"]
    with pytest.raises(ValueError, match="plane must be one of"):
        vr.position_words("oblique", 0.5)


def test_unmasked_report_contains_the_fields(phenotypic):
    report = vr.subject_report(phenotypic.iloc[0], "axial", 0.5, mask_prob=0.0)
    assert len(report) == 7
    text = " ".join(report)
    assert "14-year-old right-handed male" in text
    assert "Impression: autism." in report
    assert "ADOS total score of 12" in text
    assert vr.MASK_TOKEN not in text
    control = vr.subject_report(phenotypic.iloc[1], "axial", 0.5, mask_prob=0.0)
    assert len(control) == 5
    assert control[-1] == "Impression: typical development."


def test_full_masking_hides_meta_but_never_the_diagnosis(phenotypic):
    row = phenotypic.iloc[0]
    unmasked = vr.subject_report(row, "axial", 0.5, mask_prob=0.0)
    masked = vr.subject_report(row, "axial", 0.5, rng=np.random.default_rng(0), mask_prob=1.0)
    assert all(vr.MASK_TOKEN in sentence for sentence in masked[:4])  # procedure, patient, image, IQ
    assert not any(vr.MASK_TOKEN in sentence for sentence in masked[4:])  # findings, impression, assessment
    assert masked[4:] == unmasked[4:]


def test_masking_is_reproducible_from_the_seed(phenotypic):
    row = phenotypic.iloc[0]
    a = vr.subject_report(row, "axial", 0.5, rng=np.random.default_rng(0))
    b = vr.subject_report(row, "axial", 0.5, rng=np.random.default_rng(0))
    c = vr.subject_report(row, "axial", 0.5, rng=np.random.default_rng(1))
    unmasked = vr.subject_report(row, "axial", 0.5, mask_prob=0.0)
    assert a == b
    assert a != c
    assert a != unmasked


@pytest.mark.parametrize("index", [0, 1])
def test_class_prompts_differ_only_in_the_label_sentences(phenotypic, index):
    """Zero-shot prompts: unmasked, identical meta, and nothing only an ASD record would carry."""
    prompts = vr.class_prompts(phenotypic.iloc[index], "coronal", 0.1)
    assert set(prompts) == {1, 0}
    asd, control = prompts[1], prompts[0]
    assert len(asd) == len(control)
    assert asd[:-2] == control[:-2]
    for sentence in asd + control:
        assert vr.MASK_TOKEN not in sentence
        assert "ADOS" not in sentence
        assert "DSM" not in sentence
    assert asd[-2] == "Findings: the subject has been diagnosed with autism spectrum disorder."
    assert asd[-1] == "Impression: autism."
    assert control[-1] == "Impression: typical development."
