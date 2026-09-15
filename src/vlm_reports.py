"""
vlm_reports.py -- pipeline B, step 2: subject record -> a short text report.

ABIDE has no radiology reports, so write one from the phenotypic table, the
way MaMA does from its tabular fields: procedure, patient, image, cognition,
findings, impression, assessment. Returned as a list of sentences because the
local alignment loss works at the sentence level.
Personal details (site, age, sex, handedness, eyes, IQ) get blanked with
p = 0.8 during training so the model can't memorise people. Diagnosis never.
class_prompts() makes the two candidate reports (autism / typical) for
zero-shot classification: no masking, and no ADOS since that's unknown at test.
`python src/vlm_reports.py` prints examples.

Technical notes
---------------
Template-based text reports for ABIDE subjects: the text side of the VLM.

Model reference: Du, Y., Onofrey, J., & Dvornek, N. C. (2024). Multi-view and
multi-scale alignment for contrastive language-image pre-training in
mammography (MaMA). IPMI 2025. https://arxiv.org/abs/2409.18119

MaMA has no free-text radiology reports: it *generates* a report for every
image from the structured (tabular) fields shipped with the dataset, in the
segment order of a clinical report (study procedure, patient
meta-information, image meta-information, breast composition, findings,
clinical impression, overall assessment). Each meta-information keyword is
masked with probability 0.8 when the caption is built, so the model cannot
take the shortcut of memorising patient metadata instead of reading the
image. The findings are never masked.

This module does the same with the ABIDE phenotypic table. A report is a
list of sentences (the symmetric local alignment loss in vlm_model.py aligns
image patches with sentences, so sentence boundaries matter):

    1. procedure   resting-state fMRI, C-PAC preprocessing, site      meta
    2. patient     age, sex, handedness                               meta
    3. image       plane and position of the slice, eyes open/closed  meta*
    4. cognition   full-scale IQ and its band (cf. breast composition) meta
    5. findings    diagnosis, DSM-IV-TR subtype                       label
    6. impression  one-word summary of the diagnosis                  label
    7. assessment  ADOS total and calibrated severity, when measured  label

Maskable keywords are site, age, sex, handedness, eye status and IQ. The
plane and position of the slice (*) are left visible: they describe what
is actually in the picture and carry no information about the subject.

The "label" sentences are the only ones that separate the two classes.
class_prompts() builds the two candidate reports (ASD / typical control)
for a subject and slice from the meta fields alone, without masking, and
zero-shot classification picks the one the image embedding is closer to.
ADOS scores are only recorded for the ASD group, so they belong in the
training reports (like BI-RADS findings in MaMA) but never in the prompts.
"""

import numpy as np

PLANES = ("axial", "coronal", "sagittal")
MASK_PROB = 0.8
MASK_TOKEN = "[MASK]"  # BERT's mask token; any placeholder word works for other tokenizers

# DSM-IV-TR codes in the phenotypic file (0 = control).
DSM_SUBTYPES = {
    1: "autism",
    2: "Asperger syndrome",
    3: "pervasive developmental disorder not otherwise specified",
    4: "Asperger syndrome or pervasive developmental disorder",
}
HANDEDNESS = {"R": "right-handed", "L": "left-handed", "Ambi": "ambidextrous", "Mixed": "mixed-handed"}
MISSING_TEXT = {"", "-9999", "`", "nan", "None"}


def _number(row, key):
    """Numeric phenotypic field, or None when absent or coded -9999."""
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return None if (np.isnan(value) or value == -9999) else value


def _text(row, key):
    """String phenotypic field, or None when absent or coded as missing."""
    try:
        value = row[key]
    except KeyError:
        return None
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = str(value).strip()
    return None if value in MISSING_TEXT else value


def iq_band(fiq):
    if fiq is None:
        return None
    if fiq < 85:
        return "below the average range"
    if fiq > 115:
        return "above the average range"
    return "in the average range"


def subject_fields(row):
    """Clean the phenotypic fields of one subject into plain words.

    Parameters
    ----------
    row : mapping
        One row of the ABIDE phenotypic table (a pandas Series or a dict
        from DataFrame.to_dict("records")).

    Returns
    -------
    dict
        Keys: site, age, sex, handedness, eyes, fiq, fiq_band, asd (bool),
        dsm_subtype, ados_total, ados_severity. Missing values are None.
    """
    age = _number(row, "AGE_AT_SCAN")
    fiq = _number(row, "FIQ")
    ados_total = _number(row, "ADOS_TOTAL")
    severity = _number(row, "ADOS_GOTHAM_SEVERITY")
    return {
        "site": _text(row, "SITE_ID"),
        "age": None if age is None else f"{age:.0f}",
        "sex": {1: "male", 2: "female"}.get(_number(row, "SEX")),
        "handedness": HANDEDNESS.get(_text(row, "HANDEDNESS_CATEGORY")),
        "eyes": {1: "open", 2: "closed"}.get(_number(row, "EYE_STATUS_AT_SCAN")),
        "fiq": None if fiq is None else f"{fiq:.0f}",
        "fiq_band": iq_band(fiq),
        "asd": _number(row, "DX_GROUP") == 1,
        "dsm_subtype": DSM_SUBTYPES.get(_number(row, "DSM_IV_TR")),
        "ados_total": None if ados_total is None else f"{ados_total:.0f}",
        "ados_severity": None if severity is None else f"{severity:.0f}",
    }


def position_words(plane, position):
    """Describe where a slice sits from its relative position along its axis.

    ``position`` is in [0, 1] over the brain's bounding box along the slicing
    axis, in voxel-index order (see vlm_slices.py). On the MNI grid used by the
    PCP, the z index grows towards superior, the y index towards anterior,
    and the x index towards the *left* hemisphere.
    """
    third = 0 if position < 1 / 3 else 1 if position < 2 / 3 else 2
    words = {
        "axial": ("inferior", "mid-level", "superior"),
        "coronal": ("posterior", "mid", "anterior"),
        "sagittal": ("right hemisphere", "midline", "left hemisphere"),
    }
    if plane not in words:
        raise ValueError(f"plane must be one of {PLANES}, got {plane!r}")
    return words[plane][third]


def report_from_fields(fields, plane, position, rng=None, mask_prob=MASK_PROB, mask_token=MASK_TOKEN):
    """Assemble the report sentences for one (subject, slice) pair.

    Parameters
    ----------
    fields : dict
        As returned by subject_fields (possibly edited, see class_prompts).
    plane : str
        One of PLANES.
    position : float
        Relative position of the slice along its axis, in [0, 1].
    rng : numpy.random.Generator or None
        Source of randomness for meta-information masking.
    mask_prob : float
        Probability of replacing each meta keyword with ``mask_token``.
        0 disables masking (used for the zero-shot prompts).
    mask_token : str

    Returns
    -------
    list of str
        One string per sentence, in report order. Sentences whose fields are
        all missing are omitted.
    """
    rng = np.random.default_rng() if rng is None else rng

    def meta(word):
        """A meta keyword, masked with probability mask_prob."""
        if mask_prob > 0 and rng.random() < mask_prob:
            return mask_token
        return word

    sentences = []
    site = fields.get("site")
    sentences.append(
        "Resting-state functional MRI of the brain"
        + (f" acquired at the {meta(site)} site" if site else "")
        + ", preprocessed with the C-PAC pipeline."
    )

    patient = []  # e.g. "a 14-year-old right-handed male"
    if fields.get("age"):
        patient.append(f"{meta(fields['age'])}-year-old")
    if fields.get("handedness"):
        patient.append(meta(fields["handedness"]))
    patient.append(meta(fields["sex"]) if fields.get("sex") else "subject")
    sentences.append("The subject is a " + " ".join(patient) + ".")

    sentence = (
        f"This image is one time point of the recording, shown as a "
        f"{position_words(plane, position)} {plane} slice"
    )
    if fields.get("eyes"):
        sentence += f", eyes {meta(fields['eyes'])} during the scan"
    sentences.append(sentence + ".")

    if fields.get("fiq"):
        sentences.append(f"Full-scale IQ of {meta(fields['fiq'])}, {meta(fields['fiq_band'])}.")

    if fields["asd"]:
        finding = "Findings: the subject has been diagnosed with autism spectrum disorder"
        if fields.get("dsm_subtype"):
            finding += f", {fields['dsm_subtype']} on the DSM-IV-TR"
        sentences.append(finding + ".")
        sentences.append("Impression: autism.")
        if fields.get("ados_total"):
            assessment = f"Assessment: ADOS total score of {fields['ados_total']}"
            if fields.get("ados_severity"):
                assessment += f", calibrated severity {fields['ados_severity']} out of 10"
            sentences.append(assessment + ".")
    else:
        sentences.append(
            "Findings: the subject is a typically developing control with no autism spectrum disorder."
        )
        sentences.append("Impression: typical development.")
    return sentences


def subject_report(row, plane, position, rng=None, mask_prob=MASK_PROB, mask_token=MASK_TOKEN):
    """Training report for one subject and one slice (see module docstring)."""
    return report_from_fields(subject_fields(row), plane, position, rng, mask_prob, mask_token)


def class_prompts(row, plane, position):
    """The two candidate reports used for zero-shot classification.

    Both share the subject's meta information (unmasked) and the slice
    description; they differ only in the label sentences. ASD-only scores
    (DSM subtype, ADOS) are dropped because they are unknown at test time.

    Returns
    -------
    dict
        {1: sentences for "autism spectrum disorder", 0: sentences for
        "typically developing control"}, matching download.phenotypic_targets.
    """
    fields = subject_fields(row)
    fields.update(dsm_subtype=None, ados_total=None, ados_severity=None)
    prompts = {}
    for label in (1, 0):
        fields["asd"] = label == 1
        prompts[label] = report_from_fields(fields, plane, position, mask_prob=0.0)
    return prompts


if __name__ == "__main__":
    # Print example reports for the first few subjects of the phenotypic file.
    import pandas as pd

    import download

    phenotypic = pd.read_csv(download.DATA_DIR / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv")
    rng = np.random.default_rng(0)
    for _, row in phenotypic.head(2).iterrows():
        print(f"--- {row['FILE_ID']} (DX_GROUP={row['DX_GROUP']}) training report, masked:")
        print("\n".join(subject_report(row, "axial", 0.5, rng)))
        print("--- same subject, unmasked:")
        print("\n".join(subject_report(row, "axial", 0.5, rng, mask_prob=0.0)))
        print("--- zero-shot prompts:")
        for label, sentences in class_prompts(row, "axial", 0.5).items():
            print(f"[{label}]", " ".join(sentences))
        print()
