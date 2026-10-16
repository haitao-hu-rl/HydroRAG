"""Regression tests for numeric-claim extraction in the hallucination guard.

The guard must not report a fabricated value merely because PDF text extraction
mangled the range dash in a sentence such as "减小45%～50%". Real examples from
this corpus include "一--" (U+4E00 followed by two hyphens) in place of "～".
"""

import pytest

from hydrorag.validation import (
    audit_numeric_claims,
    extract_numeric_claims,
)


@pytest.mark.parametrize(
    "text",
    [
        "减小45%～50%",       # fullwidth tilde (correct form)
        "减小45%~50%",        # ascii tilde
        "减小45％～50％",      # fullwidth percent signs
        "减小45%-50%",        # hyphen
        "减小45％一--50％",    # observed PDF artefact: U+4E00 + two hyphens
        "减小45%－50%",       # fullwidth hyphen-minus
        "减小45%—50%",        # em dash
        "减小45%–50%",        # en dash
    ],
)
def test_range_dash_never_produces_a_negative_claim(text):
    claims = extract_numeric_claims(text)
    assert (45.0, "%") in claims
    assert (50.0, "%") in claims
    # A range separator must never be read as a negative sign.
    assert not any(value < 0 for value, _ in claims), claims


@pytest.mark.parametrize(
    "text",
    ["减小45%至50%", "减小45%到50%", "减小45%~50%"],
)
def test_range_phrasings_agree(text):
    assert extract_numeric_claims(text) == [(45.0, "%"), (50.0, "%")]


def test_units_survive_range_normalisation():
    """The unit attached to the first number must not be swallowed."""

    assert extract_numeric_claims("3-5米") == [(3.0, ""), (5.0, "m")]
    assert extract_numeric_claims("45%-50%") == [(45.0, "%"), (50.0, "%")]
    assert extract_numeric_claims("350-420 m³/s") == [(350.0, ""), (420.0, "m3/s")]


def test_genuine_negative_values_are_preserved():
    """A real negative reading must still be extracted as negative."""

    claims = extract_numeric_claims("水位偏差为 -0.35 米")
    assert (-0.35, "m") in claims


def test_audit_accepts_answer_when_evidence_has_pdf_artefact():
    """The exact false positive observed against the live index."""

    evidence = "参与运行可以使其减小45％一--50％。"
    answer = "参与运行可以使其减小45％～50％ [E1]。"
    audit = audit_numeric_claims(answer, evidence)
    assert audit["supported"] is True
    assert audit["unsupported"] == []


def test_audit_still_detects_a_fabricated_number():
    """The fix must not weaken genuine hallucination detection."""

    evidence = "设计流量350 m³/s，校核流量420 m³/s。"
    answer = "设计流量 999 m³/s [E1]。"
    audit = audit_numeric_claims(answer, evidence)
    assert audit["supported"] is False
    assert {"value": 999.0, "unit": "m3/s"} in audit["unsupported"]


def test_audit_still_detects_a_fabricated_unit():
    evidence = "设计流量350 m³/s。"
    answer = "设计流量 350 m³/h [E1]。"
    audit = audit_numeric_claims(answer, evidence)
    assert audit["supported"] is False
