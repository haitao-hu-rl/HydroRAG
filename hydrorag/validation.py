from __future__ import annotations

import re
from typing import Any


NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,])[-+]?\d+(?:[,.]\d+)?(?:[eE][-+]?\d+)?\s*"
    r"(m²/s|m2/s|m³/s|m3/s|万m³|万m3|%|％|米水位|米|m³|m3|万立方米|万元|m|年)?"
)

# PDF text extraction mangles the range dash "～" into assorted sequences such as
# "一--", "－", "-" or "~". When such a dash sits between two numbers -- optionally
# with a unit attached to the first, e.g. "45％一--50％" -- it separates a range and
# must not be read as the sign of the second number; otherwise the evidence yields
# -50 while the answer yields +50 and the guard reports a fabricated value.
#
# Only the dash itself is replaced, and the captured unit is written back, so
# "45％一--50％" becomes "45％ 50％" and both numbers keep their unit.
_UNIT_TAIL = r"(?:m²/s|m2/s|m³/s|m3/s|万m³|万m3|%|％|米水位|米|m³|m3|万立方米|万元|m|年)?"
_RANGE_DASH = re.compile(
    rf"(?<=\d)(?P<unit>{_UNIT_TAIL})\s*[-~～－—–一]+\s*(?=\d)"
)


def extract_numeric_claims(text: str) -> list[tuple[float, str]]:
    claims: list[tuple[float, str]] = []
    normalised = _normalise_range_dashes(text)
    for match in NUMBER_PATTERN.finditer(normalised):
        if not match.group(1) and normalised[match.end() :].lstrip().startswith("."):
            continue
        raw = match.group(0).replace(",", "").strip()
        unit_text = (match.group(1) or "").strip()
        unit = UNIT_ALIASES.get(unit_text, unit_text)
        number_text = raw[: len(raw) - len(unit_text)].strip()
        try:
            value = float(number_text)
        except ValueError:
            continue
        claims.append((value, unit))
    return claims


def _normalise_range_dashes(text: str) -> str:
    """Replace range separators between two numbers with a plain space.

    Mirrors the PDF artefacts seen in this corpus, e.g. "45％一--50％" or "45%-50%".
    The unit attached to the first number is preserved.
    """

    return _RANGE_DASH.sub(lambda m: f"{m.group('unit')} ", text)

UNIT_ALIASES = {
    "%": "%",
    "％": "%",
    "米": "m",
    "m": "m",
    "米水位": "m",
    "年": "年",
    "万元": "万元",
    "万立方米": "万m3",
    "万m³": "万m3",
    "m³": "m3",
    "m3": "m3",
    "m²/s": "m2/s",
    "m2/s": "m2/s",
    "m³/s": "m3/s",
    "m3/s": "m3/s",
    "万m³": "万m3",
    "万m3": "万m3",
}


def audit_numeric_claims(answer: str, evidence: str) -> dict[str, Any]:
    evidence_claims = extract_numeric_claims(evidence)
    answer_claims = extract_numeric_claims(answer)
    unsupported = [
        {"value": value, "unit": unit}
        for value, unit in answer_claims
        if not any(
            value == evidence_value
            and (not unit or not evidence_unit or unit == evidence_unit)
            for evidence_value, evidence_unit in evidence_claims
        )
    ]
    return {
        "supported": not unsupported,
        "unsupported": unsupported,
        "answer_claims": answer_claims,
        "evidence_claims": evidence_claims,
    }
