from __future__ import annotations

import re
from typing import Any


NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,])[-+]?\d+(?:[,.]\d+)?(?:[eE][-+]?\d+)?\s*"
    r"(m²/s|m2/s|m³/s|m3/s|万m³|万m3|%|％|米水位|米|m³|m3|万立方米|万元|m|年)?"
)

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


def extract_numeric_claims(text: str) -> list[tuple[float, str]]:
    claims: list[tuple[float, str]] = []
    for match in NUMBER_PATTERN.finditer(text):
        if not match.group(1) and text[match.end() :].lstrip().startswith("."):
            continue
        raw = match.group(0).replace(",", "").strip()
        unit_text = (match.group(1) or "").strip()
        number_text = raw[: len(raw) - len(unit_text)].strip()
        try:
            claims.append((float(number_text), UNIT_ALIASES.get(unit_text, unit_text)))
        except ValueError:
            continue
    return claims


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
