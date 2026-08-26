from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REMEDIATION_PATH = REPOSITORY_ROOT / "docs" / "REVIEW_REMEDIATION.md"
RELATED_WORK_PATH = REPOSITORY_ROOT / "docs" / "RELATED_WORK.md"
REFERENCES_PATH = REPOSITORY_ROOT / "docs" / "REFERENCES.md"


EXPECTED_VERDICTS = {
    "R01": "Partially valid",
    "R02": "Valid",
    "R03": "Valid",
    "R04": "Valid",
    "R05": "Valid",
    "R06": "Valid limitation",
    "R07": "Valid",
    "R08": "Valid",
    "R09": "Valid",
    "R10": "Scientifically useful but excluded by the approved scope",
    "R11": "Strongly valid",
    "R12": "Valid limitation",
    "R13": "Valid limitation",
    "R14": "Valid limitation",
}


def _review_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^## (R\d{2})\b", text, flags=re.MULTILINE))
    return {
        match.group(1): text[
            match.start() : matches[index + 1].start() if index + 1 < len(matches) else None
        ]
        for index, match in enumerate(matches)
    }


def test_review_ledger_disposes_every_substantive_criticism() -> None:
    text = REMEDIATION_PATH.read_text(encoding="utf-8")
    sections = _review_sections(text)

    assert set(sections) == set(EXPECTED_VERDICTS)
    assert len(re.findall(r"^## R\d{2}\b", text, flags=re.MULTILINE)) == 14
    for review_id, verdict in EXPECTED_VERDICTS.items():
        section = sections[review_id]
        assert f"**Verdict:** {verdict}" in section
        assert "**Criticism:**" in section
        assert "**Action:**" in section
        assert "**Acceptance evidence:**" in section


def test_review_ledger_preserves_scope_decisions_and_rq3_response() -> None:
    text = REMEDIATION_PATH.read_text(encoding="utf-8")
    sections = _review_sections(text)

    assert "1624389a763f283cae828a7bf5f0922b2acb724b8b687c88f19ad9c869cb3701" in text
    assert "Critical Only" in sections["R01"]
    assert "supply-blind" in sections["R01"]
    assert "Training-objective score (diagnostic)" in sections["R02"]
    assert "GPU acceleration" in sections["R03"]
    assert "What types of data and rules are needed" in sections["R05"]
    assert "±25%" in sections["R05"]
    assert "feeder" in sections["R05"].casefold()
    assert "proportional" in sections["R06"].casefold()
    assert "81001–81010" in sections["R08"]
    assert "50001–50003" in sections["R08"]
    assert "one-hour persistence" in sections["R09"]
    assert "training-provenance.yaml" in sections["R11"]
    assert "global optimum" in sections["R12"]
    assert "3^N" in sections["R13"]
    assert "shortage-only" in sections["R14"]


def test_related_work_is_domain_grounded_and_claim_bounded() -> None:
    text = RELATED_WORK_PATH.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert 900 <= len(text.split()) <= 1_600
    assert all(f"[{index}]" in text for index in range(9, 16))
    assert "Khartoum-inspired synthetic case" in text
    assert "not calibrated" in text
    assert "does not establish" in normalized
    assert "MAE" in text and "RMSE" in text
    assert "Fixed Rotation" in text and "PPO" in text

    forbidden_claims = (
        "calibrated Khartoum grid model",
        "digital twin",
        "real-time grid controller",
        "globally novel",
        "first system to",
        "proves PPO is superior",
        "guarantees critical service",
    )
    assert not any(claim.casefold() in normalized.casefold() for claim in forbidden_claims)


def test_reference_list_adds_exactly_seven_vetted_domain_sources() -> None:
    text = REFERENCES_PATH.read_text(encoding="utf-8")
    entries = re.findall(r"^(\d+)\. ", text, flags=re.MULTILINE)

    assert entries == [str(index) for index in range(1, 16)]
    for identifier in (
        "P507788",
        "10.1016/j.enpol.2018.06.025",
        "10.1007/s10458-019-09428-8",
        "10.1109/TPWRS.2018.2860256",
        "10.1016/j.apenergy.2018.11.002",
        "10.1016/j.arcontrol.2019.09.008",
        "10.1186/s43067-020-00021-8",
    ):
        assert identifier in text


def test_new_review_documents_are_clean_utf8_english() -> None:
    corpus = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (REMEDIATION_PATH, RELATED_WORK_PATH, REFERENCES_PATH)
    )

    assert "\ufffd" not in corpus
    assert "Ã" not in corpus
    assert "Ø" not in corpus
    assert "Ù" not in corpus
