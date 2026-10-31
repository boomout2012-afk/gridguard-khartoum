"""Check the published research entry points against their linked evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINTS = ("README.md", "SUBMISSION_README.md", "docs/REVIEWER_GUIDE.md")
STUDY_PATH = "data/evidence/focused-ppo-study.json"
STUDY_SHA256 = "a4d5a849612f06eedef96e0d28e08247e5e3757eba0b4f2e4f1a893e9d4b8247"


def _linked_local_paths(document: Path) -> set[Path]:
    text = document.read_text(encoding="utf-8")
    targets = re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", text)
    targets.extend(re.findall(r'<img\b[^>]*\bsrc=["\']([^"\']+)["\']', text))
    local = set()
    for target in targets:
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        path = (document.parent / unquote(parsed.path)).resolve()
        assert path.is_relative_to(ROOT), (document, target)
        assert path.is_file(), (document, target)
        local.add(path)
    return local


@pytest.mark.parametrize("relative", ENTRY_POINTS)
def test_public_document_links_and_images_resolve(relative: str) -> None:
    _linked_local_paths(ROOT / relative)


def test_readme_links_the_current_academic_outputs() -> None:
    paths = _linked_local_paths(ROOT / "README.md")
    for relative in (
        "output/pdf/updated/gridguard-khartoum-thesis.pdf",
        "output/pdf/updated/gridguard-khartoum-project-presentation.pdf",
        "output/pdf/gridguard-khartoum-gui-pages.pdf",
    ):
        assert (ROOT / relative).resolve() in paths


def _readme_result_table() -> dict[str, dict[str, str]]:
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        headings = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not headings or headings[0] != "Controller":
            continue
        rows = {}
        for row in lines[index + 2 :]:
            if not row.startswith("|"):
                break
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            assert len(cells) == len(headings), row
            assert cells[0] not in rows, cells[0]
            rows[cells[0]] = dict(zip(headings, cells, strict=True))
        return rows
    raise AssertionError("No controller results table in the research overview")


@pytest.mark.parametrize("controller", ("Fixed Rotation", "Constrained Optimization", "PPO"))
def test_readme_result_values_match_the_original_matched_study(controller: str) -> None:
    raw = (ROOT / STUDY_PATH).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == STUDY_SHA256
    study = json.loads(raw)
    assert study["matched_scenario_count"] == 960
    row = _readme_result_table()[controller]
    source = study["procedure_means" if controller == "PPO" else "controller_means"][controller]
    columns = {
        "Served energy (MWh/day)": ("served_mwh", 1),
        "Critical interruption (locality-h)": ("critical_interruption_locality_hours", 1),
        "Jain fairness": ("fairness", 100),
        "Utilization": ("utilization", 100),
    }
    for heading, (metric, scale) in columns.items():
        displayed = float(row[heading].replace(",", "").removesuffix("%"))
        assert displayed == pytest.approx(source[metric] * scale, abs=0.005), heading


def test_research_guide_links_separate_evidence_for_the_three_experiments() -> None:
    paths = _linked_local_paths(ROOT / "docs/REVIEWER_GUIDE.md")
    for relative in (
        STUDY_PATH,
        "data/evidence/committee-baseline-audit.json",
        "data/evidence/thesis-learning-audit.json",
    ):
        assert (ROOT / relative).resolve() in paths
