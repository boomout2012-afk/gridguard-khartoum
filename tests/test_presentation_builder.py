from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPOSITORY_ROOT / "tools" / "build_presentation.ps1"
DECK_PATH = REPOSITORY_ROOT / "output" / "presentation" / "gridguard-khartoum-defense.pptx"
PAIRED_PATH = REPOSITORY_ROOT / "data" / "evidence" / "paired-benchmark.json"
ROBUSTNESS_PATH = REPOSITORY_ROOT / "data" / "evidence" / "robustness-supplement.json"
PRESENTATION_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"p": PRESENTATION_NS, "a": DRAWING_NS}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_presentation_builder_is_evidence_bound_and_uses_native_powerpoint() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "PowerPoint.Application" in source
    assert "Get-FileHash" in source
    assert "ConvertFrom-Json" in source
    assert "$pairedEvidence.aggregate" in source
    assert "$pairedEvidence.paired_inference" in source
    assert "$robustnessEvidence.base_evaluation" in source
    assert _sha256(PAIRED_PATH) in source
    assert _sha256(ROBUSTNESS_PATH) in source
    assert "training-provenance.yaml" in source
    assert "--summary-json" in source
    assert "GPU-first simulation, PPO inference, and paired bootstrap" in source
    assert "$allPrimaryCellsFavorable" not in source
    assert "all-primary-cells-favorable" not in source
    assert "Every release calculation stays on the NVIDIA GPU" not in source
    assert "99.72%" not in source
    assert "42.63%" not in source
    assert "+2,008 MWh" not in source
    assert "docs\\assets\\screenshots\\overview.png" in source
    assert "docs\\assets\\screenshots\\live.png" in source
    assert "schematic" not in source.casefold()
    assert "administrative locality boundaries" in source
    assert '"GRIDGUARD KHARTOUM" 48 18 300 22 16' in source
    assert '"OFFLINE GPU RESEARCH SIMULATION" 48 503 320 22 16' in source
    assert '"GRADUATION PROJECT DEFENSE" 60 58 420 24 16' in source
    assert "Assert-PresentationLayout" in source
    assert "BoundHeight" in source
    assert source.count("Add-Notes") >= 13  # Function plus one call for every slide.


def test_final_deck_has_thirteen_slides_notes_and_no_forbidden_controller_name() -> None:
    assert DECK_PATH.is_file()

    with zipfile.ZipFile(DECK_PATH) as archive:
        names = archive.namelist()
        slide_names = [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
        note_names = [
            name for name in names if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)
        ]
        xml_text = "\n".join(
            archive.read(name).decode("utf-8", errors="replace")
            for name in slide_names + note_names
        )

    assert len(slide_names) == 13
    assert len(note_names) == 13
    assert xml_text.count("[Sources]") == 13
    acronym = "".join(("B", "C"))
    forbidden_patterns = (
        r"\bbehavior\s+cloning\s*\+\s*PPO\b",
        rf"\b{acronym}\s*\+\s*PPO\b",
        re.escape("Smart " + "AI"),
    )
    assert not any(
        re.search(pattern, xml_text, flags=re.IGNORECASE) for pattern in forbidden_patterns
    )
    for phrase in (
        "Training-objective score (diagnostic)",
        "critical-outage locality-hours",
        "320 independent base draws",
        "25 parameterizations",
        "one-hour persistence",
        "accepted reproduction",
        "shortage-only",
    ):
        assert phrase.casefold() in xml_text.casefold()


def test_final_deck_follows_the_locked_cumulative_narrative() -> None:
    expected_slide_phrases = (
        "GridGuard Khartoum",
        "seven simultaneous allocation choices",
        "controller evidence",
        "same scenario",
        "historical training record",
        "GPU-first simulation",
        "desktop replays",
        "untouched confirmatory evidence",
        "24 demand-allocation variants",
        "one-hour persistence",
        "accepted reproduction",
        "shortage-only",
        "locked synthetic evaluation protocol",
    )
    with zipfile.ZipFile(DECK_PATH) as archive:
        for slide_number, phrase in enumerate(expected_slide_phrases, start=1):
            xml = archive.read(f"ppt/slides/slide{slide_number}.xml").decode(
                "utf-8", errors="replace"
            )
            assert phrase.casefold() in xml.casefold(), (slide_number, phrase)

        outcome_xml = archive.read("ppt/slides/slide8.xml").decode("utf-8", errors="replace")
    for phrase in (
        "critical-outage locality-hours",
        "Jain fairness",
        "Utilization",
        "power-balance violation",
    ):
        assert phrase.casefold() in outcome_xml.casefold()
    for forbidden in ("episode reward", "served energy", "critical service"):
        assert forbidden.casefold() not in outcome_xml.casefold()


def test_final_deck_has_legible_text_closed_placeholders_and_in_bounds_shapes() -> None:
    with zipfile.ZipFile(DECK_PATH) as archive:
        presentation = ElementTree.fromstring(archive.read("ppt/presentation.xml"))
        slide_size = presentation.find("p:sldSz", NS)
        assert slide_size is not None
        slide_width = int(slide_size.attrib["cx"])
        slide_height = int(slide_size.attrib["cy"])

        for slide_number in range(1, 14):
            root = ElementTree.fromstring(archive.read(f"ppt/slides/slide{slide_number}.xml"))
            assert root.find(".//p:ph", NS) is None

            font_sizes = [
                int(element.attrib["sz"])
                for element in root.iter()
                if element.tag.rsplit("}", 1)[-1] in {"rPr", "defRPr", "endParaRPr"}
                and "sz" in element.attrib
            ]
            assert font_sizes
            assert min(font_sizes) >= 1_600, slide_number

            for transform in root.findall(".//a:xfrm", NS):
                offset = transform.find("a:off", NS)
                extent = transform.find("a:ext", NS)
                if offset is None or extent is None:
                    continue
                left = int(offset.attrib["x"])
                top = int(offset.attrib["y"])
                width = int(extent.attrib["cx"])
                height = int(extent.attrib["cy"])
                assert left >= 0 and top >= 0, slide_number
                assert left + width <= slide_width + 1_000, slide_number
                assert top + height <= slide_height + 1_000, slide_number
