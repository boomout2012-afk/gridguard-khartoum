from __future__ import annotations

import re
import zipfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPOSITORY_ROOT / "tools" / "build_presentation.ps1"
DECK_PATH = REPOSITORY_ROOT / "output" / "presentation" / "gridguard-khartoum-defense.pptx"


def test_presentation_builder_is_evidence_bound_and_uses_native_powerpoint() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "PowerPoint.Application" in source
    assert "Get-FileHash" in source
    assert "ConvertFrom-Json" in source
    assert "$evidence.aggregate" in source
    assert "$evidence.paired_inference" in source
    assert "2c3e71e48c4c7bff4fe609d42a76ca4f770f6dd717cd7d45d5e8e0d99fdf2d11" in source
    assert "GPU-first simulation, PPO inference, and paired bootstrap" in source
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
    assert source.count("Add-Notes") >= 13  # Function plus one call for every slide.


def test_final_deck_has_twelve_slides_notes_and_no_forbidden_controller_name() -> None:
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

    assert len(slide_names) == 12
    assert len(note_names) == 12
    assert xml_text.count("[Sources]") == 12
    acronym = "".join(("B", "C"))
    forbidden_patterns = (
        re.escape("behavior " + "cloning"),
        rf"\b{acronym}\s*\+\s*PPO\b",
        rf"\b{acronym}\b",
        re.escape("Smart " + "AI"),
    )
    assert not any(
        re.search(pattern, xml_text, flags=re.IGNORECASE) for pattern in forbidden_patterns
    )
