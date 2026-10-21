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
ATTRIBUTION_BOUNDARY = (
    "No behavior-cloning-only ablation was run, so this comparison cannot isolate the "
    "incremental effect of PPO fine-tuning."
)


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
    assert "$robustnessEvidence.schema.version -ne 2" in source
    assert "documented_seed_disjoint_holdout" in source
    assert "gridguard-robustness-supplement-v2" in source
    assert _sha256(PAIRED_PATH) in source
    assert _sha256(ROBUSTNESS_PATH) in source
    assert "training-provenance.yaml" in source
    assert "--summary-json" in source
    assert "GPU-first simulation, PPO inference, and paired bootstrap" in source
    assert "Off and Full Service" in source
    assert "Off/Full-only" not in source
    assert "Off/Full Service" not in source
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
    assert "not preregistered or independently custodied" in source
    assert "unadjusted for multiple comparisons" in source
    assert "No behavior-cloning-only ablation" in source
    assert "cannot select Critical Only" in source
    assert "same guarded scenario" in source
    assert "detected demand/supply mutation fails closed" in source
    assert "same immutable scenario" not in source
    assert source.count("Add-Notes") >= 13  # Function plus one call for every slide.


def test_presentation_protocol_and_training_numbers_are_projected_from_verified_inputs() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    for projection in (
        "$holdoutSeedCount = @($robustnessProtocol.seeds).Count",
        "$episodesPerHoldoutSeed = [int]$robustnessProtocol.episodes_per_seed",
        "$bootstrapReplicates = [int]$robustnessProtocol.bootstrap.replicates",
        "$acceptedSeedLabel = Format-SeedRange",
        "$holdoutSeedLabel = Format-SeedRange",
        "$selectedIteration = [int]$trainingSummary.selection.selected_iteration",
        "$selectedTrainingSeed = [int]$trainingSummary.experiment_scope.selected_run.training_seed",
        "$candidateRunCount = [int]$trainingSummary.experiment_scope."
        "candidate_experiment.candidate_run_count",
        "$candidateIterations = [int]$candidateWorkload.ppo_iterations",
        "$candidateEpisodes = [int]$candidateWorkload.complete_episodes",
        "$candidateTransitions = [int]$candidateWorkload.transitions",
        '$selectedConvergenceProperty = "selected_iteration_$selectedIteration"',
        '"x $shortageCount"',
        '"x $controllerCount"',
        '"Seeds $acceptedSeedLabel  |  $acceptedBaseDrawCount base draws"',
        '"$holdoutSeedCount seeds x $episodesPerHoldoutSeed episodes"',
        '"Documented-seed-disjoint holdout"',
        '"$holdoutSeedLabel remains separate"',
        '"Selected seed $selectedTrainingSeed"',
        '"Three-seed candidate experiment"',
    ):
        assert projection in source

    for independent_literal in (
        "10 seeds x 32 episodes",
        "selected iteration 510",
        "10,000 deterministic seed-stratified bootstrap replicates",
        "Seeds 50001-50003  |  96 base draws",
        "81001-81010 remains separate",
    ):
        assert independent_literal not in source


def test_forecast_slide_builder_labels_error_metrics_in_mw() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")
    forecast_slide = source.split("# 10 - forecast", maxsplit=1)[1].split(
        "# 11 - accepted reproduction", maxsplit=1
    )[0]

    assert '"Pooled error metric (MW)"' in forecast_slide


def test_final_forecast_slide_labels_error_metrics_in_mw() -> None:
    with zipfile.ZipFile(DECK_PATH) as archive:
        root = ElementTree.fromstring(archive.read("ppt/slides/slide10.xml"))
    labels = [element.text for element in root.iter(f"{{{DRAWING_NS}}}t")]

    assert "Pooled error metric (MW)" in labels


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
        "documented-seed-disjoint holdout",
        "not preregistered or independently custodied",
        "pointwise 95%",
        "unadjusted for multiple comparisons",
        "No behavior-cloning-only ablation",
        "cannot select Critical Only",
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
        "documented-seed-disjoint holdout",
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


def test_slide_eight_places_the_attribution_boundary_beside_primary_results() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")
    slide_eight_source = source.split("# 8 - documented-seed-disjoint outcomes", maxsplit=1)[
        1
    ].split("# 9 - sensitivity", maxsplit=1)[0]

    assert slide_eight_source.count(ATTRIBUTION_BOUNDARY) == 2

    with zipfile.ZipFile(DECK_PATH) as archive:
        outcome_xml = archive.read("ppt/slides/slide8.xml").decode("utf-8", errors="replace")
        outcome_notes_xml = archive.read("ppt/notesSlides/notesSlide8.xml").decode(
            "utf-8", errors="replace"
        )

    assert ATTRIBUTION_BOUNDARY.casefold() in outcome_xml.casefold()
    assert ATTRIBUTION_BOUNDARY.casefold() in outcome_notes_xml.casefold()


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
