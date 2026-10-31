"""Capture actual Qt/CUDA application pages and verify repeatable rendered evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZE = (1366, 768)
DEFAULT_OUTPUT = ROOT / "docs/assets/current-app-screenshots"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_captures(first: dict, second: dict) -> None:
    """Require identical policy, source identities and pixels across independent runs."""
    for key in ("policy", "sources"):
        if first[key] != second[key]:
            raise ValueError(f"Repeated capture changed {key}")
    maps = []
    for manifest in (first, second):
        pages = manifest["pages"]
        indexed = {page["filename"]: (page["sha256"], page["size"]) for page in pages}
        if len(indexed) != len(pages):
            raise ValueError("Duplicate capture filename")
        maps.append(indexed)
    if maps[0] != maps[1]:
        changed = sorted(
            key for key in maps[0].keys() | maps[1].keys() if maps[0].get(key) != maps[1].get(key)
        )
        raise ValueError("Repeated capture changed pages: " + ", ".join(changed))


def capture_application(output: Path) -> dict:
    """Run genuine matched inference and capture every documented page at 1366 by 768."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QLabel, QTableWidget

    from gridguard.ui.live_runner import run_live_scenario
    from gridguard.ui.presentation_model import load_presentation_snapshot
    from gridguard.ui.theme import apply_theme
    from gridguard.ui.window import PresentationWindow

    output.mkdir(parents=True, exist_ok=True)
    if QApplication.instance() is not None:
        raise RuntimeError("Run the capture CLI in its own process")
    app = QApplication([])
    apply_theme(app)
    window = PresentationWindow(load_presentation_snapshot(), runtime_label="Verified CUDA")
    pages = []

    def capture(widget, filename: str, title: str, **state) -> None:
        widget.resize(*SIZE)
        widget.show()
        app.processEvents()
        QTest.qWait(100)
        widget.grab()
        app.processEvents()
        QTest.qWait(200)
        pixmap = widget.grab()
        if (pixmap.width(), pixmap.height()) != SIZE:
            raise ValueError(f"Unexpected capture dimensions: {filename}")
        path = output / filename
        if not pixmap.save(str(path), "PNG"):
            raise RuntimeError(f"Could not write capture: {filename}")
        labels = [label.text() for label in widget.findChildren(QLabel) if label.isVisible()]
        removed = (
            "met the service-improvement criterion",
            "The full optimizer comparison did not pass",
        )
        if any(phrase in label for phrase in removed for label in labels):
            raise ValueError("A user-removed app sentence has returned")
        tables = [
            {
                "name": table.accessibleName(),
                "rows": table.rowCount(),
                "columns": table.columnCount(),
                "horizontal_scroll_maximum": table.horizontalScrollBar().maximum(),
                "vertical_scroll_maximum": table.verticalScrollBar().maximum(),
            }
            for table in widget.findChildren(QTableWidget)
            if table.isVisible()
        ]
        pages.append(
            {
                "page": len(pages) + 1,
                "title": title,
                "filename": filename,
                "sha256": digest(path),
                "size": list(SIZE),
                "state": state,
                "tables": tables,
            }
        )
        print(f"Captured {len(pages):02d}: {filename}", flush=True)

    try:
        window.resize(*SIZE)
        window.show()
        QTest.qWait(100)
        window.tabs.setCurrentIndex(0)
        capture(window, "overview.png", "Overview")
        window.tabs.setCurrentIndex(1)
        capture(window, "live-ready.png", "Live Scenario · ready to run")
        comparison = run_live_scenario("moderate", 501)
        window.live_page.set_comparison(comparison)
        for filename, controller, title in (
            ("live-ppo.png", "PPO", "Live Scenario · PPO"),
            ("live-fixed.png", "Fixed Rotation", "Live Scenario · Fixed Rotation"),
            ("live-optimizer.png", "Optimization", "Live Scenario · constrained optimizer"),
        ):
            window.live_page.controller_combo.setCurrentText(controller)
            window.live_page.timeline.setValue(12)
            window.live_page.map_view.select_locality("east_nile")
            capture(
                window,
                filename,
                title,
                shortage="Moderate",
                scenario_seed=501,
                hour=12,
                controller=controller,
                locality="East Nile",
            )
        window.tabs.setCurrentIndex(2)
        current = window.results_page
        current.results_tabs.setCurrentIndex(0)
        capture(window, "results.png", "Results · current five-run study")
        current.results_tabs.setCurrentIndex(1)
        followup = current.followup_page
        if not hasattr(followup, "detail_tabs"):
            raise ValueError("Follow-up evidence failed to load")
        for index, filename in enumerate(
            ("followup-controllers.png", "followup-paired.png", "followup-stress.png")
        ):
            followup.detail_tabs.setCurrentIndex(index)
            capture(
                window,
                filename,
                "Follow-up · " + followup.detail_tabs.tabText(index),
                tab=index,
                study="post-hoc matched audit",
            )
        current.results_tabs.setCurrentIndex(0)
        current.details_button.click()
        details = current.details_dialog
        for index, name in enumerate(
            (
                "current-paired-fixed",
                "current-paired-optimizer",
                "current-individual-runs",
                "current-shortage-levels",
                "current-evidence",
            )
        ):
            details.tabs.setCurrentIndex(index)
            capture(
                details, name + ".png", "Current study · " + details.tabs.tabText(index), tab=index
            )
        details.reject()
        current.earlier_button.click()
        capture(current.historical_dialog, "earlier-results.png", "Earlier frozen results")
        historical = current.historical_page
        historical.evidence_details_button.click()
        evidence = historical.evidence_dialog
        evidence.resize(*SIZE)
        evidence.show()
        app.processEvents()
        QTest.qWait(100)
        scroll = evidence.scroll_area.verticalScrollBar()
        positions = [0]
        while positions[-1] < scroll.maximum():
            positions.append(
                min(scroll.maximum(), positions[-1] + evidence.scroll_area.viewport().height() - 70)
            )
        for index, position in enumerate(positions, 1):
            scroll.setValue(position)
            capture(
                evidence,
                f"earlier-evidence-{index:02d}.png",
                f"Earlier evidence · section {index} of {len(positions)}",
                scroll_offset=position,
                scroll_maximum=scroll.maximum(),
            )
        evidence.reject()
        historical.extended_study_button.click()
        extended = historical.proposal_dialog
        if extended is None:
            raise RuntimeError("Extended study dialog did not open")
        extended.tabs.setCurrentIndex(0)
        capture(extended, "extended-controllers.png", "Extended study · controller outcomes")
        extended.tabs.setCurrentIndex(1)
        for index in range(extended.pair_selector.count()):
            extended.pair_selector.setCurrentIndex(index)
            pair = extended.pair_selector.currentText()
            capture(
                extended,
                f"extended-paired-{index + 1:02d}.png",
                "Extended study · " + pair,
                pair=pair,
            )
        extended.tabs.setCurrentIndex(2)
        capture(extended, "extended-forecast.png", "Extended study · forecasting")
        extended.reject()
        current.historical_dialog.reject()
        source_paths = sorted(
            path.relative_to(ROOT).as_posix() for path in (ROOT / "src/gridguard").rglob("*.py")
        )
        source_paths += [
            "data/evidence/focused-ppo-study.json",
            "data/evidence/focused-ppo-training.json",
            "data/evidence/committee-baseline-audit.json",
            "data/evidence/committee-baseline-audit.validation.json",
            comparison.policy_identity.model_relative_path,
        ]
        manifest = {
            "schema": "gridguard.current_app_capture.1",
            "captured_utc": datetime.now(UTC).isoformat(),
            "capture_method": "Actual PySide6 QWidget.grab; Qt offscreen platform, scale factor 1",
            "capture_size": list(SIZE),
            "cuda_device": torch.cuda.get_device_name(0),
            "policy": asdict(comparison.policy_identity),
            "policy_caption": window.live_page.scope_note.text(),
            "live_execution": {
                "shortage": "moderate",
                "scenario_seed": 501,
                "hour": 12,
                "controllers": list(comparison.frames_by_controller),
                "genuine_cuda_execution": True,
            },
            "sources": {path: digest(ROOT / path) for path in source_paths},
            "capture_script": {
                "path": Path(__file__).relative_to(ROOT).as_posix(),
                "sha256": digest(Path(__file__)),
            },
            "coverage": {
                "main_navigation_pages": ["Overview", "Live Scenario", "Results"],
                "live_states": "Ready plus three controllers; Moderate/501, noon, East Nile",
                "current_study_tabs": [
                    details.tabs.tabText(i) for i in range(details.tabs.count())
                ],
                "followup_tabs": [
                    followup.detail_tabs.tabText(i) for i in range(followup.detail_tabs.count())
                ],
                "earlier_results": "Earlier results and every evidence scroll section",
                "extended_study": "Controllers, all paired-comparison choices and Forecasting",
                "excluded": "Transient messages, menus, tooltips and every parameter combination",
                "dialog_size": "Resizable dialogs expanded to 1366 x 768",
            },
            "pages": pages,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return manifest
    finally:
        app.closeAllWindows()
        app.processEvents()
        app.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compare-to", type=Path, help="Directory with a prior capture manifest")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = capture_application(output)
    if args.compare_to:
        comparison = args.compare_to if args.compare_to.is_absolute() else ROOT / args.compare_to
        compare_captures(
            json.loads((comparison / "manifest.json").read_text(encoding="utf-8")), manifest
        )
        print("Independent captures have identical policy, source identities and page pixels.")
    print(json.dumps({"pages": len(manifest["pages"]), "manifest": str(output / "manifest.json")}))


if __name__ == "__main__":
    main()
