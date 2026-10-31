from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage

from gridguard.domain import SECTORS, ControllerName, ServiceTier
from gridguard.ui.live_models import (
    LiveEpisodeSummary,
    LiveScenarioComparison,
    LiveScenarioFrame,
    LocalityHourDetail,
)
from gridguard.ui.live_runner import run_live_scenario
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_TOOL = REPOSITORY_ROOT / "tools" / "capture_screenshots.py"
COMMITTED_SCREENSHOT_DIR = REPOSITORY_ROOT / "docs" / "assets" / "app-regression-screenshots"


def _region_digest(image: QImage, region: QRect) -> str:
    cropped = image.copy(region).convertToFormat(QImage.Format.Format_RGBA8888)
    return hashlib.sha256(cropped.constBits().tobytes()).hexdigest()


def _load_capture_tool():
    spec = importlib.util.spec_from_file_location("gridguard_capture_screenshots", CAPTURE_TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _portable_comparison() -> LiveScenarioComparison:
    """Minimal display data injected only into portable rendering tests."""

    frames = tuple(
        LiveScenarioFrame(
            hour=hour,
            service_tiers={locality: ServiceTier.OFF for locality in EXPECTED_LOCALITY_IDS},
            demand_mw=70.0,
            supply_mw=35.0,
            locality_details={
                locality: LocalityHourDetail(
                    demand_mw=10.0,
                    served_mw=0.0,
                    critical_service_ratio=0.0,
                    sector_service_ratio={sector.value: 0.0 for sector in SECTORS},
                    next_hour_projected_demand_mw=10.0 if 2 <= hour <= 22 else None,
                )
                for locality in EXPECTED_LOCALITY_IDS
            },
        )
        for hour in range(24)
    )
    summary = LiveEpisodeSummary(
        served_mwh=0.0,
        critical_service_ratio=0.0,
        fairness=0.0,
        curtailed_mwh=1680.0,
        power_balance_violation_rate=0.0,
    )
    return LiveScenarioComparison(
        shortage_level="moderate",
        seed=50002,
        device="cuda:0",
        frames_by_controller={controller.value: frames for controller in ControllerName},
        metrics_by_controller={controller.value: summary for controller in ControllerName},
    )


def test_live_capture_uses_the_canonical_runner_without_changing_its_result(monkeypatch) -> None:
    capture = _load_capture_tool()
    comparison = _portable_comparison()
    calls = []

    def runner(shortage_level, seed):
        calls.append((shortage_level, seed))
        return comparison

    monkeypatch.setattr(capture, "run_live_scenario", runner, raising=False)

    assert capture.build_live_capture_comparison() is comparison
    assert calls == [("moderate", 50002)]


@pytest.mark.gpu
def test_live_capture_matches_actual_cuda_simulation_and_causal_projection() -> None:
    capture = _load_capture_tool()

    actual = capture.build_live_capture_comparison()
    expected = run_live_scenario("moderate", 50002)

    assert actual == expected


def test_capture_page_selection_aligns_navigation_focus(qapp, monkeypatch) -> None:
    capture = _load_capture_tool()
    monkeypatch.setattr(capture, "build_live_capture_comparison", _portable_comparison)
    window = capture._prepare_window(qapp)
    window.show()
    qapp.processEvents()

    try:
        for page_index in range(3):
            capture._select_capture_page(window, page_index)
            qapp.processEvents()

            assert window.tabs.currentIndex() == page_index
            assert window.navigation_buttons[page_index].hasFocus()
    finally:
        window.close()
        qapp.processEvents()


def test_capture_writes_three_distinct_1366_by_768_pngs(
    qapp,
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture = _load_capture_tool()
    prepared_windows = []
    installed_comparisons = []
    comparison = _portable_comparison()
    runner_calls = []
    prepare_window = capture._prepare_window

    def run_comparison():
        runner_calls.append(True)
        return comparison

    def record_prepared_window(application, *, comparison=None):
        installed_comparisons.append(comparison)
        window = prepare_window(application, comparison=comparison)
        prepared_windows.append(window)
        return window

    monkeypatch.setattr(capture, "build_live_capture_comparison", run_comparison)
    monkeypatch.setattr(capture, "_prepare_window", record_prepared_window)

    paths = capture.capture_screenshots(tmp_path, application=qapp)

    assert len(prepared_windows) == 3
    assert len(runner_calls) == 1
    assert all(value is comparison for value in installed_comparisons)
    assert all(not window.isVisible() for window in prepared_windows)
    assert tuple(path.name for path in paths) == (
        "overview.png",
        "live.png",
        "results.png",
    )
    digests: set[str] = set()
    wordmark_digests: set[str] = set()
    for path in paths:
        assert path.stat().st_size > 20_000
        image = QImage(str(path))
        assert not image.isNull()
        assert (image.width(), image.height()) == (1366, 768)
        digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
        wordmark_digests.add(_region_digest(image, QRect(52, 40, 250, 40)))
    assert len(digests) == 3
    assert len(wordmark_digests) == 1


@pytest.mark.gpu
def test_cli_capture_matches_committed_cuda_screenshots(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(CAPTURE_TOOL), "--output-dir", str(tmp_path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    paths = sorted(tmp_path.glob("*.png"))
    assert len(paths) == 3
    for path in paths:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = hashlib.sha256((COMMITTED_SCREENSHOT_DIR / path.name).read_bytes()).hexdigest()
        assert actual == expected, path.name
    header_digests = {_region_digest(QImage(str(path)), QRect(52, 40, 250, 40)) for path in paths}
    assert len(header_digests) == 1
