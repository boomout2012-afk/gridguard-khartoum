from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QRect
from PySide6.QtGui import QImage

from gridguard.domain import ServiceTier

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_TOOL = REPOSITORY_ROOT / "tools" / "capture_screenshots.py"


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


def test_live_capture_fixture_is_matched_deterministic_and_visually_varied() -> None:
    capture = _load_capture_tool()

    first = capture.build_live_capture_comparison()
    second = capture.build_live_capture_comparison()

    assert first == second
    assert first.shortage_level == "moderate"
    assert first.seed == 50002
    assert first.device == "cuda:0"
    assert set(first.frames("PPO")[18].service_tiers.values()) == {
        ServiceTier.OFF,
        ServiceTier.CRITICAL_ONLY,
        ServiceTier.FULL_SERVICE,
    }
    for ppo_frame, fixed_frame in zip(
        first.frames("PPO"),
        first.frames("Fixed Rotation"),
        strict=True,
    ):
        assert ppo_frame.demand_mw == fixed_frame.demand_mw
        assert ppo_frame.supply_mw == fixed_frame.supply_mw
    assert first.metrics("PPO").served_mwh > first.metrics("Fixed Rotation").served_mwh
    assert (
        first.metrics("PPO").critical_service_ratio
        > first.metrics("Fixed Rotation").critical_service_ratio
    )
    assert first.metrics("PPO").power_balance_violation_rate == 0.0
    assert first.metrics("Fixed Rotation").power_balance_violation_rate == 0.0


def test_capture_page_selection_aligns_navigation_focus(qapp) -> None:
    capture = _load_capture_tool()
    window = capture._prepare_window(qapp)
    window.show()
    qapp.processEvents()

    try:
        for page_index in range(4):
            capture._select_capture_page(window, page_index)
            qapp.processEvents()

            assert window.tabs.currentIndex() == page_index
            assert window.navigation_buttons[page_index].hasFocus()
    finally:
        window.close()
        qapp.processEvents()


def test_capture_writes_four_distinct_1366_by_768_pngs(
    qapp,
    tmp_path: Path,
    monkeypatch,
) -> None:
    capture = _load_capture_tool()
    prepared_windows = []
    prepare_window = capture._prepare_window

    def record_prepared_window(application):
        window = prepare_window(application)
        prepared_windows.append(window)
        return window

    monkeypatch.setattr(capture, "_prepare_window", record_prepared_window)

    paths = capture.capture_screenshots(tmp_path, application=qapp)

    assert len(prepared_windows) == 4
    assert all(not window.isVisible() for window in prepared_windows)
    assert tuple(path.name for path in paths) == (
        "overview.png",
        "live.png",
        "results.png",
        "methodology.png",
    )
    digests: set[str] = set()
    rail_header_digests: set[str] = set()
    trust_strip_digests: set[str] = set()
    for path in paths:
        assert path.stat().st_size > 20_000
        image = QImage(str(path))
        assert not image.isNull()
        assert (image.width(), image.height()) == (1366, 768)
        digests.add(hashlib.sha256(path.read_bytes()).hexdigest())
        rail_header_digests.add(_region_digest(image, QRect(0, 0, 176, 125)))
        trust_strip_digests.add(_region_digest(image, QRect(0, 727, 1366, 41)))
    assert len(digests) == 4
    assert len(rail_header_digests) == 1
    assert len(trust_strip_digests) == 1


def test_cli_capture_paints_the_complete_navigation_header_consistently(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(CAPTURE_TOOL), "--output-dir", str(tmp_path)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    header_digests = {
        _region_digest(QImage(str(path)), QRect(0, 0, 176, 125))
        for path in sorted(tmp_path.glob("*.png"))
    }
    assert len(header_digests) == 1
