"""Capture deterministic 1366x768 screenshots from the real offline desktop UI."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

# These settings must be fixed before Qt is imported so host display scaling cannot change pixels.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_SCALE_FACTOR", "1")
os.environ.setdefault("QT_FONT_DPI", "96")

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication

from gridguard.domain import SECTOR_SHARES, ControllerName, ServiceTier
from gridguard.ui.live_models import (
    LiveEpisodeSummary,
    LiveScenarioComparison,
    LiveScenarioFrame,
    LocalityHourDetail,
)
from gridguard.ui.map_asset import EXPECTED_LOCALITY_IDS, load_map_asset
from gridguard.ui.presentation_model import load_presentation_snapshot
from gridguard.ui.theme import apply_theme
from gridguard.ui.window import PresentationWindow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "assets" / "screenshots"
CAPTURE_SIZE = QSize(1366, 768)
CAPTURE_HOUR = 18
CAPTURE_LOCALITY_ID = "east_nile"
PAGE_CAPTURES = (
    ("overview", 0),
    ("live", 1),
    ("results", 2),
    ("methodology", 3),
)

_LOCALITY_BASE_DEMAND_MW = (82.0, 78.0, 69.0, 64.0, 56.0, 59.0, 73.0)
_PPO_TIER_SEQUENCE = (
    ServiceTier.FULL_SERVICE,
    ServiceTier.CRITICAL_ONLY,
    ServiceTier.FULL_SERVICE,
    ServiceTier.CRITICAL_ONLY,
    ServiceTier.OFF,
    ServiceTier.CRITICAL_ONLY,
    ServiceTier.FULL_SERVICE,
)
_TIER_SERVICE_RATIOS = {
    ServiceTier.OFF: (0.0, 0.0, 0.0, 0.0, 0.0),
    ServiceTier.CRITICAL_ONLY: (0.99, 0.97, 0.95, 0.0, 0.0),
    ServiceTier.FULL_SERVICE: (0.94, 0.93, 0.92, 0.90, 0.88),
}


def _hourly_multiplier(hour: int) -> float:
    morning = math.exp(-0.5 * ((hour - 7.0) / 2.4) ** 2)
    evening = math.exp(-0.5 * ((hour - 19.0) / 3.1) ** 2)
    return 0.72 + 0.12 * morning + 0.31 * evening


def _locality_demand(hour: int, locality_index: int) -> tuple[float, ...]:
    total = _LOCALITY_BASE_DEMAND_MW[locality_index] * _hourly_multiplier(hour)
    return tuple(total * share for share in SECTOR_SHARES)


def _service_tier(controller: ControllerName, hour: int, locality_index: int) -> ServiceTier:
    if controller is ControllerName.PPO:
        rotation = (hour // 4) % len(EXPECTED_LOCALITY_IDS)
        return _PPO_TIER_SEQUENCE[(locality_index + rotation) % len(EXPECTED_LOCALITY_IDS)]
    block_start = ((hour // 3) * 3) % len(EXPECTED_LOCALITY_IDS)
    active = {(block_start + offset) % len(EXPECTED_LOCALITY_IDS) for offset in range(3)}
    return ServiceTier.FULL_SERVICE if locality_index in active else ServiceTier.OFF


def _frames_for_controller(controller: ControllerName) -> tuple[LiveScenarioFrame, ...]:
    frames: list[LiveScenarioFrame] = []
    for hour in range(24):
        locality_details: dict[str, LocalityHourDetail] = {}
        tiers: dict[str, ServiceTier] = {}
        for locality_index, locality_id in enumerate(EXPECTED_LOCALITY_IDS):
            demand = _locality_demand(hour, locality_index)
            tier = _service_tier(controller, hour, locality_index)
            service_ratios = _TIER_SERVICE_RATIOS[tier]
            served = tuple(
                sector_demand * ratio
                for sector_demand, ratio in zip(demand, service_ratios, strict=True)
            )
            critical_demand = sum(demand[:3])
            critical_served = sum(served[:3])
            next_hour_projection = (
                sum(_locality_demand(hour + 1, locality_index)) if 2 <= hour <= 22 else None
            )
            locality_details[locality_id] = LocalityHourDetail(
                demand_mw=sum(demand),
                served_mw=sum(served),
                critical_service_ratio=critical_served / critical_demand,
                sector_service_ratio={
                    sector_name: ratio
                    for sector_name, ratio in zip(
                        ("Hospital", "Water", "Bakery", "Residential", "Commercial"),
                        service_ratios,
                        strict=True,
                    )
                },
                next_hour_projected_demand_mw=next_hour_projection,
            )
            tiers[locality_id] = tier
        demand_mw = sum(detail.demand_mw for detail in locality_details.values())
        supply_ratio = 0.58 if 17 <= hour <= 20 else 0.68
        frames.append(
            LiveScenarioFrame(
                hour=hour,
                service_tiers=tiers,
                demand_mw=demand_mw,
                supply_mw=demand_mw * supply_ratio,
                locality_details=locality_details,
            )
        )
    return tuple(frames)


def _episode_summary(frames: tuple[LiveScenarioFrame, ...]) -> LiveEpisodeSummary:
    served_mwh = sum(
        detail.served_mw for frame in frames for detail in frame.locality_details.values()
    )
    demand_mwh = sum(
        detail.demand_mw for frame in frames for detail in frame.locality_details.values()
    )
    critical_demand_mwh = 0.0
    critical_served_mwh = 0.0
    cumulative_demand = [0.0] * len(EXPECTED_LOCALITY_IDS)
    cumulative_served = [0.0] * len(EXPECTED_LOCALITY_IDS)
    hourly_fairness: list[float] = []
    critical_share = sum(SECTOR_SHARES[:3])
    for frame in frames:
        for index, locality_id in enumerate(EXPECTED_LOCALITY_IDS):
            detail = frame.locality_details[locality_id]
            critical_demand = detail.demand_mw * critical_share
            critical_demand_mwh += critical_demand
            critical_served_mwh += critical_demand * detail.critical_service_ratio
            cumulative_demand[index] += detail.demand_mw
            cumulative_served[index] += detail.served_mw
        ratios = [
            served / demand
            for served, demand in zip(cumulative_served, cumulative_demand, strict=True)
        ]
        squared_sum = sum(ratios) ** 2
        sum_of_squares = sum(ratio**2 for ratio in ratios)
        hourly_fairness.append(squared_sum / (len(ratios) * sum_of_squares + 1e-8))
    return LiveEpisodeSummary(
        served_mwh=served_mwh,
        critical_service_ratio=critical_served_mwh / critical_demand_mwh,
        fairness=sum(hourly_fairness) / len(hourly_fairness),
        curtailed_mwh=demand_mwh - served_mwh,
        power_balance_violation_rate=0.0,
    )


def build_live_capture_comparison() -> LiveScenarioComparison:
    """Build deterministic, visibly varied live data used only for documentation capture."""

    frames_by_controller = {
        controller.value: _frames_for_controller(controller)
        for controller in (ControllerName.PPO, ControllerName.FIXED_ROTATION)
    }
    return LiveScenarioComparison(
        shortage_level="moderate",
        seed=50002,
        device="cuda:0",
        frames_by_controller=frames_by_controller,
        metrics_by_controller={
            name: _episode_summary(frames) for name, frames in frames_by_controller.items()
        },
    )


def _prepare_window(application: QApplication) -> PresentationWindow:
    application.setApplicationName("GridGuard Khartoum")
    application.setOrganizationName("GridGuard Khartoum")
    apply_theme(application)
    snapshot = load_presentation_snapshot()
    window = PresentationWindow(
        snapshot,
        runtime_label="Presentation capture | verified evidence | offline",
        map_asset=load_map_asset(),
    )
    window.setFixedSize(CAPTURE_SIZE)
    window.live_page.shortage_combo.setCurrentText("Moderate")
    window.live_page.seed_input.setValue(50002)
    window.live_page.set_comparison(build_live_capture_comparison())
    window.live_page.timeline.setValue(CAPTURE_HOUR)
    window.live_page.map_view.select_locality(CAPTURE_LOCALITY_ID)
    window.live_page.state_label.setProperty("state", "ready")
    window.live_page.state_label.setText(
        "Deterministic live presentation fixture · screenshot capture only · not frozen evidence."
    )
    window.live_page.state_label.setAccessibleName(window.live_page.state_label.text())
    window.live_page.state_label.style().unpolish(window.live_page.state_label)
    window.live_page.state_label.style().polish(window.live_page.state_label)
    return window


def _select_capture_page(window: PresentationWindow, page_index: int) -> None:
    window.tabs.setCurrentIndex(page_index)
    window.navigation_buttons[page_index].setFocus(Qt.FocusReason.OtherFocusReason)


def capture_screenshots(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    application: QApplication | None = None,
) -> tuple[Path, ...]:
    """Render all four real application pages to deterministic PNG files."""

    app = application or QApplication.instance() or QApplication(["gridguard-screenshot-capture"])
    if not isinstance(app, QApplication):
        raise RuntimeError("screenshot capture requires a QApplication")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    captured: list[Path] = []
    for name, page_index in PAGE_CAPTURES:
        window = _prepare_window(app)
        try:
            window.show()
            _select_capture_page(window, page_index)
            if name == "overview":
                window.overview_page.map_view.reset_view()
            elif name == "live":
                window.live_page.map_view.reset_view()
                window.live_page.map_view.select_locality(CAPTURE_LOCALITY_ID)
            elif name == "methodology":
                window.methodology_page.scroll_area.verticalScrollBar().setValue(0)
            window.navigation_buttons[page_index].setFocus(Qt.FocusReason.OtherFocusReason)
            app.processEvents()
            pixmap = window.grab()
            output_path = destination / f"{name}.png"
            temporary_path = destination / f".{name}.png.tmp"
            if not pixmap.save(str(temporary_path), "PNG"):
                raise OSError(f"could not encode screenshot: {output_path}")
            temporary_path.replace(output_path)
            captured.append(output_path.resolve())
        finally:
            window.close()
            app.processEvents()
    return tuple(captured)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for path in capture_screenshots(args.output_dir):
        print(f"Screenshot written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
