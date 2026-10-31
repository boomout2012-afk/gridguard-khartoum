"""Frozen Windows entry point for GridGuard Khartoum."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# The selected PPO was evaluated with deterministic CUDA matrix operations.
# Set this before the preflight can initialize the CUDA context.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

    from gridguard.ui.window import PresentationWindow

_PRIVATE_SMOKE_OPTION = "--_gridguard-frozen-smoke-report"


def _frozen_smoke_report_root() -> Path:
    """Resolve the build-owned smoke directory from the frozen executable layout."""

    try:
        executable = Path(sys.executable).resolve(strict=True)
    except OSError as error:
        raise ValueError("frozen executable path cannot be resolved") from error
    distribution_root = executable.parent
    dist_root = distribution_root.parent
    if (
        executable.name.casefold() != "gridguard khartoum.exe"
        or distribution_root.name.casefold() != "gridguard khartoum"
    ):
        raise ValueError("frozen executable is outside the expected release layout")

    if dist_root.name.casefold() == "dist":
        expected_root = dist_root.parent / "build" / "gridguard"
    elif dist_root.name.casefold() == "app" and dist_root.parent.name.casefold() == "output":
        expected_root = dist_root.parent.parent / "tmp/proposal-extended/build/gridguard"
    else:
        raise ValueError("frozen executable is outside the expected release layout")
    try:
        resolved_root = expected_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("build-owned frozen smoke root must already exist") from error
    if resolved_root != expected_root or not resolved_root.is_dir():
        raise ValueError("build-owned frozen smoke root must be an unredirected directory")
    return resolved_root


def _validated_smoke_report_path(raw_path: str, *, allowed_root: Path) -> Path:
    report_path = Path(raw_path)
    if not report_path.is_absolute():
        raise ValueError("frozen smoke report path must be absolute")
    if report_path.name != "frozen-smoke.json":
        raise ValueError("frozen smoke report must be named frozen-smoke.json")
    try:
        report_root = Path(allowed_root).resolve(strict=True)
        report_parent = report_path.parent.resolve(strict=True)
    except OSError as error:
        raise ValueError("frozen smoke report root and parent must already exist") from error
    if not report_root.is_dir() or report_parent != report_root:
        raise ValueError("frozen smoke report must remain inside the build-owned root")

    validated_path = report_root / report_path.name
    if validated_path.is_symlink() or validated_path.exists():
        raise ValueError("frozen smoke report must not already exist")
    return validated_path


def _remove_ambient_offscreen_platform() -> None:
    os.environ.pop("QT_QPA_PLATFORM", None)


def _write_smoke_report(report_path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with report_path.open("xb") as report_file:
        report_file.write(encoded)


def _exception_summary(error: BaseException | None) -> dict[str, object] | None:
    if error is None:
        return None
    return {
        "error_type": type(error).__name__,
        "error": str(error),
        "missing_module": getattr(error, "name", None),
    }


def _deployed_policy_smoke_facts(comparison: object) -> dict[str, object]:
    """Reject an earlier policy or violated critical-service deployment constraints."""

    from dataclasses import asdict

    from gridguard.live_policy import DEPLOYED_LIVE_POLICY

    identity = getattr(comparison, "policy_identity", None)
    if identity != DEPLOYED_LIVE_POLICY:
        raise RuntimeError("frozen live policy differs from the validation-selected actor")
    metrics = getattr(comparison, "metrics_by_controller", {})
    for name in ("PPO", "Optimization"):
        summary = metrics.get(name)
        duration = getattr(summary, "duration", None)
        if (
            summary is None
            or duration is None
            or summary.power_balance_violation_rate != 0.0
            or duration.critical_interruption_locality_hours != 0.0
        ):
            raise RuntimeError(f"frozen {name} critical-service/power constraint check failed")
    return {**asdict(identity), "critical_and_power_checks_passed": True}


def _live_smoke_facts(
    comparison: object, *, expected_shortage_level: str = "moderate"
) -> dict[str, object]:
    from gridguard.domain import ControllerName
    from gridguard.ui.live_models import LiveScenarioComparison

    if not isinstance(comparison, LiveScenarioComparison):
        raise TypeError("frozen live smoke did not return a LiveScenarioComparison")

    controller_names = (
        ControllerName.PPO.value,
        ControllerName.FIXED_ROTATION.value,
        ControllerName.OPTIMIZATION.value,
    )
    if comparison.shortage_level != expected_shortage_level or comparison.seed != 50002:
        raise ValueError("frozen live smoke returned the wrong deterministic scenario")
    if comparison.device != "cuda:0":
        raise ValueError("frozen live smoke left required device cuda:0")
    if set(comparison.frames_by_controller) != set(controller_names):
        raise ValueError("frozen live smoke returned unexpected controller identities")

    frame_counts: dict[str, int] = {}
    core_values: list[float] = []
    for controller_name in controller_names:
        frames = comparison.frames(controller_name)
        frame_counts[controller_name] = len(frames)
        if len(frames) != 24 or tuple(frame.hour for frame in frames) != tuple(range(24)):
            raise ValueError(f"{controller_name} frozen live smoke must contain 24 frames")
        for frame in frames:
            core_values.extend((frame.demand_mw, frame.supply_mw))
            for detail in frame.locality_details.values():
                core_values.extend(
                    (
                        detail.demand_mw,
                        detail.served_mw,
                        detail.critical_service_ratio,
                    )
                )
                core_values.extend(detail.sector_service_ratio.values())
                if detail.next_hour_projected_demand_mw is not None:
                    core_values.append(detail.next_hour_projected_demand_mw)
    if not core_values or not all(math.isfinite(value) for value in core_values):
        raise ValueError("frozen live smoke produced non-finite core values")

    ppo_hour_zero = comparison.frames(ControllerName.PPO.value)[0]
    fixed_hour_zero = comparison.frames(ControllerName.FIXED_ROTATION.value)[0]
    optimization_hour_zero = comparison.frames(ControllerName.OPTIMIZATION.value)[0]
    return {
        "shortage_level": comparison.shortage_level,
        "seed": comparison.seed,
        "device": comparison.device,
        "controllers": list(controller_names),
        "frame_counts": frame_counts,
        "finite_core_values": True,
        "validated_core_value_count": len(core_values),
        "deployed_policy": _deployed_policy_smoke_facts(comparison),
        "hour_zero": {
            "demand_mw": ppo_hour_zero.demand_mw,
            "supply_mw": ppo_hour_zero.supply_mw,
            "ppo_served_mw": sum(
                detail.served_mw for detail in ppo_hour_zero.locality_details.values()
            ),
            "fixed_rotation_served_mw": sum(
                detail.served_mw for detail in fixed_hour_zero.locality_details.values()
            ),
            "optimization_served_mw": sum(
                detail.served_mw for detail in optimization_hour_zero.locality_details.values()
            ),
        },
    }


def _capture_presentation_pages(
    application: QApplication,
    window: PresentationWindow,
    report_path: Path,
) -> None:
    """Export actual packaged pages beside the smoke report for visual inspection."""

    from PySide6.QtCore import Qt

    labels = tuple(window.tabs.tabText(index) for index in range(window.tabs.count()))
    if labels != ("Overview", "Live Scenario", "Results"):
        raise RuntimeError("frozen presentation must contain the three approved pages")
    destination = report_path.parent / "screenshots"
    destination.mkdir(exist_ok=True)
    previous_index = window.tabs.currentIndex()
    previous_size = window.size()
    try:
        window.resize(1366, 768)
        for index, name in enumerate(("overview", "live", "results")):
            window.tabs.setCurrentIndex(index)
            window.navigation_buttons[index].setFocus(Qt.FocusReason.OtherFocusReason)
            application.processEvents()
            output_path = destination / f"{name}.png"
            temporary_path = destination / f".{name}.png.tmp"
            if not window.grab().save(str(temporary_path), "PNG"):
                raise OSError(f"could not encode packaged presentation screenshot: {output_path}")
            temporary_path.replace(output_path)
        window.tabs.setCurrentIndex(2)
        window.results_page.results_tabs.setCurrentIndex(1)
        for index, name in enumerate(("followup-means", "followup-paired", "followup-stress")):
            window.results_page.followup_page.detail_tabs.setCurrentIndex(index)
            application.processEvents()
            output_path = destination / f"{name}.png"
            temporary_path = destination / f".{name}.png.tmp"
            if not window.grab().save(str(temporary_path), "PNG"):
                raise OSError(f"could not encode packaged follow-up screenshot: {output_path}")
            temporary_path.replace(output_path)
    finally:
        window.results_page.results_tabs.setCurrentIndex(0)
        window.tabs.setCurrentIndex(previous_index)
        window.resize(previous_size)
        application.processEvents()


def _focused_study_smoke_facts(page: object) -> dict[str, object]:
    """Require the packaged current study to load and display its pinned values."""

    view = getattr(page, "study_view", None)
    table = getattr(page, "current_results_table", None)
    if view is None or table is None:
        raise RuntimeError("packaged current study could not be verified")
    if table.rowCount() != len(view.means.rows) or table.columnCount() != len(view.means.columns):
        raise RuntimeError("packaged current study table shape differs from evidence")
    for row, cells in enumerate(view.means.rows):
        for column, expected in enumerate(cells):
            item = table.item(row, column)
            if item is None or item.text() != expected:
                raise RuntimeError("packaged current study display differs from evidence")
    return {
        "display_matches_pinned_study": True,
        "displayed_outcomes": table.rowCount(),
        "evidence": dict(view.evidence),
        "fixed_rotation_conclusion": view.fixed_conclusion,
        "optimizer_conclusion": view.optimizer_conclusion,
        "live_policy_boundary": view.live_statement,
    }


def _followup_smoke_facts(page: object) -> dict[str, object]:
    """Require every follow-up table to display its separately pinned audit values."""
    followup = getattr(page, "followup_page", None)
    view = getattr(followup, "view", None)
    means = getattr(followup, "means_table", None)
    stress = getattr(followup, "stress_table", None)
    paired = getattr(followup, "paired_tables", ())
    if view is None or means is None or stress is None or len(paired) != len(view.paired):
        raise RuntimeError("packaged follow-up evidence could not be verified")
    tables = [(means, view.means), (stress, view.stress)] + [
        (table, data) for table, (_, data) in zip(paired, view.paired, strict=True)
    ]
    for table, data in tables:
        if table.rowCount() != len(data.rows) or table.columnCount() != len(data.columns):
            raise RuntimeError("packaged follow-up table shape differs from evidence")
        for row, cells in enumerate(data.rows):
            for column, expected in enumerate(cells):
                item = table.item(row, column)
                if item is None or item.text() != expected:
                    raise RuntimeError("packaged follow-up display differs from evidence")
    return {
        "status": "PASS",
        "display_matches_pinned_audit": True,
        "controller_rows": means.rowCount(),
        "paired_tables": len(paired),
        "stress_rows": stress.rowCount(),
        "scope_statement": view.scope_statement,
        "evidence": dict(view.evidence),
    }


def _run_frozen_smoke(report_path: Path) -> int:
    from PySide6.QtWidgets import QApplication

    from gridguard import __version__
    from gridguard.ui.app import format_runtime_label, prepare_startup
    from gridguard.ui.live_runner import run_live_scenario
    from gridguard.ui.theme import apply_theme
    from gridguard.ui.window import PresentationWindow

    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("GridGuard Khartoum")
    application.setOrganizationName("GridGuard Khartoum")
    apply_theme(application)
    window = None
    try:
        startup = prepare_startup()
        expected_results_table_rows = len(startup.snapshot.metrics)
        if expected_results_table_rows != 3:
            raise RuntimeError("frozen presentation snapshot must contain three primary metrics")
        window = PresentationWindow(
            startup.snapshot,
            runtime_label=format_runtime_label(startup.gpu),
            live_runner=run_live_scenario,
        )
        window.show()
        application.processEvents()
        comparison = run_live_scenario("moderate", 50002)
        live_facts = _live_smoke_facts(comparison)
        all_shortage_facts = {"moderate": live_facts.copy()}
        for shortage_level in ("mild", "severe"):
            matched = run_live_scenario(shortage_level, 50002)
            all_shortage_facts[shortage_level] = _live_smoke_facts(
                matched, expected_shortage_level=shortage_level
            )
        window.live_page.set_comparison(comparison)
        window.tabs.setCurrentWidget(window.live_page)
        window.live_page.timeline.setValue(23)
        window.live_page.controller_combo.setCurrentText("Optimization")
        application.processEvents()
        if (
            not window.live_page.timeline.isEnabled()
            or not window.live_page.controller_combo.isEnabled()
            or window.live_page.controller_combo.currentText() != "Optimization"
            or window.live_page.timeline.value() != 23
            or not window.live_page.locality_context.text().endswith("Hour 23:00")
        ):
            raise RuntimeError("frozen live comparison was not installed in the presentation UI")
        results_table_rows = window.results_page.results_table.rowCount()
        if results_table_rows != expected_results_table_rows:
            raise RuntimeError("frozen presentation results table is missing expected rows")
        current_study_facts = _focused_study_smoke_facts(window.results_page)
        followup_facts = _followup_smoke_facts(window.results_page)
        window.live_page.controller_combo.setCurrentText("PPO")
        application.processEvents()
        live_facts.update(
            {
                "installed_in_live_page": True,
                "selected_controller": window.live_page.controller_combo.currentText(),
                "selected_hour": window.live_page.timeline.value(),
                "expected_results_table_rows": expected_results_table_rows,
                "results_table_rows": results_table_rows,
            }
        )
        _capture_presentation_pages(application, window, report_path)
        _write_smoke_report(report_path.parent / "followup-smoke.json", followup_facts)
        _write_smoke_report(
            report_path,
            {
                "release_id": f"gridguard-khartoum-v{__version__}",
                "status": "PASS",
                "window_title": window.windowTitle(),
                "gpu": {
                    "device_name": startup.gpu.device_name,
                    "torch_version": startup.gpu.torch_version,
                    "torch_cuda_version": startup.gpu.torch_cuda_version,
                    "cupy_version": startup.gpu.cupy_version,
                },
                "live_scenario": live_facts,
                "live_scenarios_by_shortage": all_shortage_facts,
                "current_study": current_study_facts,
            },
        )
        return 0
    except Exception as error:
        _write_smoke_report(
            report_path,
            {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "cause": _exception_summary(error.__cause__),
            },
        )
        return 1
    finally:
        if window is not None:
            window.close()
            application.processEvents()


def _run_desktop() -> None:
    from gridguard.ui.app import main

    raise SystemExit(main())


def _run_launcher(arguments: list[str]) -> int:
    if _PRIVATE_SMOKE_OPTION not in arguments:
        _remove_ambient_offscreen_platform()
        _run_desktop()
        return 0

    if len(arguments) != 2 or arguments[0] != _PRIVATE_SMOKE_OPTION:
        raise ValueError("invalid private frozen-smoke invocation")
    smoke_report = _validated_smoke_report_path(
        arguments[1],
        allowed_root=_frozen_smoke_report_root(),
    )
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    return _run_frozen_smoke(smoke_report)


if __name__ == "__main__":
    raise SystemExit(_run_launcher(sys.argv[1:]))
