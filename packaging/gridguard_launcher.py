"""Frozen Windows entry point for GridGuard Khartoum."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

_SMOKE_REPORT_ENV = "GRIDGUARD_SMOKE_REPORT"

if os.environ.get(_SMOKE_REPORT_ENV):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _write_smoke_report(report_path: Path, payload: dict[str, object]) -> None:
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _exception_summary(error: BaseException | None) -> dict[str, object] | None:
    if error is None:
        return None
    return {
        "error_type": type(error).__name__,
        "error": str(error),
        "missing_module": getattr(error, "name", None),
    }


def _live_smoke_facts(comparison: object) -> dict[str, object]:
    from gridguard.domain import ControllerName
    from gridguard.ui.live_models import LiveScenarioComparison

    if not isinstance(comparison, LiveScenarioComparison):
        raise TypeError("frozen live smoke did not return a LiveScenarioComparison")

    controller_names = (
        ControllerName.PPO.value,
        ControllerName.FIXED_ROTATION.value,
    )
    if comparison.shortage_level != "moderate" or comparison.seed != 50002:
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
    return {
        "shortage_level": comparison.shortage_level,
        "seed": comparison.seed,
        "device": comparison.device,
        "controllers": list(controller_names),
        "frame_counts": frame_counts,
        "finite_core_values": True,
        "validated_core_value_count": len(core_values),
        "hour_zero": {
            "demand_mw": ppo_hour_zero.demand_mw,
            "supply_mw": ppo_hour_zero.supply_mw,
            "ppo_served_mw": sum(
                detail.served_mw for detail in ppo_hour_zero.locality_details.values()
            ),
            "fixed_rotation_served_mw": sum(
                detail.served_mw for detail in fixed_hour_zero.locality_details.values()
            ),
        },
    }


def _run_frozen_smoke(report_path: Path) -> int:
    from PySide6.QtWidgets import QApplication

    from gridguard.ui.app import format_runtime_label, prepare_startup
    from gridguard.ui.live_runner import run_live_scenario
    from gridguard.ui.window import PresentationWindow

    application = QApplication.instance() or QApplication(sys.argv[:1])
    window = None
    try:
        startup = prepare_startup()
        expected_results_table_rows = len(startup.snapshot.metrics)
        if expected_results_table_rows != 7:
            raise RuntimeError("frozen presentation snapshot must contain seven locked metrics")
        window = PresentationWindow(
            startup.snapshot,
            runtime_label=format_runtime_label(startup.gpu),
            live_runner=run_live_scenario,
        )
        window.show()
        application.processEvents()
        comparison = run_live_scenario("moderate", 50002)
        live_facts = _live_smoke_facts(comparison)
        window.live_page.set_comparison(comparison)
        window.tabs.setCurrentWidget(window.live_page)
        window.live_page.timeline.setValue(23)
        window.live_page.controller_combo.setCurrentText("Fixed Rotation")
        application.processEvents()
        if (
            not window.live_page.timeline.isEnabled()
            or not window.live_page.controller_combo.isEnabled()
            or window.live_page.controller_combo.currentText() != "Fixed Rotation"
            or window.live_page.timeline.value() != 23
            or not window.live_page.locality_context.text().endswith("Hour 23:00")
        ):
            raise RuntimeError("frozen live comparison was not installed in the presentation UI")
        results_table_rows = window.results_page.results_table.rowCount()
        if results_table_rows != expected_results_table_rows:
            raise RuntimeError("frozen presentation results table is missing expected rows")
        live_facts.update(
            {
                "installed_in_live_page": True,
                "selected_controller": window.live_page.controller_combo.currentText(),
                "selected_hour": window.live_page.timeline.value(),
                "expected_results_table_rows": expected_results_table_rows,
                "results_table_rows": results_table_rows,
            }
        )
        _write_smoke_report(
            report_path,
            {
                "status": "PASS",
                "window_title": window.windowTitle(),
                "gpu": {
                    "device_name": startup.gpu.device_name,
                    "torch_version": startup.gpu.torch_version,
                    "torch_cuda_version": startup.gpu.torch_cuda_version,
                    "cupy_version": startup.gpu.cupy_version,
                },
                "live_scenario": live_facts,
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


if __name__ == "__main__":
    smoke_report = os.environ.get(_SMOKE_REPORT_ENV)
    if smoke_report:
        raise SystemExit(_run_frozen_smoke(Path(smoke_report)))
    _run_desktop()
