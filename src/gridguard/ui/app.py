"""Offline desktop entry point with fail-closed CUDA and evidence startup."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from PySide6.QtWidgets import QApplication, QMessageBox

from gridguard.cli import ProjectPreflightResult, run_project_preflight
from gridguard.evaluation import EvaluationArtifactError
from gridguard.gpu import GpuPreflightError, GpuRuntimeInfo
from gridguard.policy import PolicyArtifactError
from gridguard.provenance import ProvenanceError
from gridguard.resources import ResourceLocationError
from gridguard.ui.live_runner import run_live_scenario
from gridguard.ui.map_asset import MapAssetValidationError
from gridguard.ui.presentation_model import (
    PresentationDataError,
    PresentationSnapshot,
    load_presentation_snapshot,
)
from gridguard.ui.theme import apply_theme
from gridguard.ui.window import PresentationWindow


@dataclass(frozen=True, slots=True)
class StartupBundle:
    """Dependencies proven safe before the first presentation window is shown."""

    gpu: GpuRuntimeInfo
    snapshot: PresentationSnapshot


def prepare_startup(
    *,
    preflight: Callable[[], ProjectPreflightResult] = run_project_preflight,
    snapshot_loader: Callable[[], PresentationSnapshot] = load_presentation_snapshot,
) -> StartupBundle:
    """Run the complete project gate, then load its pinned presentation evidence."""

    project = preflight()
    snapshot = snapshot_loader()
    return StartupBundle(gpu=project.gpu, snapshot=snapshot)


def format_runtime_label(info: GpuRuntimeInfo) -> str:
    """Format the verified runtime using presentation-safe ASCII separators."""

    return (
        f"{info.device_name} | PyTorch {info.torch_version} | "
        f"CUDA {info.torch_cuda_version} | CuPy {info.cupy_version}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Start the offline desktop shell, refusing to open after a failed preflight."""

    application = QApplication.instance() or QApplication(list(argv or sys.argv))
    application.setApplicationName("GridGuard Khartoum")
    application.setOrganizationName("GridGuard Khartoum")
    apply_theme(application)
    try:
        startup = prepare_startup()
        window = PresentationWindow(
            startup.snapshot,
            runtime_label=format_runtime_label(startup.gpu),
            live_runner=run_live_scenario,
        )
    except (
        EvaluationArtifactError,
        GpuPreflightError,
        MapAssetValidationError,
        PolicyArtifactError,
        PresentationDataError,
        ProvenanceError,
        ResourceLocationError,
    ) as error:
        QMessageBox.critical(
            None,
            "GridGuard startup blocked",
            f"The presentation cannot start because a required trust check failed.\n\n{error}",
        )
        print(f"GridGuard desktop startup: FAIL - {error}", file=sys.stderr)
        return 1
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
