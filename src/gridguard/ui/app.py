"""Offline desktop entry point with fail-closed CUDA and evidence startup."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from gridguard.cli import ProjectPreflightResult, run_project_preflight
from gridguard.evaluation import EvaluationArtifactError
from gridguard.gpu import GpuPreflightError, GpuRuntimeInfo
from gridguard.policy import PolicyArtifactError
from gridguard.provenance import ProvenanceError
from gridguard.resources import ResourceLocationError
from gridguard.robustness import RobustnessArtifactError
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


def _write_startup_diagnostic(path: str | None, *, error: Exception | None = None) -> None:
    """Publish startup state atomically without allowing a logging failure to block the UI."""

    if not path:
        return
    temporary_path: Path | None = None
    try:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        cause = None if error is None else error.__cause__
        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "source": "application",
            "stage": "application-startup",
            "status": "READY" if error is None else "FAILED",
            "message": "Verified presentation window opened." if error is None else str(error),
            "executable_path": sys.executable,
            "exit_code": None if error is None else 1,
            "error_type": None if error is None else type(error).__name__,
            "cause": (
                None
                if cause is None
                else {"error_type": type(cause).__name__, "message": str(cause)}
            ),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=".startup-", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, allow_nan=False, indent=2, sort_keys=True)
            temporary.write("\n")
        temporary_path.replace(destination)
    except (OSError, ValueError):
        # The original trust failure and its dialog remain authoritative if logging is unavailable.
        pass
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Start the offline desktop shell, refusing to open after a failed preflight."""

    application = QApplication.instance() or QApplication(list(argv or sys.argv))
    application.setApplicationName("GridGuard Khartoum")
    application.setOrganizationName("GridGuard Khartoum")
    apply_theme(application)
    diagnostic_path = os.environ.get("GRIDGUARD_STARTUP_DIAGNOSTIC")
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
        RobustnessArtifactError,
    ) as error:
        _write_startup_diagnostic(diagnostic_path, error=error)
        if sys.stderr is not None:
            with suppress(OSError, ValueError):
                print(f"GridGuard desktop startup: FAIL - {error}", file=sys.stderr)
        QMessageBox.critical(
            None,
            "GridGuard startup blocked",
            f"The presentation cannot start because a required trust check failed.\n\n{error}",
        )
        return 1
    window.show()
    _write_startup_diagnostic(diagnostic_path)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
