from __future__ import annotations

import re
import runpy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import PyInstaller.utils.hooks as pyinstaller_hooks

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _StubAnalysis:
    pure: list[Any] = field(default_factory=list)
    scripts: list[Any] = field(default_factory=list)
    binaries: list[Any] = field(default_factory=list)
    datas: list[Any] = field(default_factory=list)


def test_pyinstaller_spec_resolves_inputs_from_the_repository_root(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    copied_metadata: list[str] = []

    monkeypatch.setattr(pyinstaller_hooks, "collect_all", lambda _package: ([], [], []))
    monkeypatch.setattr(
        pyinstaller_hooks,
        "collect_data_files",
        lambda _package: [
            ("C:/cuda/vJitLink_static.lib", "nvidia/vjitlink/lib/x64"),
            ("C:/cuda/cublasLt64_12.dll", "nvidia/cublas/bin"),
            ("C:/cuda/LICENSE.txt", "nvidia/cublas"),
        ],
    )
    monkeypatch.setattr(
        pyinstaller_hooks,
        "collect_dynamic_libs",
        lambda _package: [
            ("C:/cuda/cublasLt64_12.dll", "nvidia/cublas/bin"),
            ("C:/cuda/cupy_only.dll", "nvidia/runtime/bin"),
        ],
    )
    monkeypatch.setattr(pyinstaller_hooks, "collect_submodules", lambda _package: [])

    def copy_metadata(distribution: str) -> list[tuple[str, str]]:
        copied_metadata.append(distribution)
        return [
            (
                "C:/metadata/gridguard_khartoum-1.0.0.dist-info/METADATA",
                "gridguard_khartoum-1.0.0.dist-info",
            )
        ]

    monkeypatch.setattr(pyinstaller_hooks, "copy_metadata", copy_metadata)

    def analysis(scripts, **kwargs):
        captured["scripts"] = scripts
        captured["datas"] = kwargs["datas"]
        captured["hiddenimports"] = kwargs["hiddenimports"]
        return _StubAnalysis()

    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard.spec"),
        init_globals={
            "SPECPATH": str(REPOSITORY_ROOT / "packaging"),
            "Analysis": analysis,
            "PYZ": lambda *_args, **_kwargs: object(),
            "EXE": lambda *_args, **_kwargs: object(),
            "COLLECT": lambda *_args, **_kwargs: object(),
        },
    )

    assert namespace["PROJECT_ROOT"] == REPOSITORY_ROOT
    assert captured["scripts"] == [str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py")]
    assert all(
        Path(source).is_relative_to(REPOSITORY_ROOT)
        for source, _target in captured["datas"]
        if not source.startswith(("C:/cuda/", "C:/metadata/"))
    )
    assert copied_metadata == ["gridguard-khartoum"]
    assert (
        "C:/metadata/gridguard_khartoum-1.0.0.dist-info/METADATA",
        "gridguard_khartoum-1.0.0.dist-info",
    ) in captured["datas"]
    assert "graphlib" in captured["hiddenimports"]
    bundled_cuda_sources = {
        source
        for source, _target in (*captured["datas"], *namespace["binaries"])
        if source.startswith("C:/cuda/")
    }
    assert "C:/cuda/vJitLink_static.lib" not in bundled_cuda_sources
    assert "C:/cuda/cublasLt64_12.dll" not in bundled_cuda_sources
    assert "C:/cuda/LICENSE.txt" in bundled_cuda_sources
    assert "C:/cuda/cupy_only.dll" in bundled_cuda_sources


def test_pyinstaller_spec_is_one_directory_and_bundles_trusted_runtime_assets() -> None:
    spec = (REPOSITORY_ROOT / "packaging" / "gridguard.spec").read_text(encoding="utf-8")

    assert "COLLECT(" in spec
    assert "exclude_binaries=True" in spec
    assert 'name="GridGuard Khartoum"' in spec
    assert "data/evidence" in spec
    assert "data/model" in spec
    assert "graduation-project-proposal.pdf" in spec
    assert "gridguard/assets" in spec
    assert 'collect_all("cupy")' in spec
    assert 'collect_dynamic_libs("nvidia")' in spec


def test_runtime_hook_searches_torch_libraries_before_unique_nvidia_components() -> None:
    runtime_hook = (REPOSITORY_ROOT / "packaging" / "runtime_gpu_paths.py").read_text(
        encoding="utf-8"
    )

    assert 'bundle_root / "torch" / "lib"' in runtime_hook
    assert 'nvidia_root.glob("*/bin")' in runtime_hook


def test_frozen_launcher_only_delegates_to_the_verified_desktop_entry_point() -> None:
    launcher = (REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py").read_text(encoding="utf-8")

    assert "from gridguard.ui.app import main" in launcher
    assert "raise SystemExit(main())" in launcher


def test_frozen_launcher_has_a_fail_closed_automated_smoke_mode() -> None:
    launcher = (REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py").read_text(encoding="utf-8")

    assert "GRIDGUARD_SMOKE_REPORT" in launcher
    assert "prepare_startup()" in launcher
    assert "PresentationWindow(" in launcher
    assert "application.processEvents()" in launcher
    assert '"status": "PASS"' in launcher
    assert '"status": "FAIL"' in launcher
    assert '"cause": _exception_summary(error.__cause__)' in launcher


def test_frozen_smoke_runs_and_installs_the_primary_live_cuda_scenario() -> None:
    launcher = (REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py").read_text(encoding="utf-8")

    assert 'comparison = run_live_scenario("moderate", 50002)' in launcher
    assert "window.live_page.set_comparison(comparison)" in launcher
    assert "math.isfinite" in launcher
    assert '"shortage_level": comparison.shortage_level' in launcher
    assert '"seed": comparison.seed' in launcher
    assert '"controllers": list(controller_names)' in launcher
    assert '"frame_counts": frame_counts' in launcher
    assert '"finite_core_values": True' in launcher
    assert '"installed_in_live_page": True' in launcher
    assert "expected_results_table_rows = len(startup.snapshot.metrics)" in launcher
    assert "if expected_results_table_rows != 3" in launcher
    assert '"expected_results_table_rows": expected_results_table_rows' in launcher
    assert '"results_table_rows": results_table_rows' in launcher


def test_presentation_scripts_use_the_locked_spec_and_exact_executable() -> None:
    build_script = (REPOSITORY_ROOT / "scripts" / "build-presentation.ps1").read_text(
        encoding="utf-8"
    )
    run_script = (REPOSITORY_ROOT / "scripts" / "run-presentation.ps1").read_text(encoding="utf-8")

    assert "packaging\\gridguard.spec" in build_script
    assert "[switch]$VerifyOnly" in build_script
    assert "if (-not $VerifyOnly)" in build_script
    assert "uv sync --locked --extra dev --extra docs" in build_script
    assert "uv run pyinstaller" in build_script
    assert "GRIDGUARD_SOURCE_REPOSITORY" in build_script
    assert "GRIDGUARD_SOURCE_CHECKPOINT" in build_script
    assert "--junitxml" in build_script
    assert "$LASTEXITCODE" in build_script
    assert "$skippedCount -ne 0" in build_script
    assert "UV_CACHE_DIR" not in build_script
    assert re.search(r"[A-Za-z]:\\", build_script) is None
    assert "GRIDGUARD_SMOKE_REPORT" in build_script
    assert "QT_QPA_PLATFORM" in build_script
    assert "Start-Process" in build_script
    assert "-Wait" in build_script
    assert 'status -ne "PASS"' in build_script
    assert '$smokePayload.live_scenario.shortage_level -ne "moderate"' in build_script
    assert "$smokePayload.live_scenario.seed -ne 50002" in build_script
    assert "$smokePayload.live_scenario.frame_counts.PPO -ne 24" in build_script
    assert '$smokePayload.live_scenario.frame_counts."Fixed Rotation" -ne 24' in build_script
    assert '$smokePayload.live_scenario.controllers[0] -ne "PPO"' in build_script
    assert '$smokePayload.live_scenario.controllers[1] -ne "Fixed Rotation"' in build_script
    assert "$smokePayload.live_scenario.finite_core_values -ne $true" in build_script
    assert "$smokePayload.live_scenario.installed_in_live_page -ne $true" in build_script
    assert "$smokePayload.live_scenario.expected_results_table_rows -ne 3" in build_script
    assert "$smokePayload.live_scenario.results_table_rows -ne 3" in build_script
    assert 'Join-Path $projectRoot "RELEASE_MANIFEST.json"' in build_script
    assert "Get-FileHash -Algorithm SHA256" in build_script
    assert "$releaseManifest.windows_build.executable_sha256" in build_script
    assert "$releaseManifest.windows_build.executable_size_bytes" in build_script
    assert "$releaseManifest.windows_build.distribution_file_count" in build_script
    assert "$releaseManifest.windows_build.distribution_size_bytes" in build_script
    assert "$releaseManifest.windows_build.frozen_smoke_report.path" in build_script
    assert "$releaseManifest.windows_build.frozen_smoke_report.sha256" in build_script
    assert "$releaseManifest.windows_build.frozen_smoke_report.size_bytes" in build_script
    assert "$releaseManifest.windows_build.validated_runtime.gpu" in build_script
    assert "$releaseManifest.windows_build.validated_runtime.pytorch" in build_script
    assert "$releaseManifest.windows_build.validated_runtime.cuda" in build_script
    assert "$releaseManifest.windows_build.validated_runtime.cupy" in build_script
    assert "GridGuard Khartoum.exe" in run_script
    assert "Start-Process" in run_script
    assert "Test-Path -LiteralPath" in run_script
