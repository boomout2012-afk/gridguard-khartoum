from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import PyInstaller.utils.hooks as pyinstaller_hooks
import pytest

from gridguard.robustness_provenance import ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


RUNTIME_PACKAGE_FILES = (
    "data/evidence/frozen-benchmark.json",
    "data/evidence/paired-benchmark.json",
    "data/evidence/robustness-supplement.json",
    "data/evidence/proposal-supplement.json",
    "data/evidence/focused-ppo-study.json",
    "data/evidence/focused-ppo-training.json",
    "docs/focused-ppo-validation.json",
    "data/evidence/source-manifest.json",
    "data/model/manifest.json",
    "data/model/ppo-policy.npz",
    "data/model/focused-ppo-study/seed-8111-selected.npz",
    "data/model/training-provenance.yaml",
    "docs/source/graduation-project-proposal.pdf",
    "docs/PROPOSAL_EXTENSION.md",
    "tools/evaluate_proposal_extension.py",
    "src/gridguard/assets/fonts/AtkinsonHyperlegible-Bold.ttf",
    "src/gridguard/assets/fonts/AtkinsonHyperlegible-Regular.ttf",
    "src/gridguard/assets/fonts/IBMPlexMono-Regular.ttf",
    "src/gridguard/assets/fonts/IBMPlexMono-SemiBold.ttf",
    "src/gridguard/assets/fonts/SourceSerif4-Regular.ttf",
    "src/gridguard/assets/fonts/SourceSerif4-Semibold.ttf",
    "src/gridguard/assets/fonts/THIRD_PARTY_FONTS.md",
    "src/gridguard/assets/fonts/licenses/AtkinsonHyperlegible-OFL.txt",
    "src/gridguard/assets/fonts/licenses/IBMPlexMono-license.txt",
    "src/gridguard/assets/fonts/licenses/SourceSerif4-LICENSE.md",
    "src/gridguard/assets/map/THIRD_PARTY_DATA.md",
    "src/gridguard/assets/map/khartoum-localities.geojson",
    "src/gridguard/assets/map/khartoum-nile.geojson",
    "src/gridguard/assets/map/map-provenance.json",
)


PROVENANCE_SOURCE_FILES = (
    "src/gridguard/robustness_provenance.py",
    *(
        source_path
        for source_path, _sha256, _size_bytes in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
    ),
    "src/gridguard/duration_metrics.py",
    "src/gridguard/forecast_model.py",
    "src/gridguard/optimization.py",
    "src/gridguard/proposal_evidence.py",
    "src/gridguard/proposal_provenance.py",
    *(
        f"data/evidence/accepted-sources/{source_path}"
        for source_path, _sha256, _size_bytes in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS
    ),
)


GRIDGUARD_METADATA_CONTENTS = {
    "METADATA": (
        b"Metadata-Version: 2.5\n"
        b"Name: gridguard-khartoum\n"
        b"Version: 1.2.0\n"
        b"Summary: Offline PPO load-shedding decision-support simulation for Khartoum State\n"
        b"Requires-Python: <3.13,>=3.12\n"
        b"Requires-Dist: cupy-cuda12x[ctk]<16,>=13.6\n"
        b"Requires-Dist: numpy<3,>=2.0\n"
        b"Requires-Dist: pyside6-essentials<7,>=6.8\n"
        b"Requires-Dist: torch==2.8.0\n"
        b"Provides-Extra: dev\n"
        b"Requires-Dist: hatchling==1.32.0; extra == 'dev'\n"
        b"Requires-Dist: pyinstaller<7,>=6.15; extra == 'dev'\n"
        b"Requires-Dist: pytest-qt<5,>=4.5; extra == 'dev'\n"
        b"Requires-Dist: pytest<10,>=8.4; extra == 'dev'\n"
        b"Requires-Dist: ruff<1,>=0.12; extra == 'dev'\n"
        b"Provides-Extra: docs\n"
        b"Requires-Dist: arabic-reshaper==3.0.0; extra == 'docs'\n"
        b"Requires-Dist: lxml<7,>=6; extra == 'docs'\n"
        b"Requires-Dist: pdfplumber<1,>=0.11; extra == 'docs'\n"
        b"Requires-Dist: pypdf<7,>=6; extra == 'docs'\n"
        b"Requires-Dist: python-bidi<0.7,>=0.6; extra == 'docs'\n"
        b"Requires-Dist: python-docx<2,>=1.2; extra == 'docs'\n"
        b"Requires-Dist: pyyaml<7,>=6; extra == 'docs'\n"
        b"Requires-Dist: reportlab<5,>=4.4; extra == 'docs'\n"
    ),
    "WHEEL": (b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
    "entry_points.txt": (
        b"[console_scripts]\n"
        b"gridguard-desktop = gridguard.ui.app:main\n"
        b"gridguard-doctor = gridguard.cli:doctor_main\n"
        b"gridguard-evaluate = gridguard.cli:main\n"
    ),
}


GRIDGUARD_METADATA_FILES = tuple(GRIDGUARD_METADATA_CONTENTS)


@dataclass
class _StubAnalysis:
    pure: list[Any] = field(default_factory=list)
    scripts: list[Any] = field(default_factory=list)
    binaries: list[Any] = field(default_factory=list)
    datas: list[Any] = field(default_factory=list)


def _load_pyinstaller_spec_namespace(
    monkeypatch,
    torch_package_root: Path,
    *,
    finalized_binaries: tuple[tuple[str, str, str], ...] = (),
    finalized_datas: tuple[tuple[str, str, str], ...] = (),
) -> dict[str, Any]:
    (torch_package_root / "lib").mkdir(parents=True, exist_ok=True)
    # The real build deliberately narrows PATH. Keep that side effect inside
    # the test fixture so subsequent PowerShell/uv/Git tests retain their tools.
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setattr(pyinstaller_hooks, "collect_all", lambda _package: ([], [], []))
    monkeypatch.setattr(pyinstaller_hooks, "collect_data_files", lambda _package: [])
    monkeypatch.setattr(pyinstaller_hooks, "collect_dynamic_libs", lambda _package: [])
    monkeypatch.setattr(pyinstaller_hooks, "collect_submodules", lambda _package: [])

    def analysis(*_args, **_kwargs):
        return _StubAnalysis(
            binaries=list(finalized_binaries),
            datas=list(finalized_datas),
        )

    return runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard.spec"),
        init_globals={
            "SPECPATH": str(REPOSITORY_ROOT / "packaging"),
            "Analysis": analysis,
            "PYZ": lambda *_args, **_kwargs: object(),
            "EXE": lambda *_args, **_kwargs: object(),
            "COLLECT": lambda *_args, **_kwargs: object(),
        },
    )


def test_spec_namespace_does_not_leak_build_path_into_other_tests(monkeypatch, tmp_path) -> None:
    original_path = os.environ.get("PATH", "")
    with monkeypatch.context() as isolated:
        _load_pyinstaller_spec_namespace(isolated, tmp_path / "torch")
    assert os.environ.get("PATH", "") == original_path


def test_pyinstaller_spec_deduplicates_hook_added_finalized_nvidia_toc_entries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    torch_package_root = tmp_path / "torch"
    torch_dll = torch_package_root / "lib" / "cublasLt64_12.dll"
    torch_dll.parent.mkdir(parents=True)
    torch_dll.write_bytes(b"shared-cuda-bytes")
    nvidia_root = tmp_path / "nvidia"
    identical = nvidia_root / "cublas" / "bin" / torch_dll.name
    alias = nvidia_root / "runtime" / "bin" / "renamed_cuda.dll"
    same_name_different = nvidia_root / "runtime" / "bin" / torch_dll.name
    unique = nvidia_root / "runtime" / "bin" / "nvrtc64_120_0.dll"
    hook_added_static = nvidia_root / "nvjitlink" / "lib" / "nvJitLink_static.lib"
    unrelated = tmp_path / "plugins" / torch_dll.name
    for path in (identical, alias, same_name_different, unique, hook_added_static, unrelated):
        path.parent.mkdir(parents=True, exist_ok=True)
    identical.write_bytes(torch_dll.read_bytes())
    alias.write_bytes(torch_dll.read_bytes())
    same_name_different.write_bytes(b"different-cuda-xx")
    assert same_name_different.stat().st_size == torch_dll.stat().st_size
    unique.write_bytes(b"unique-runtime")
    hook_added_static.write_bytes(b"link-time-only")
    unrelated.write_bytes(torch_dll.read_bytes())

    torch_entry = (f"torch/lib/{torch_dll.name}", str(torch_dll), "BINARY")
    identical_entry = (
        f"nvidia/cublas/bin/{identical.name}",
        str(identical),
        "BINARY",
    )
    alias_entry = (f"nvidia/runtime/bin/{alias.name}", str(alias), "BINARY")
    different_entry = (
        f"nvidia/runtime/bin/{same_name_different.name}",
        str(same_name_different),
        "BINARY",
    )
    unique_entry = (f"nvidia/runtime/bin/{unique.name}", str(unique), "BINARY")
    static_entry = (
        f"nvidia/nvjitlink/lib/{hook_added_static.name}",
        str(hook_added_static),
        "BINARY",
    )
    unrelated_entry = (f"plugins/{unrelated.name}", str(unrelated), "BINARY")
    identical_data_entry = (
        f"nvidia/cublas/bin/{identical.name}",
        str(identical),
        "DATA",
    )
    alias_data_entry = (f"nvidia/runtime/bin/{alias.name}", str(alias), "DATA")
    namespace = _load_pyinstaller_spec_namespace(
        monkeypatch,
        torch_package_root,
        finalized_binaries=(
            unrelated_entry,
            torch_entry,
            identical_entry,
            alias_entry,
            different_entry,
            unique_entry,
            static_entry,
        ),
        finalized_datas=(identical_data_entry, alias_data_entry),
    )

    expected_binaries = [
        unrelated_entry,
        torch_entry,
        alias_entry,
        different_entry,
        unique_entry,
    ]
    assert namespace["analysis"].binaries == expected_binaries
    assert all(
        actual is expected
        for actual, expected in zip(namespace["analysis"].binaries, expected_binaries, strict=True)
    )
    assert namespace["analysis"].datas == [alias_data_entry]
    assert namespace["analysis"].datas[0] is alias_data_entry


def test_finalized_nvidia_filter_fails_closed_when_a_dll_payload_is_unavailable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    torch_package_root = tmp_path / "torch"
    torch_dll = torch_package_root / "lib" / "torch_cuda.dll"
    torch_dll.parent.mkdir(parents=True)
    torch_dll.write_bytes(b"pinned")
    missing_dll = tmp_path / "nvidia" / "runtime" / "bin" / "missing.dll"
    torch_entry = (f"torch/lib/{torch_dll.name}", str(torch_dll), "BINARY")
    missing_entry = (
        f"nvidia/runtime/bin/{torch_dll.name}",
        str(missing_dll),
        "BINARY",
    )

    with pytest.raises(FileNotFoundError, match="NVIDIA payload.*unavailable"):
        _load_pyinstaller_spec_namespace(
            monkeypatch,
            torch_package_root,
            finalized_binaries=(torch_entry, missing_entry),
        )


def test_pyinstaller_spec_resolves_inputs_from_the_repository_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    # The spec deliberately restricts DLL discovery; restore the runner's tool PATH.
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    captured: dict[str, Any] = {}
    copied_metadata: list[str] = []
    real_copy_metadata = pyinstaller_hooks.copy_metadata
    metadata_directories = real_copy_metadata("gridguard-khartoum")
    assert len(metadata_directories) == 1
    metadata_root = Path(metadata_directories[0][0])
    assert metadata_root.is_dir()
    assert metadata_directories[0][1] == metadata_root.name
    assert {
        "direct_url.json",
        "RECORD",
        *GRIDGUARD_METADATA_FILES,
    }.issubset(path.name for path in metadata_root.iterdir())

    torch_package_root = tmp_path / "torch"
    torch_dll = torch_package_root / "lib" / "torch_cuda_payload.dll"
    torch_dll.parent.mkdir(parents=True)
    torch_dll.write_bytes(b"shared-cublas-bytes")

    cuda_root = tmp_path / "cuda"
    static_library = cuda_root / "vJitLink_static.lib"
    duplicate_dll = cuda_root / "cublasLt64_12.dll"
    license_file = cuda_root / "LICENSE.txt"
    unique_dll = cuda_root / "cupy_only.dll"
    cuda_root.mkdir()
    static_library.write_bytes(b"link-time-only")
    duplicate_dll.write_bytes(torch_dll.read_bytes())
    license_file.write_text("license", encoding="utf-8")
    unique_dll.write_bytes(b"cupy-only-runtime")
    external_cuda_sources = {
        str(static_library),
        str(duplicate_dll),
        str(license_file),
        str(unique_dll),
    }

    monkeypatch.setattr(pyinstaller_hooks, "collect_all", lambda _package: ([], [], []))
    monkeypatch.setattr(
        pyinstaller_hooks,
        "collect_data_files",
        lambda _package: [
            (str(static_library), "nvidia/vjitlink/lib/x64"),
            (str(duplicate_dll), "nvidia/cublas/bin"),
            (str(license_file), "nvidia/cublas"),
        ],
    )
    monkeypatch.setattr(
        pyinstaller_hooks,
        "collect_dynamic_libs",
        lambda _package: [
            (str(duplicate_dll), "nvidia/cublas/bin"),
            (str(unique_dll), "nvidia/runtime/bin"),
        ],
    )
    monkeypatch.setattr(pyinstaller_hooks, "collect_submodules", lambda _package: [])

    def copy_metadata(distribution: str) -> list[tuple[str, str]]:
        copied_metadata.append(distribution)
        return real_copy_metadata(distribution)

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
        if source not in external_cuda_sources and not Path(source).is_relative_to(metadata_root)
    )
    assert copied_metadata == ["gridguard-khartoum"]
    bundled_gridguard_metadata = {
        Path(source).name
        for source, target in captured["datas"]
        if target == "gridguard_khartoum-1.2.0.dist-info"
    }
    assert bundled_gridguard_metadata == {"METADATA", "WHEEL", "entry_points.txt"}
    assert {
        Path(source) for source, target in captured["datas"] if target == metadata_root.name
    } == {metadata_root / name for name in GRIDGUARD_METADATA_FILES}
    assert "graphlib" in captured["hiddenimports"]
    bundled_cuda_sources = {
        source
        for source, _target in (*captured["datas"], *namespace["binaries"])
        if source in external_cuda_sources
    }
    assert str(static_library) not in bundled_cuda_sources
    assert str(duplicate_dll) in bundled_cuda_sources
    assert str(license_file) in bundled_cuda_sources
    assert str(unique_dll) in bundled_cuda_sources

    bundled_provenance_sources = {
        Path(source).resolve().relative_to(REPOSITORY_ROOT).as_posix()
        for source, target in captured["datas"]
        if target == "gridguard/_bundled/src/gridguard"
    }
    expected_provenance_sources = {
        source_path for source_path in PROVENANCE_SOURCE_FILES if source_path.startswith("src/")
    }
    assert bundled_provenance_sources == expected_provenance_sources

    runtime_sources = {
        Path(source).resolve().relative_to(REPOSITORY_ROOT).as_posix()
        for source, _target in captured["datas"]
        if source not in external_cuda_sources
        and not Path(source).is_relative_to(metadata_root)
        and "_bundled/src/gridguard" not in _target
        and "_bundled/data/evidence/accepted-sources/" not in _target
    }
    assert runtime_sources == set(RUNTIME_PACKAGE_FILES) | {
        "data/evidence/committee-baseline-audit.json",
        "data/evidence/committee-baseline-audit.validation.json",
    }
    assert all(
        not Path(source).is_dir()
        for source, _target in captured["datas"]
        if source not in external_cuda_sources
    )


def test_pyinstaller_spec_is_one_directory_and_bundles_trusted_runtime_assets() -> None:
    spec = (REPOSITORY_ROOT / "packaging" / "gridguard.spec").read_text(encoding="utf-8")

    assert "COLLECT(" in spec
    assert "exclude_binaries=True" in spec
    assert 'name="GridGuard Khartoum"' in spec
    assert "data/evidence" in spec
    assert "data/model" in spec
    assert "graduation-project-proposal.pdf" in spec
    assert "robustness_provenance.py" in spec
    assert "*_bundled_provenance_sources()" in spec
    assert "gridguard/assets" in spec
    assert 'collect_all("cupy")' in spec
    assert 'collect_dynamic_libs("nvidia")' in spec
    assert "_minimal_distribution_metadata" in spec
    assert "_audited_runtime_data" in spec
    assert '(str(PROJECT_ROOT / "data" / "evidence"),' not in spec
    assert '(str(PROJECT_ROOT / "data" / "model"),' not in spec
    assert '(str(PROJECT_ROOT / "src" / "gridguard" / "assets"),' not in spec


def test_wheel_force_includes_only_audited_runtime_files() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'exclude = ["src/gridguard/assets/**"]' in pyproject
    assert '"data/evidence" = ' not in pyproject
    assert '"data/model" = ' not in pyproject
    for source_path in RUNTIME_PACKAGE_FILES:
        assert f'"{source_path}" = ' in pyproject

    assert "[tool.hatch.build.targets.sdist]" in pyproject


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

    assert "GRIDGUARD_SMOKE_REPORT" not in launcher
    assert '"--_gridguard-frozen-smoke-report"' in launcher
    assert "_validated_smoke_report_path" in launcher
    assert "prepare_startup()" in launcher
    assert "PresentationWindow(" in launcher
    assert "application.processEvents()" in launcher
    assert '"status": "PASS"' in launcher
    assert '"status": "FAIL"' in launcher
    assert "from gridguard import __version__" in launcher
    assert '"release_id": f"gridguard-khartoum-v{__version__}"' in launcher
    assert '"cause": _exception_summary(error.__cause__)' in launcher


def test_frozen_launcher_ignores_ambient_smoke_controls_for_normal_launch(monkeypatch) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    launch = namespace["_run_launcher"]
    launcher_globals = launch.__globals__
    calls: list[str] = []
    launcher_globals["_run_desktop"] = lambda: calls.append("desktop")
    launcher_globals["_run_frozen_smoke"] = lambda _path: calls.append("smoke") or 1
    monkeypatch.setenv("GRIDGUARD_SMOKE_REPORT", "C:/ambient/frozen-smoke.json")
    monkeypatch.setenv("QT_QPA_PLATFORM", " OffScreen:fontengine=freetype ")

    result = launch([])

    assert result == 0
    assert calls == ["desktop"]
    assert "QT_QPA_PLATFORM" not in launcher_globals["os"].environ
    assert launcher_globals["os"].environ["GRIDGUARD_SMOKE_REPORT"] == (
        "C:/ambient/frozen-smoke.json"
    )


def test_frozen_launcher_requires_an_explicit_validated_smoke_report_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    launch = namespace["_run_launcher"]
    launcher_globals = launch.__globals__
    captured: dict[str, object] = {}
    report_path = tmp_path / "frozen-smoke.json"

    def run_smoke(path: Path) -> int:
        captured["path"] = path
        captured["qt_platform"] = launcher_globals["os"].environ.get("QT_QPA_PLATFORM")
        return 17

    launcher_globals["_run_frozen_smoke"] = run_smoke
    launcher_globals["_run_desktop"] = lambda: captured.setdefault("desktop", True)
    launcher_globals["_frozen_smoke_report_root"] = lambda: tmp_path.resolve()
    monkeypatch.setenv("QT_QPA_PLATFORM", "windows")

    result = launch(["--_gridguard-frozen-smoke-report", str(report_path.resolve())])

    assert result == 17
    assert captured == {
        "path": report_path.resolve(),
        "qt_platform": "offscreen",
    }


def test_frozen_launcher_rejects_relative_or_non_json_smoke_paths(tmp_path: Path) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    validate = namespace["_validated_smoke_report_path"]

    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    for invalid_path in (
        "frozen-smoke.json",
        str(allowed_root / "frozen-smoke.txt"),
        str(tmp_path / "frozen-smoke.json"),
    ):
        with pytest.raises(ValueError):
            validate(invalid_path, allowed_root=allowed_root)


def test_frozen_smoke_root_is_derived_from_the_exact_release_layout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    resolve_root = namespace["_frozen_smoke_report_root"]
    release_root = tmp_path / "release root with spaces"
    executable = release_root / "dist" / "GridGuard Khartoum" / "GridGuard Khartoum.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    expected_root = release_root / "build" / "gridguard"
    expected_root.mkdir(parents=True)
    monkeypatch.setattr(namespace["sys"], "executable", str(executable))
    monkeypatch.setenv("GRIDGUARD_SMOKE_REPORT", str(tmp_path / "ambient"))

    assert resolve_root() == expected_root.resolve()


def test_frozen_smoke_path_must_not_exist_and_report_creation_is_exclusive(
    tmp_path: Path,
) -> None:
    namespace = runpy.run_path(
        str(REPOSITORY_ROOT / "packaging" / "gridguard_launcher.py"),
        run_name="gridguard_launcher_test",
    )
    validate = namespace["_validated_smoke_report_path"]
    write_report = namespace["_write_smoke_report"]
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    report_path = allowed_root / "frozen-smoke.json"

    validated = validate(str(report_path), allowed_root=allowed_root)
    write_report(validated, {"status": "PASS"})
    accepted_bytes = report_path.read_bytes()

    with pytest.raises(FileExistsError):
        write_report(validated, {"status": "FAIL"})
    with pytest.raises(ValueError, match="must not already exist"):
        validate(str(report_path), allowed_root=allowed_root)
    assert report_path.read_bytes() == accepted_bytes
    assert accepted_bytes == b'{\n  "status": "PASS"\n}\n'
    assert json.loads(accepted_bytes) == {"status": "PASS"}


@pytest.mark.skipif(sys.platform != "win32", reason="requires Windows Start-Process semantics")
def test_frozen_smoke_argument_quoting_preserves_exactly_two_child_arguments(
    tmp_path: Path,
) -> None:
    space_root = tmp_path / "process probe with spaces"
    space_root.mkdir()
    report_path = space_root / "frozen-smoke.json"
    capture_path = space_root / "captured arguments.json"
    probe_path = space_root / "argument recorder.py"
    powershell_path = space_root / "argument probe.ps1"
    probe_path.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps(sys.argv[2:]), encoding='utf-8')\n",
        encoding="utf-8",
    )
    powershell_path.write_text(
        """param(
    [string]$PythonExecutable,
    [string]$ProbePath,
    [string]$CapturePath,
    [string]$ReportPath
)
$childArguments = @(
    ('"{0}"' -f $ProbePath),
    ('"{0}"' -f $CapturePath),
    "--_gridguard-frozen-smoke-report",
    ('"{0}"' -f $ReportPath)
)
$process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $childArguments `
    -NoNewWindow `
    -Wait `
    -PassThru
exit $process.ExitCode
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(powershell_path),
            "-PythonExecutable",
            sys.executable,
            "-ProbePath",
            str(probe_path),
            "-CapturePath",
            str(capture_path),
            "-ReportPath",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    captured = json.loads(capture_path.read_text(encoding="utf-8"))
    assert captured == ["--_gridguard-frozen-smoke-report", str(report_path)]


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
