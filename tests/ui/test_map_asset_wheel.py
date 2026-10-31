from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], *, cwd: Path, environment: dict[str, str]) -> None:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


def test_built_wheel_loads_geographic_map_and_attribution_outside_source_tree(
    tmp_path: Path,
) -> None:
    uv = shutil.which("uv")
    assert uv is not None, "uv is required for the wheel smoke test"

    wheel_directory = tmp_path / "wheel"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PIP_NO_INDEX"] = "1"
    environment["UV_CACHE_DIR"] = str(tmp_path / "empty-uv-cache")
    _run(
        [
            uv,
            "run",
            "--no-sync",
            "--offline",
            "--",
            uv,
            "build",
            "--wheel",
            "--offline",
            "--no-index",
            "--no-build-isolation",
            "--out-dir",
            str(wheel_directory),
        ],
        cwd=REPOSITORY_ROOT,
        environment=environment,
    )
    wheels = tuple(wheel_directory.glob("*.whl"))
    assert len(wheels) == 1

    virtual_environment = tmp_path / "installed"
    _run(
        [sys.executable, "-m", "venv", str(virtual_environment)],
        cwd=tmp_path,
        environment=environment,
    )
    python = virtual_environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
        cwd=tmp_path,
        environment=environment,
    )
    _run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import hashlib; "
                "import json; "
                "from importlib.resources import files; "
                "from gridguard.robustness_provenance import "
                "ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS; "
                "from gridguard.ui import load_map_asset; "
                "from gridguard.provenance import verify_project_source_manifest; "
                "from gridguard.resources import resource_path, runtime_root; "
                "asset = load_map_asset(); "
                "assert asset.map_id == 'khartoum-locality-boundaries-2020'; "
                "assert asset.schema_version == 2; "
                "assert len(asset.localities) == 7; "
                "assert asset.projection.crs == 'EPSG:3857'; "
                "map_root = files('gridguard').joinpath('assets', 'map'); "
                "assert map_root.joinpath('khartoum-localities.geojson').is_file(); "
                "assert map_root.joinpath('khartoum-nile.geojson').is_file(); "
                "assert map_root.joinpath('map-provenance.json').is_file(); "
                "assert map_root.joinpath('THIRD_PARTY_DATA.md').is_file(); "
                "root = runtime_root(); "
                "manifest = resource_path('data', 'evidence', 'source-manifest.json'); "
                "assert resource_path('src', 'gridguard', "
                "'robustness_provenance.py').is_file(); "
                "assert len(verify_project_source_manifest(manifest, repository_root=root)) == 8; "
                "robustness = json.loads(resource_path('data', 'evidence', "
                "'robustness-supplement.json').read_text(encoding='utf-8')); "
                "sources = robustness['provenance']['implementation_sources']; "
                "expected_sources = [dict(path=path, sha256=sha256, size_bytes=size_bytes) "
                "for path, sha256, size_bytes in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS]; "
                "assert sources == expected_sources; "
                "assert all((raw_path := resource_path('data', 'evidence', 'accepted-sources', "
                "*source_path.split('/'))).is_file() "
                "and raw_path.stat().st_size == size_bytes "
                "and hashlib.sha256(raw_path.read_bytes()).hexdigest() == sha256 "
                "for source_path, sha256, size_bytes "
                "in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS); "
                "from gridguard.focused_results import load_focused_results; "
                "current = load_focused_results(); "
                "assert len(current.means.rows) == 5; "
                "assert current.means.rows[1][-1] == '97.02%'; "
                "assert len(current.evidence) == 3; "
                "assert resource_path('data', 'model', 'ppo-policy.npz').is_file(); "
                "deployed = resource_path('data', 'model', 'focused-ppo-study', "
                "'seed-8111-selected.npz'); "
                "assert hashlib.sha256(deployed.read_bytes()).hexdigest() == "
                "'9c3dbed2cdd979bdd566004ecaada8c78a3aef26397eb7e90a2b953d1da4f2d3'; "
                "assert resource_path('docs', 'source', "
                "'graduation-project-proposal.pdf').is_file()"
            ),
        ],
        cwd=tmp_path,
        environment=environment,
    )
