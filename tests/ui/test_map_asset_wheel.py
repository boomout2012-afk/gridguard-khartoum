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
    _run(
        [uv, "build", "--wheel", "--out-dir", str(wheel_directory)],
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
                "import json; "
                "from importlib.resources import files; "
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
                "robustness = json.loads(resource_path('data', 'evidence', "
                "'robustness-supplement.json').read_text(encoding='utf-8')); "
                "assert all(resource_path(*item['path'].split('/')).is_file() "
                "for item in robustness['provenance']['implementation_sources']); "
                "assert len(verify_project_source_manifest(manifest, repository_root=root)) == 8; "
                "assert resource_path('data', 'model', 'ppo-policy.npz').is_file(); "
                "assert resource_path('docs', 'source', "
                "'graduation-project-proposal.pdf').is_file()"
            ),
        ],
        cwd=tmp_path,
        environment=environment,
    )
