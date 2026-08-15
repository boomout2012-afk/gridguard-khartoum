from __future__ import annotations

from pathlib import Path

import pytest

from gridguard.resources import ResourceLocationError, resource_path, runtime_root

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_source_runtime_root_resolves_the_project_assets() -> None:
    assert runtime_root() == REPOSITORY_ROOT
    assert resource_path("data", "evidence", "source-manifest.json").is_file()
    assert resource_path("data", "model", "ppo-policy.npz").is_file()
    assert resource_path("docs", "source", "graduation-project-proposal.pdf").is_file()


@pytest.mark.parametrize("unsafe_path", ["../secret", "/absolute", "C:/absolute"])
def test_resource_path_rejects_paths_outside_the_runtime_root(unsafe_path: str) -> None:
    with pytest.raises(ResourceLocationError, match="relative"):
        resource_path(unsafe_path)
