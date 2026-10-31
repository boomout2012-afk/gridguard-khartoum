"""The repeat-capture check compares evidence identity and every rendered page."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "capture_current_app", ROOT / "tools/capture_current_app.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(digest="same"):
    return {
        "policy": {"training_seed": 8111},
        "sources": {"source.py": "source-hash"},
        "pages": [{"filename": "overview.png", "sha256": digest, "size": [1366, 768]}],
    }


def test_repeat_capture_ignores_only_capture_time():
    first = {**_manifest(), "captured_utc": "earlier"}
    second = {**_manifest(), "captured_utc": "later"}
    _module().compare_captures(first, second)


@pytest.mark.parametrize("field", ["pixels", "policy", "sources", "missing", "duplicate"])
def test_repeat_capture_rejects_changed_or_missing_rendered_evidence(field):
    changed = _manifest()
    if field == "pixels":
        changed["pages"][0]["sha256"] = "changed"
    elif field in ("policy", "sources"):
        changed[field] = {"changed": True}
    elif field == "missing":
        changed["pages"] = []
    else:
        changed["pages"] *= 2
    with pytest.raises(ValueError):
        _module().compare_captures(_manifest(), changed)
