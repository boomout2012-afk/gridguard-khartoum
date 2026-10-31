from __future__ import annotations

import hashlib
import json
import os
import runpy
from pathlib import Path

import pytest

import gridguard.cli as cli
from gridguard.robustness_provenance import ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("data/evidence/accepted-sources")


def test_packaging_dll_search_excludes_ambient_document_tools(monkeypatch, tmp_path: Path) -> None:
    from tests.test_packaging import _load_pyinstaller_spec_namespace

    monkeypatch.setenv("PATH", os.environ["PATH"])
    namespace = _load_pyinstaller_spec_namespace(monkeypatch, tmp_path / "torch")
    environment = tmp_path / "python"
    windows = tmp_path / "Windows"
    scripts = environment / "Scripts"
    system32 = windows / "System32"
    poppler = tmp_path / "document-tools/poppler/bin"
    lookalike = tmp_path / "python-untrusted"
    for path in (scripts, system32, poppler, lookalike):
        path.mkdir(parents=True)
    inherited = os.pathsep.join(
        [str(poppler), str(scripts), "", str(lookalike), "relative", str(system32), str(scripts)]
    )
    assert namespace["_trusted_build_dll_path"](
        inherited, (environment, windows)
    ) == os.pathsep.join([str(scripts), str(system32)])


def test_preserved_frozen_sources_have_every_original_byte() -> None:
    for relative, digest, size in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS:
        encoded = (ROOT / ARCHIVE / relative).read_bytes()
        assert len(encoded) == size
        assert hashlib.sha256(encoded).hexdigest() == digest


@pytest.mark.parametrize("preserved", [False, True])
def test_robustness_startup_uses_preserved_root_when_present(
    monkeypatch, tmp_path: Path, preserved: bool
) -> None:
    archive = tmp_path / ARCHIVE
    if preserved:
        archive.mkdir(parents=True)
    roots = []
    monkeypatch.setattr(cli, "verify_robustness_implementation_sources", roots.append)
    monkeypatch.setattr(
        cli,
        "verify_robustness_supplement",
        lambda _path: {"schema": {"version": cli.ROBUSTNESS_SCHEMA_VERSION}},
    )
    monkeypatch.setattr(cli, "verify_robustness_inference_on_cuda", lambda _payload: None)
    cli._verify_robustness_evidence(tmp_path)
    assert roots == [archive if preserved else tmp_path]


def test_proposal_startup_verifies_bytes_sources_and_cuda_intervals(monkeypatch) -> None:
    events = []
    payload = {"test": True}
    monkeypatch.setattr(
        cli, "load_proposal_supplement", lambda path: events.append(path) or payload
    )
    monkeypatch.setattr(cli, "verify_proposal_implementation_sources", events.append)
    monkeypatch.setattr(cli, "verify_proposal_inference_on_cuda", events.append)
    assert cli._verify_proposal_evidence(ROOT) is True
    assert events == [ROOT / "data/evidence/proposal-supplement.json", ROOT, payload]


def test_proposal_startup_legacy_without_supplement_remains_supported(tmp_path: Path) -> None:
    assert cli._verify_proposal_evidence(tmp_path) is False


@pytest.mark.parametrize("stage", ["load", "source", "cuda"])
def test_proposal_startup_fails_closed_on_each_verification_stage(monkeypatch, stage) -> None:
    def fail(_value):
        raise ValueError("invalid proposal evidence")

    monkeypatch.setattr(cli, "load_proposal_supplement", lambda _path: {})
    monkeypatch.setattr(cli, "verify_proposal_implementation_sources", lambda _root: None)
    monkeypatch.setattr(cli, "verify_proposal_inference_on_cuda", lambda _payload: None)
    name = {
        "load": "load_proposal_supplement",
        "source": "verify_proposal_implementation_sources",
        "cuda": "verify_proposal_inference_on_cuda",
    }[stage]
    monkeypatch.setattr(cli, name, fail)
    with pytest.raises(cli.EvaluationArtifactError, match="invalid proposal evidence"):
        cli._verify_proposal_evidence(ROOT)


def test_build_hook_bundles_historical_and_current_raw_sources() -> None:
    namespace = runpy.run_path(str(ROOT / "hatch_build.py"))
    sources = dict(namespace["_validated_provenance_sources"](ROOT))
    assert len(sources) == len(set(sources))
    for relative, _digest, _size in ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS:
        assert sources[ARCHIVE / relative] == ROOT / ARCHIVE / relative
        assert sources[Path(relative)] == ROOT / relative
    proposal = json.loads((ROOT / "data/evidence/proposal-supplement.json").read_text())
    for relative in proposal["source_sha256"]:
        if relative.startswith("src/"):
            assert sources[Path(relative)] == ROOT / relative
    for name in ("proposal_evidence", "proposal_provenance"):
        assert Path(f"src/gridguard/{name}.py") in sources


def test_build_hook_rejects_corrupt_historical_source(tmp_path: Path) -> None:
    import shutil

    shutil.copytree(ROOT / "src/gridguard", tmp_path / "src/gridguard")
    shutil.copytree(ROOT / ARCHIVE, tmp_path / ARCHIVE)
    relative = ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS[0][0]
    (tmp_path / ARCHIVE / relative).write_bytes(b"tampered")
    namespace = runpy.run_path(str(ROOT / "hatch_build.py"))
    with pytest.raises(ValueError, match="provenance source size mismatch"):
        namespace["_validated_provenance_sources"](tmp_path)


def test_frozen_launcher_requires_three_controllers_and_records_optimization() -> None:
    from dataclasses import replace

    from gridguard.live_policy import DEPLOYED_LIVE_POLICY
    from gridguard.ui.live_models import LiveDurationSummary
    from tests.ui.test_live_extension import _three_controller_comparison
    from tests.ui.test_live_page import _comparison

    namespace = runpy.run_path(str(ROOT / "packaging/gridguard_launcher.py"))
    inspect = namespace["_live_smoke_facts"]
    comparison = _three_controller_comparison()
    safe_duration = LiveDurationSummary(
        blackout_locality_hours=0.0,
        critical_interruption_locality_hours=0.0,
        blackout_duration_range_hours=0.0,
        blackout_duration_mean_pairwise_gap_hours=0.0,
        critical_interruption_duration_range_hours=0.0,
        critical_interruption_duration_mean_pairwise_gap_hours=0.0,
        longest_blackout_hours=0.0,
        longest_critical_interruption_hours=0.0,
    )
    comparison = replace(
        comparison,
        shortage_level="moderate",
        seed=50002,
        policy_identity=DEPLOYED_LIVE_POLICY,
        metrics_by_controller={
            name: replace(summary, duration=safe_duration)
            if name in {"PPO", "Optimization"}
            else summary
            for name, summary in comparison.metrics_by_controller.items()
        },
    )
    payload = inspect(comparison)
    assert payload["controllers"] == ["PPO", "Fixed Rotation", "Optimization"]
    assert payload["frame_counts"] == {name: 24 for name in payload["controllers"]}
    assert payload["hour_zero"]["optimization_served_mw"] == pytest.approx(310.0)
    assert payload["deployed_policy"]["training_seed"] == 8111
    assert payload["deployed_policy"]["critical_and_power_checks_passed"] is True
    with pytest.raises(ValueError, match="unexpected controller identities"):
        inspect(replace(_comparison(), shortage_level="moderate", seed=50002))


def test_extended_app_smoke_root_is_bound_to_its_separate_build_directory(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(ROOT / "packaging/gridguard_launcher.py"))
    executable = tmp_path / "output/app/GridGuard Khartoum/GridGuard Khartoum.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    expected_root = tmp_path / "tmp/proposal-extended/build/gridguard"
    expected_root.mkdir(parents=True)
    monkeypatch.setattr(namespace["sys"], "executable", str(executable))
    assert namespace["_frozen_smoke_report_root"]() == expected_root.resolve()


def test_extended_app_smoke_rejects_an_unapproved_output_parent(
    monkeypatch, tmp_path: Path
) -> None:
    namespace = runpy.run_path(str(ROOT / "packaging/gridguard_launcher.py"))
    executable = tmp_path / "unapproved/app/GridGuard Khartoum/GridGuard Khartoum.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(namespace["sys"], "executable", str(executable))
    with pytest.raises(ValueError, match="outside the expected release layout"):
        namespace["_frozen_smoke_report_root"]()


@pytest.mark.parametrize("tamper", [None, "missing", "extra", "power"])
def test_three_controller_smoke_report_contract(tmp_path: Path, tamper: str | None) -> None:
    from tests.test_release_manifest import (
        _canonical_smoke_json,
        _run_frozen_smoke_parser,
        _valid_frozen_smoke_payload,
    )

    payload = _valid_frozen_smoke_payload()
    live = payload["live_scenario"]
    live["controllers"].append("Optimization")
    live["frame_counts"]["Optimization"] = 24
    live["selected_controller"] = "Optimization"
    live["hour_zero"]["optimization_served_mw"] = 60.0
    if tamper == "missing":
        del live["hour_zero"]["optimization_served_mw"]
    elif tamper == "extra":
        live["hour_zero"]["unbound"] = 0.0
    elif tamper == "power":
        live["hour_zero"]["optimization_served_mw"] = 61.0
    path = tmp_path / "smoke.json"
    path.write_bytes(_canonical_smoke_json(payload))
    result = _run_frozen_smoke_parser(path)
    assert (result.returncode == 0) is (tamper is None), result.stderr


@pytest.mark.parametrize("tamper", ["extra", "missing"])
def test_release_policy_rejects_unknown_or_missing_historical_sources(
    tmp_path: Path, tamper
) -> None:
    from tests.test_packaging import (
        _distribution_policy_error,
        _run_distribution_policy,
        _stage_complete_release_tree,
    )

    distribution = tmp_path / "dist/GridGuard Khartoum"
    _stage_complete_release_tree(distribution)
    archive = distribution / "_internal/gridguard/_bundled" / ARCHIVE / "src/gridguard"
    if tamper == "extra":
        (archive / "unbound.py").write_bytes(b"must not ship")
    else:
        (archive / "controllers.py").unlink()
    result = _run_distribution_policy(distribution)
    assert result.returncode != 0
    assert "Exact packaged provenance-source inventory mismatch" in _distribution_policy_error(
        result
    )
