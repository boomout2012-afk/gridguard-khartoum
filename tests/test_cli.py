from __future__ import annotations

from pathlib import Path

import pytest

import gridguard.cli as cli
from gridguard.evaluation import PAIRED_BENCHMARK_SHA256
from gridguard.gpu import GpuPreflightError, GpuRuntimeInfo

ROOT = Path(__file__).parents[1]


def _gpu_info() -> GpuRuntimeInfo:
    return GpuRuntimeInfo(
        device_name="NVIDIA Test GPU",
        compute_capability=(9, 1),
        total_memory_bytes=8 * 1024**3,
        torch_version="2.8.0+cu128",
        torch_cuda_version="12.8",
        torch_device_count=1,
        cupy_version="14.1.1",
        cupy_device_count=1,
        cupy_runtime_version=12080,
        cupy_driver_version=13000,
    )


def test_doctor_main_reports_gpu_and_evidence(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "run_gpu_preflight", _gpu_info)
    monkeypatch.setattr(cli, "_verify_ppo_and_smoke", lambda _root: (1, 1, 1, 1, 1, 1, 1))

    exit_code = cli.doctor_main(["--repository-root", str(ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out
    assert "NVIDIA Test GPU" in captured.out
    assert "PyTorch 2.8.0+cu128 / CUDA 12.8" in captured.out
    assert "CuPy 14.1.1" in captured.out
    assert "5 source artifacts verified" in captured.out
    assert "Paired benchmark semantics verified" in captured.out
    assert "PPO artifact and CUDA inference verified" in captured.out
    assert captured.err == ""


def test_project_preflight_returns_shared_desktop_startup_state(monkeypatch) -> None:
    monkeypatch.setattr(cli, "run_gpu_preflight", _gpu_info)
    monkeypatch.setattr(cli, "_verify_repository_sources", lambda _root: ("a", "b"))
    monkeypatch.setattr(
        cli,
        "_verify_paired_evidence",
        lambda _root: {"protocol": {"matched_scenario_count": 288}},
    )
    monkeypatch.setattr(cli, "_verify_ppo_and_smoke", lambda _root: (1, 1, 1, 1, 1, 1, 1))

    result = cli.run_project_preflight(ROOT)

    assert result.gpu == _gpu_info()
    assert result.source_artifact_count == 2
    assert result.matched_scenario_count == 288
    assert result.ppo_smoke_action == (1, 1, 1, 1, 1, 1, 1)


def test_doctor_main_fails_closed_on_gpu_error(monkeypatch, capsys) -> None:
    def fail() -> GpuRuntimeInfo:
        raise GpuPreflightError("CUDA unavailable")

    monkeypatch.setattr(cli, "run_gpu_preflight", fail)

    exit_code = cli.doctor_main(["--repository-root", str(ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "FAIL" in captured.err
    assert "CUDA unavailable" in captured.err


def test_doctor_main_fails_closed_on_ppo_verification_error(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "run_gpu_preflight", _gpu_info)

    def fail_policy(_root: Path) -> tuple[int, ...]:
        raise cli.PolicyArtifactError("actor hash mismatch")

    monkeypatch.setattr(cli, "_verify_ppo_and_smoke", fail_policy)

    exit_code = cli.doctor_main(["--repository-root", str(ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "actor hash mismatch" in captured.err


def test_evaluate_cli_verifies_frozen_source_evidence(capsys) -> None:
    exit_code = cli.main(["--verify-evidence", "--repository-root", str(ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "5 source artifacts" in captured.out
    assert "288 matched scenario conditions" in captured.out
    assert "PPO artifact verified" in captured.out
    assert captured.err == ""


def test_evaluate_cli_fails_closed_on_paired_semantic_error(monkeypatch, capsys) -> None:
    def fail_paired(_root: Path) -> dict:
        raise cli.EvaluationArtifactError("paired aggregate mismatch")

    monkeypatch.setattr(cli, "_verify_paired_evidence", fail_paired)

    exit_code = cli.main(["--verify-evidence", "--repository-root", str(ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "paired aggregate mismatch" in captured.err


def test_evaluate_cli_reproduces_paired_benchmark(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "reproduced.json"

    def reproduce(repository_root: Path, output_path: Path) -> str:
        assert repository_root == ROOT
        assert output_path == output
        return PAIRED_BENCHMARK_SHA256

    monkeypatch.setattr(cli, "_reproduce_paired_evidence", reproduce)

    exit_code = cli.main(
        [
            "--reproduce-paired",
            "--output",
            str(output),
            "--repository-root",
            str(ROOT),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Paired reproduction: PASS" in captured.out
    assert PAIRED_BENCHMARK_SHA256 in captured.out
    assert captured.err == ""


def test_evaluate_cli_reproduces_robustness_supplement(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "robustness.json"
    digest = "a" * 64

    def reproduce(repository_root: Path, output_path: Path) -> str:
        assert repository_root == ROOT
        assert output_path == output
        return digest

    monkeypatch.setattr(cli, "_reproduce_robustness_evidence", reproduce)

    exit_code = cli.main(
        [
            "--reproduce-robustness",
            "--output",
            str(output),
            "--repository-root",
            str(ROOT),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Robustness reproduction: PASS" in captured.out
    assert digest in captured.out
    assert captured.err == ""


def test_robustness_reproduction_refuses_to_overwrite_trusted_evidence() -> None:
    trusted = ROOT / "data" / "evidence" / "robustness-supplement.json"

    with pytest.raises(cli.RobustnessArtifactError, match="refuses to overwrite"):
        cli._reproduce_robustness_evidence(ROOT, trusted)


def test_verify_evidence_checks_robustness_when_trust_anchor_is_finalized(
    monkeypatch,
    capsys,
) -> None:
    checked: list[Path] = []
    monkeypatch.setattr(cli, "ROBUSTNESS_SUPPLEMENT_SHA256", "a" * 64)
    monkeypatch.setattr(cli, "_verify_repository_sources", lambda _root: ("source",))
    monkeypatch.setattr(
        cli,
        "_verify_paired_evidence",
        lambda _root: {"protocol": {"matched_scenario_count": 288}},
    )
    monkeypatch.setattr(cli, "_verify_ppo_host", lambda _root: None)
    monkeypatch.setattr(
        cli,
        "_verify_robustness_evidence",
        lambda root: checked.append(root) or {"schema": {"version": 1}},
    )

    exit_code = cli.main(["--verify-evidence", "--repository-root", str(ROOT)])

    assert exit_code == 0
    assert checked == [ROOT]
    assert "robustness supplement" in capsys.readouterr().out


def test_robustness_host_verification_rehashes_current_implementation_sources(
    monkeypatch,
    tmp_path,
) -> None:
    digest = "a" * 64
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(cli, "ROBUSTNESS_SUPPLEMENT_SHA256", digest)
    monkeypatch.setattr(
        cli,
        "verify_robustness_implementation_sources",
        lambda root: calls.append(("sources", root)) or (),
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_supplement",
        lambda path, *, expected_sha256: (
            calls.append(("artifact", path)) or {"expected_sha256": expected_sha256}
        ),
    )

    verified = cli._verify_robustness_evidence(tmp_path)

    assert calls == [
        ("sources", tmp_path),
        ("artifact", tmp_path / "data" / "evidence" / "robustness-supplement.json"),
    ]
    assert verified == {"expected_sha256": digest}


def test_reproduction_refuses_to_overwrite_trusted_evidence() -> None:
    trusted = ROOT / "data" / "evidence" / "paired-benchmark.json"

    with pytest.raises(cli.EvaluationArtifactError, match="refuses to overwrite"):
        cli._reproduce_paired_evidence(ROOT, trusted)


@pytest.mark.gpu
def test_reproduction_is_byte_identical_on_presentation_gpu(tmp_path: Path) -> None:
    output = tmp_path / "reproduced.json"

    digest = cli._reproduce_paired_evidence(ROOT, output)

    assert digest == PAIRED_BENCHMARK_SHA256
    assert (
        output.read_bytes() == (ROOT / "data" / "evidence" / "paired-benchmark.json").read_bytes()
    )


def test_evaluate_cli_reports_missing_default_assets_without_traceback(
    monkeypatch,
    capsys,
) -> None:
    def fail_root() -> Path:
        raise cli.ResourceLocationError("runtime assets missing")

    monkeypatch.setattr(cli, "runtime_root", fail_root)

    exit_code = cli.main(["--verify-evidence"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "runtime assets missing" in captured.err
