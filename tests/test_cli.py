from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import gridguard.cli as cli
from gridguard.evaluation import PAIRED_BENCHMARK_SHA256
from gridguard.gpu import GpuPreflightError, GpuRuntimeInfo

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def _isolate_additive_cuda_gate_in_portable_cli_tests(monkeypatch, request) -> None:
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(cli, "_verify_proposal_evidence", lambda _root: False)


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


def test_quality_portable_selector_explicitly_hides_cuda() -> None:
    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    portable_step = workflow.split("- name: Run portable tests", 1)[1].split("- name:", 1)[0]

    assert 'CUDA_VISIBLE_DEVICES: "-1"' in portable_step
    assert '-m "not gpu and not source_archive and not release_artifact"' in portable_step


def test_doctor_main_reports_gpu_and_evidence(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "run_gpu_preflight", _gpu_info)
    monkeypatch.setattr(cli, "verify_paired_inference_on_cuda", lambda _payload: 0.0)
    monkeypatch.setattr(cli, "_verify_robustness_evidence", lambda _root: {})
    monkeypatch.setattr(cli, "_verify_ppo_and_smoke", lambda _root: (1, 1, 1, 1, 1, 1, 1))

    exit_code = cli.doctor_main(["--repository-root", str(ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out
    assert "NVIDIA Test GPU" in captured.out
    assert "PyTorch 2.8.0+cu128 / CUDA 12.8" in captured.out
    assert "CuPy 14.1.1" in captured.out
    assert "8 source artifacts verified" in captured.out
    assert "Paired benchmark semantics verified" in captured.out
    assert "PPO artifact and CUDA inference verified" in captured.out
    assert captured.err == ""


def test_project_preflight_returns_shared_desktop_startup_state(monkeypatch) -> None:
    paired = {"protocol": {"matched_scenario_count": 288}}
    reconstructed: list[dict] = []
    monkeypatch.setattr(cli, "run_gpu_preflight", _gpu_info)
    monkeypatch.setattr(cli, "_verify_repository_sources", lambda _root: ("a", "b"))
    monkeypatch.setattr(
        cli,
        "_verify_paired_evidence",
        lambda _root: paired,
    )
    monkeypatch.setattr(
        cli,
        "verify_paired_inference_on_cuda",
        lambda payload: reconstructed.append(payload) or 0.0,
    )
    monkeypatch.setattr(cli, "_verify_robustness_evidence", lambda _root: {})
    monkeypatch.setattr(cli, "_verify_ppo_and_smoke", lambda _root: (1, 1, 1, 1, 1, 1, 1))

    result = cli.run_project_preflight(ROOT)

    assert result.gpu == _gpu_info()
    assert result.source_artifact_count == 2
    assert result.matched_scenario_count == 288
    assert result.ppo_smoke_action == (1, 1, 1, 1, 1, 1, 1)
    assert reconstructed == [paired]


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
    monkeypatch.setattr(cli, "verify_paired_inference_on_cuda", lambda _payload: 0.0)
    monkeypatch.setattr(cli, "_verify_robustness_evidence", lambda _root: {})

    def fail_policy(_root: Path) -> tuple[int, ...]:
        raise cli.PolicyArtifactError("actor hash mismatch")

    monkeypatch.setattr(cli, "_verify_ppo_and_smoke", fail_policy)

    exit_code = cli.doctor_main(["--repository-root", str(ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "actor hash mismatch" in captured.err


@pytest.mark.parametrize("failure_stage", ["allocation", "actor"])
def test_ppo_smoke_normalizes_cuda_runtime_failures(
    monkeypatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    class FailingActor:
        def act(self, _observation):
            raise RuntimeError("CUDA actor kernel failed")

    monkeypatch.setattr(cli, "load_ppo_policy", lambda *_args, **_kwargs: FailingActor())
    if failure_stage == "allocation":

        def fail_zeros(*_args, **_kwargs):
            raise RuntimeError("CUDA tensor allocation failed")

        expected_message = "CUDA tensor allocation failed"
        monkeypatch.setattr(torch, "zeros", fail_zeros)
    else:
        expected_message = "CUDA actor kernel failed"
        monkeypatch.setattr(torch, "zeros", lambda *_args, **_kwargs: object())

    with pytest.raises(
        cli.PolicyArtifactError,
        match=rf"PPO CUDA smoke inference failed: {expected_message}",
    ) as raised:
        cli._verify_ppo_and_smoke(tmp_path)

    assert type(raised.value.__cause__) is RuntimeError


def test_ppo_smoke_preserves_policy_artifact_errors(monkeypatch, tmp_path: Path) -> None:
    expected = cli.PolicyArtifactError("actor rejected the trusted artifact")

    class FailingActor:
        def act(self, _observation):
            raise expected

    monkeypatch.setattr(cli, "load_ppo_policy", lambda *_args, **_kwargs: FailingActor())
    monkeypatch.setattr(torch, "zeros", lambda *_args, **_kwargs: object())

    with pytest.raises(cli.PolicyArtifactError) as raised:
        cli._verify_ppo_and_smoke(tmp_path)

    assert raised.value is expected
    assert raised.value.__cause__ is None


@pytest.mark.gpu
def test_evaluate_cli_verifies_frozen_source_evidence(capsys) -> None:
    exit_code = cli.main(["--verify-evidence", "--repository-root", str(ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "8 source artifacts" in captured.out
    assert "288 matched scenario conditions" in captured.out
    assert "PPO artifact verified" in captured.out
    assert captured.err == ""


def test_evaluate_cli_fails_closed_on_paired_semantic_error(monkeypatch, capsys) -> None:
    def fail_paired(_root: Path) -> dict:
        raise cli.EvaluationArtifactError("paired aggregate mismatch")

    monkeypatch.setattr(cli, "run_gpu_preflight", _gpu_info)
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


def test_evaluate_cli_writes_untrusted_paired_candidate(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "paired-candidate.json"
    digest = "c" * 64

    def reproduce(repository_root: Path, output_path: Path) -> str:
        assert repository_root == ROOT
        assert output_path == output
        return digest

    monkeypatch.setattr(cli, "_reproduce_paired_candidate", reproduce)

    exit_code = cli.main(
        [
            "--reproduce-paired-candidate",
            "--output",
            str(output),
            "--repository-root",
            str(ROOT),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Paired untrusted candidate: PASS" in captured.out
    assert digest in captured.out
    assert captured.err == ""


def test_paired_candidate_verifies_its_own_digest_before_anchor_promotion(
    monkeypatch,
    tmp_path,
) -> None:
    output = tmp_path / "paired-candidate.json"
    payload = {"schema_version": 3}
    encoded = b"paired-candidate-v3"
    digest = hashlib.sha256(encoded).hexdigest()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(cli, "PAIRED_BENCHMARK_SHA256", "0" * 64)
    monkeypatch.setattr(cli, "run_gpu_preflight", lambda: _gpu_info())
    monkeypatch.setattr(
        cli,
        "_verify_repository_sources",
        lambda _root: pytest.fail("candidate generation must not require promoted source anchors"),
    )
    monkeypatch.setattr(cli, "run_full_paired_evaluation", lambda **_kwargs: payload)
    monkeypatch.setattr(cli, "canonical_paired_json_bytes", lambda _payload: encoded)
    monkeypatch.setattr(
        cli,
        "write_paired_benchmark_candidate",
        lambda _payload, path: calls.append(("candidate-write", path)) or digest,
    )
    monkeypatch.setattr(
        cli,
        "verify_paired_benchmark_candidate",
        lambda path, *, expected_sha256: (
            calls.append(("candidate-verify", (path, expected_sha256))) or payload
        ),
    )
    monkeypatch.setattr(
        cli,
        "write_paired_benchmark",
        lambda *_args, **_kwargs: pytest.fail("candidate must not use the trusted writer"),
    )
    monkeypatch.setattr(
        cli,
        "verify_paired_benchmark",
        lambda *_args, **_kwargs: pytest.fail("candidate must not use the trusted verifier"),
    )
    monkeypatch.setattr(
        cli,
        "verify_paired_inference_on_cuda",
        lambda verified: calls.append(("cuda", verified)) or 0.0,
    )

    actual = cli._reproduce_paired_candidate(ROOT, output)

    assert actual == digest
    assert calls == [
        ("candidate-write", output.resolve()),
        ("candidate-verify", (output.resolve(), digest)),
        ("cuda", payload),
    ]


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


def test_evaluate_cli_writes_untrusted_robustness_candidate(monkeypatch, tmp_path, capsys) -> None:
    output = tmp_path / "robustness-candidate.json"
    digest = "b" * 64

    def reproduce(repository_root: Path, output_path: Path) -> str:
        assert repository_root == ROOT
        assert output_path == output
        return digest

    monkeypatch.setattr(cli, "_reproduce_robustness_candidate", reproduce)

    exit_code = cli.main(
        [
            "--reproduce-robustness-candidate",
            "--output",
            str(output),
            "--repository-root",
            str(ROOT),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "untrusted candidate" in captured.out
    assert digest in captured.out
    assert captured.err == ""


def test_robustness_candidate_uses_only_untrusted_writer_and_verifier(
    monkeypatch,
    tmp_path,
) -> None:
    output = tmp_path / "candidate.json"
    payload = {"schema": {"version": 2}}
    encoded = b"candidate-v2"
    digest = hashlib.sha256(encoded).hexdigest()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(cli, "ROBUSTNESS_SUPPLEMENT_SHA256", "0" * 64)
    monkeypatch.setattr(cli, "run_gpu_preflight", lambda: _gpu_info())
    monkeypatch.setattr(
        cli,
        "_verify_repository_sources",
        lambda _root: pytest.fail("candidate generation must not require promoted source anchors"),
    )
    monkeypatch.setattr(cli, "run_full_robustness_evaluation", lambda **_kwargs: payload)
    monkeypatch.setattr(cli, "canonical_robustness_json_bytes", lambda _payload: encoded)
    monkeypatch.setattr(
        cli,
        "write_robustness_supplement_candidate",
        lambda _payload, path: calls.append(("candidate-write", path)) or digest,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_supplement_candidate",
        lambda path, *, expected_sha256: (
            calls.append(("candidate-verify", (path, expected_sha256))) or payload
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "write_robustness_supplement",
        lambda *_args, **_kwargs: pytest.fail("candidate must not use the trusted writer"),
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_supplement",
        lambda *_args, **_kwargs: pytest.fail("candidate must not use the trusted verifier"),
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_inference_on_cuda",
        lambda verified: calls.append(("cuda", verified)) or {},
    )

    actual = cli._reproduce_robustness_candidate(ROOT, output)

    assert actual == digest
    assert calls == [
        ("candidate-write", output.resolve()),
        ("candidate-verify", (output.resolve(), digest)),
        ("cuda", payload),
    ]


def test_trusted_robustness_reproduction_uses_only_trusted_writer_and_verifier(
    monkeypatch,
    tmp_path,
) -> None:
    output = tmp_path / "trusted-reproduction.json"
    payload = {"schema": {"version": 2}}
    encoded = b"trusted-v2"
    digest = hashlib.sha256(encoded).hexdigest()
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(cli, "ROBUSTNESS_SUPPLEMENT_SHA256", digest)
    monkeypatch.setattr(cli, "run_gpu_preflight", lambda: _gpu_info())
    monkeypatch.setattr(
        cli,
        "_verify_repository_sources",
        lambda root: calls.append(("sources", root)) or (),
    )
    monkeypatch.setattr(cli, "run_full_robustness_evaluation", lambda **_kwargs: payload)
    monkeypatch.setattr(cli, "canonical_robustness_json_bytes", lambda _payload: encoded)
    monkeypatch.setattr(
        cli,
        "write_robustness_supplement",
        lambda _payload, path: calls.append(("trusted-write", path)) or digest,
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_supplement",
        lambda path: calls.append(("trusted-verify", path)) or payload,
    )
    monkeypatch.setattr(
        cli,
        "write_robustness_supplement_candidate",
        lambda *_args, **_kwargs: pytest.fail("trusted path must not use the candidate writer"),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_supplement_candidate",
        lambda *_args, **_kwargs: pytest.fail("trusted path must not use the candidate verifier"),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_inference_on_cuda",
        lambda verified: calls.append(("cuda", verified)) or {},
    )

    actual = cli._reproduce_robustness_evidence(ROOT, output)

    assert actual == digest
    assert calls == [
        ("sources", ROOT),
        ("trusted-write", output.resolve()),
        ("trusted-verify", output.resolve()),
        ("cuda", payload),
    ]


def test_robustness_reproduction_refuses_to_overwrite_trusted_evidence() -> None:
    trusted = ROOT / "data" / "evidence" / "robustness-supplement.json"

    with pytest.raises(cli.RobustnessArtifactError, match="refuses to overwrite"):
        cli._reproduce_robustness_evidence(ROOT, trusted)


def test_verify_evidence_checks_robustness_when_trust_anchor_is_finalized(
    monkeypatch,
    capsys,
) -> None:
    checked: list[Path] = []
    paired_reconstructions: list[dict] = []
    preflights: list[GpuRuntimeInfo] = []
    monkeypatch.setattr(cli, "ROBUSTNESS_SUPPLEMENT_SHA256", "a" * 64)
    monkeypatch.setattr(
        cli,
        "run_gpu_preflight",
        lambda: preflights.append(_gpu_info()) or _gpu_info(),
    )
    monkeypatch.setattr(cli, "_verify_repository_sources", lambda _root: ("source",))
    paired = {"protocol": {"matched_scenario_count": 288}}
    monkeypatch.setattr(cli, "_verify_paired_evidence", lambda _root: paired)
    monkeypatch.setattr(
        cli,
        "verify_paired_inference_on_cuda",
        lambda payload: paired_reconstructions.append(payload) or 0.0,
    )
    monkeypatch.setattr(cli, "_verify_ppo_host", lambda _root: None)
    monkeypatch.setattr(
        cli,
        "_verify_robustness_evidence",
        lambda root: checked.append(root) or {"schema": {"version": 2}},
    )

    exit_code = cli.main(["--verify-evidence", "--repository-root", str(ROOT)])

    assert exit_code == 0
    assert preflights == [_gpu_info()]
    assert checked == [ROOT]
    assert paired_reconstructions == [paired]
    assert "robustness supplement" in capsys.readouterr().out


def test_robustness_verification_rehashes_sources_and_reconstructs_v2_inference(
    monkeypatch,
    tmp_path,
) -> None:
    digest = "a" * 64
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(cli, "ROBUSTNESS_SUPPLEMENT_SHA256", digest)
    monkeypatch.setattr(
        cli,
        "verify_robustness_implementation_sources",
        lambda root: calls.append(("sources", root)) or (),
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_supplement",
        lambda path: (
            calls.append(("artifact", path)) or {"schema": {"version": 2}, "trusted_sha256": digest}
        ),
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_inference_on_cuda",
        lambda payload: calls.append(("inference", payload["trusted_sha256"])) or {},
    )

    verified = cli._verify_robustness_evidence(tmp_path)

    assert calls == [
        ("sources", tmp_path),
        ("artifact", tmp_path / "data" / "evidence" / "robustness-supplement.json"),
        ("inference", digest),
    ]
    assert verified == {"schema": {"version": 2}, "trusted_sha256": digest}


def test_release_verification_rejects_legacy_robustness_artifact(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(cli, "ROBUSTNESS_SUPPLEMENT_SHA256", "a" * 64)
    monkeypatch.setattr(cli, "verify_robustness_implementation_sources", lambda _root: ())
    monkeypatch.setattr(
        cli,
        "verify_robustness_supplement",
        lambda _path: {"schema": {"version": 1}},
    )
    monkeypatch.setattr(
        cli,
        "verify_robustness_inference_on_cuda",
        lambda _payload: pytest.fail("legacy schema must not claim inference reconstruction"),
    )

    with pytest.raises(cli.RobustnessArtifactError, match="requires schema version 2"):
        cli._verify_robustness_evidence(tmp_path)


def test_evaluate_cli_reports_reconstruction_failure_without_traceback(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "ROBUSTNESS_SUPPLEMENT_SHA256", "a" * 64)
    monkeypatch.setattr(cli, "run_gpu_preflight", _gpu_info)
    monkeypatch.setattr(cli, "_verify_repository_sources", lambda _root: ("source",))
    monkeypatch.setattr(
        cli,
        "_verify_paired_evidence",
        lambda _root: {"protocol": {"matched_scenario_count": 288}},
    )
    monkeypatch.setattr(cli, "verify_paired_inference_on_cuda", lambda _payload: 0.0)
    monkeypatch.setattr(cli, "_verify_ppo_host", lambda _root: None)
    monkeypatch.setattr(
        cli,
        "_verify_robustness_evidence",
        lambda _root: (_ for _ in ()).throw(
            cli.RobustnessArtifactError("bootstrap reconstruction failed: CUDA allocation failed")
        ),
    )

    exit_code = cli.main(["--verify-evidence", "--repository-root", str(ROOT)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "bootstrap reconstruction failed" in captured.err


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
