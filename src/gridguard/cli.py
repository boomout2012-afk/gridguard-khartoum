"""Command-line verification entry points for GridGuard Khartoum."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from gridguard.evaluation import (
    PAIRED_BENCHMARK_SHA256,
    EvaluationArtifactError,
    canonical_paired_json_bytes,
    run_full_paired_evaluation,
    verify_paired_benchmark,
    write_paired_benchmark,
)
from gridguard.gpu import GpuPreflightError, GpuRuntimeInfo, run_gpu_preflight
from gridguard.policy import (
    BUNDLED_ARTIFACT_SHA256,
    OBSERVATION_DIM,
    PolicyArtifactError,
    load_ppo_policy,
    verify_ppo_artifact,
)
from gridguard.provenance import ProvenanceError, verify_project_source_manifest
from gridguard.resources import ResourceLocationError, runtime_root
from gridguard.robustness import (
    ROBUSTNESS_SUPPLEMENT_SHA256,
    RobustnessArtifactError,
    canonical_robustness_json_bytes,
    run_full_robustness_evaluation,
    verify_robustness_implementation_sources,
    verify_robustness_supplement,
    write_robustness_supplement,
)


@dataclass(frozen=True, slots=True)
class ProjectPreflightResult:
    """Verified runtime facts shared by the CLI and desktop startup gate."""

    gpu: GpuRuntimeInfo
    source_artifact_count: int
    matched_scenario_count: int
    ppo_smoke_action: tuple[int, ...]


def run_project_preflight(repository_root: Path | None = None) -> ProjectPreflightResult:
    """Run the complete fail-closed GPU, evidence, provenance, and PPO gate."""

    root = repository_root or _source_repository_root()
    gpu = run_gpu_preflight()
    verified = _verify_repository_sources(root)
    paired = _verify_paired_evidence(root)
    if ROBUSTNESS_SUPPLEMENT_SHA256:
        _verify_robustness_evidence(root)
    action = _verify_ppo_and_smoke(root)
    return ProjectPreflightResult(
        gpu=gpu,
        source_artifact_count=len(verified),
        matched_scenario_count=paired["protocol"]["matched_scenario_count"],
        ppo_smoke_action=action,
    )


def doctor_main(argv: Sequence[str] | None = None) -> int:
    """Validate the required GPU stack and committed source artifacts."""

    parser = argparse.ArgumentParser(prog="gridguard-doctor")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    try:
        preflight = run_project_preflight(args.repository_root)
    except (
        EvaluationArtifactError,
        GpuPreflightError,
        PolicyArtifactError,
        ProvenanceError,
        ResourceLocationError,
        RobustnessArtifactError,
    ) as error:
        print(f"GridGuard preflight: FAIL - {error}", file=sys.stderr)
        return 1

    gpu = preflight.gpu
    print("GridGuard preflight: PASS")
    print(
        f"GPU: {gpu.device_name} | compute {gpu.compute_capability[0]}."
        f"{gpu.compute_capability[1]} | {gpu.total_memory_gib:.1f} GiB"
    )
    print(f"PyTorch {gpu.torch_version} / CUDA {gpu.torch_cuda_version}")
    print(
        f"CuPy {gpu.cupy_version} | runtime {gpu.cupy_runtime_version} | "
        f"driver {gpu.cupy_driver_version}"
    )
    print(f"Evidence: {preflight.source_artifact_count} source artifacts verified")
    print("Paired benchmark semantics verified")
    if ROBUSTNESS_SUPPLEMENT_SHA256:
        print("Robustness supplement semantics verified")
    print("PPO artifact and CUDA inference verified")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Verify or reproduce the committed paired proposal evidence."""

    parser = argparse.ArgumentParser(prog="gridguard-evaluate")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-evidence", action="store_true")
    action.add_argument("--reproduce-paired", action="store_true")
    action.add_argument("--reproduce-robustness", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="output path required with a reproduction action",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    if args.reproduce_paired:
        if args.output is None:
            parser.error("--output is required with --reproduce-paired")
        try:
            repository_root = args.repository_root or _source_repository_root()
            digest = _reproduce_paired_evidence(repository_root, args.output)
        except (
            EvaluationArtifactError,
            GpuPreflightError,
            PolicyArtifactError,
            ProvenanceError,
            ResourceLocationError,
        ) as error:
            print(f"Paired reproduction: FAIL - {error}", file=sys.stderr)
            return 1
        print(f"Paired reproduction: PASS - byte-identical SHA-256 {digest}")
        return 0

    if args.reproduce_robustness:
        if args.output is None:
            parser.error("--output is required with --reproduce-robustness")
        try:
            repository_root = args.repository_root or _source_repository_root()
            digest = _reproduce_robustness_evidence(repository_root, args.output)
        except (
            EvaluationArtifactError,
            GpuPreflightError,
            PolicyArtifactError,
            ProvenanceError,
            ResourceLocationError,
            RobustnessArtifactError,
        ) as error:
            print(f"Robustness reproduction: FAIL - {error}", file=sys.stderr)
            return 1
        print(f"Robustness reproduction: PASS - byte-identical SHA-256 {digest}")
        return 0

    if args.output is not None:
        parser.error("--output is only valid with a reproduction action")

    try:
        repository_root = args.repository_root or _source_repository_root()
        verified = _verify_repository_sources(repository_root)
        paired = _verify_paired_evidence(repository_root)
        robustness_verified = False
        if ROBUSTNESS_SUPPLEMENT_SHA256:
            _verify_robustness_evidence(repository_root)
            robustness_verified = True
        _verify_ppo_host(repository_root)
    except (
        EvaluationArtifactError,
        PolicyArtifactError,
        ProvenanceError,
        ResourceLocationError,
        RobustnessArtifactError,
    ) as error:
        print(f"Evidence verification: FAIL - {error}", file=sys.stderr)
        return 1
    print(
        "Evidence verification: PASS - "
        f"{len(verified)} source artifacts, "
        f"{paired['protocol']['matched_scenario_count']} matched scenario conditions, "
        f"and PPO artifact verified{' with robustness supplement' if robustness_verified else ''}"
    )
    return 0


def _verify_repository_sources(repository_root: Path) -> tuple:
    return verify_project_source_manifest(
        repository_root / "data" / "evidence" / "source-manifest.json",
        repository_root=repository_root,
    )


def _verify_paired_evidence(repository_root: Path) -> dict:
    return verify_paired_benchmark(
        repository_root / "data" / "evidence" / "paired-benchmark.json",
        expected_sha256=PAIRED_BENCHMARK_SHA256,
        frozen_benchmark_path=(repository_root / "data" / "evidence" / "frozen-benchmark.json"),
    )


def _verify_ppo_host(repository_root: Path) -> None:
    verify_ppo_artifact(
        repository_root / "data" / "model" / "manifest.json",
        expected_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
    )


def _verify_robustness_evidence(repository_root: Path) -> dict:
    if not ROBUSTNESS_SUPPLEMENT_SHA256:
        raise RobustnessArtifactError("robustness supplement trust anchor is not finalized")
    verify_robustness_implementation_sources(repository_root)
    return verify_robustness_supplement(
        repository_root / "data" / "evidence" / "robustness-supplement.json",
        expected_sha256=ROBUSTNESS_SUPPLEMENT_SHA256,
    )


def _reproduce_paired_evidence(repository_root: Path, output_path: Path) -> str:
    trusted_path = (repository_root / "data" / "evidence" / "paired-benchmark.json").resolve()
    resolved_output = output_path.expanduser().resolve()
    if resolved_output == trusted_path:
        raise EvaluationArtifactError(
            "paired reproduction refuses to overwrite the trusted evidence artifact"
        )

    run_gpu_preflight()
    _verify_repository_sources(repository_root)
    payload = run_full_paired_evaluation(
        ppo_manifest_path=repository_root / "data" / "model" / "manifest.json",
        expected_ppo_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
        frozen_benchmark_path=repository_root / "data" / "evidence" / "frozen-benchmark.json",
        device="cuda:0",
    )
    encoded = canonical_paired_json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != PAIRED_BENCHMARK_SHA256:
        raise EvaluationArtifactError(
            "reproduced paired evidence differs from the project trust anchor: "
            f"expected {PAIRED_BENCHMARK_SHA256}, found {digest}"
        )
    written_digest = write_paired_benchmark(payload, resolved_output)
    if written_digest != digest:
        raise EvaluationArtifactError("paired evidence changed while it was written")
    verify_paired_benchmark(
        resolved_output,
        expected_sha256=PAIRED_BENCHMARK_SHA256,
        frozen_benchmark_path=(repository_root / "data" / "evidence" / "frozen-benchmark.json"),
    )
    return digest


def _reproduce_robustness_evidence(repository_root: Path, output_path: Path) -> str:
    trusted_path = (repository_root / "data" / "evidence" / "robustness-supplement.json").resolve()
    resolved_output = output_path.expanduser().resolve()
    if resolved_output == trusted_path:
        raise RobustnessArtifactError(
            "robustness reproduction refuses to overwrite the trusted evidence artifact"
        )
    if not ROBUSTNESS_SUPPLEMENT_SHA256:
        raise RobustnessArtifactError("robustness supplement trust anchor is not finalized")

    run_gpu_preflight()
    _verify_repository_sources(repository_root)
    payload = run_full_robustness_evaluation(
        ppo_manifest_path=repository_root / "data" / "model" / "manifest.json",
        expected_ppo_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
        device="cuda:0",
    )
    encoded = canonical_robustness_json_bytes(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != ROBUSTNESS_SUPPLEMENT_SHA256:
        raise RobustnessArtifactError(
            "reproduced robustness evidence differs from the project trust anchor: "
            f"expected {ROBUSTNESS_SUPPLEMENT_SHA256}, found {digest}"
        )
    written_digest = write_robustness_supplement(payload, resolved_output)
    if written_digest != digest:
        raise RobustnessArtifactError("robustness evidence changed while it was written")
    verify_robustness_supplement(
        resolved_output,
        expected_sha256=ROBUSTNESS_SUPPLEMENT_SHA256,
    )
    return digest


def _verify_ppo_and_smoke(repository_root: Path) -> tuple[int, ...]:
    import torch

    actor = load_ppo_policy(
        repository_root / "data" / "model" / "manifest.json",
        expected_artifact_sha256=BUNDLED_ARTIFACT_SHA256,
    )
    observation = torch.zeros(OBSERVATION_DIM, dtype=torch.float32, device="cuda:0")
    action = tuple(int(tier) for tier in actor.act(observation).tolist())
    expected = (1, 1, 1, 1, 1, 1, 1)
    if action != expected:
        raise PolicyArtifactError(
            f"PPO CUDA smoke action does not match the trusted golden action: {action}"
        )
    return action


def _source_repository_root() -> Path:
    return runtime_root()


def format_gpu_runtime(info: GpuRuntimeInfo) -> str:
    """Return a compact GPU identity string for future UI reuse."""

    return (
        f"{info.device_name} | PyTorch {info.torch_version} | "
        f"CUDA {info.torch_cuda_version} | CuPy {info.cupy_version}"
    )
