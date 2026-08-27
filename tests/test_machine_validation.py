from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.machine_validation import (
    COMMAND_KEYS,
    TRANSIENT_INPUT_SCOPE,
    MachineValidationError,
    canonical_machine_validation_bytes,
    collect_pytest_inventory,
    expected_machine_validation_commands,
    main,
    refresh_release_machine_validation,
    verify_release_machine_validation,
)
from tools.release_validation import (
    APPLICATION_SCREENSHOT_PATHS,
    refresh_release_visual_review,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = Path("docs/release-machine-validation.json")
VISUAL_RECORD_PATH = Path("docs/release-visual-review.json")
PDF_PATH = Path("output/pdf/gridguard-khartoum-final-report.pdf")
PRESENTATION_PATH = Path("output/presentation/gridguard-khartoum-defense.pptx")
PAIRED_PATH = Path("data/evidence/paired-benchmark.json")
ROBUSTNESS_PATH = Path("data/evidence/robustness-supplement.json")
UV_LOCK_PATH = Path("uv.lock")
SOURCE_GIT_COMMIT = "1" * 40
JUNIT_PATH = Path("tmp/release-validation/release-tests.xml")
COLLECTION_PATH = Path("tmp/release-validation/release-tests.collection.json")
PAIRED_REPRODUCTION_PATH = Path("tmp/release-validation/paired-benchmark-reproduced.json")
ROBUSTNESS_REPRODUCTION_PATH = Path("tmp/release-validation/robustness-supplement-reproduced.json")


def _copy_release_input(relative_path: Path, destination_root: Path) -> None:
    destination = destination_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPOSITORY_ROOT / relative_path, destination)


def _write_junit(
    path: Path,
    *,
    declared_tests: int = 2,
    result_element: str | None = None,
    testcases: tuple[tuple[str, str], ...] = (
        ("tests.test_alpha", "test_zulu"),
        ("tests.test_beta", "test_alpha"),
    ),
) -> None:
    result = "" if result_element is None else f"<{result_element} />"
    testcase_xml = "".join(
        (f'<testcase classname="{classname}" name="{name}" time="0.01">{result}</testcase>')
        for classname, name in testcases
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<testsuites name="pytest tests">'
            '<testsuite name="pytest" errors="0" failures="0" skipped="0" '
            f'tests="{declared_tests}" time="0.1">'
            f"{testcase_xml}"
            "</testsuite></testsuites>"
        ),
        encoding="utf-8",
    )


def _write_collection(
    path: Path,
    testcase_ids: tuple[str, ...] = (
        "tests.test_alpha::test_zulu",
        "tests.test_beta::test_alpha",
    ),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": "gridguard.pytest_collection_inventory",
        "schema_version": 1,
        "selection": {
            "ignore": [],
            "marker_expression": "not source_archive and not release_artifact",
        },
        "testcase_ids": sorted(testcase_ids),
        "tests": len(testcase_ids),
    }
    path.write_bytes(canonical_machine_validation_bytes(payload))


def _workspace(tmp_path: Path) -> dict[str, Path]:
    for relative_path in (
        PDF_PATH,
        PRESENTATION_PATH,
        PAIRED_PATH,
        ROBUSTNESS_PATH,
        UV_LOCK_PATH,
        *APPLICATION_SCREENSHOT_PATHS,
    ):
        _copy_release_input(relative_path, tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)
    junit = tmp_path / JUNIT_PATH
    collection = tmp_path / COLLECTION_PATH
    paired_reproduction = tmp_path / PAIRED_REPRODUCTION_PATH
    robustness_reproduction = tmp_path / ROBUSTNESS_REPRODUCTION_PATH
    _write_junit(junit)
    _write_collection(collection)
    paired_reproduction.parent.mkdir(parents=True, exist_ok=True)
    paired_reproduction.write_bytes((tmp_path / PAIRED_PATH).read_bytes())
    robustness_reproduction.write_bytes((tmp_path / ROBUSTNESS_PATH).read_bytes())
    return {
        "record": tmp_path / RECORD_PATH,
        "junit": junit,
        "collection": collection,
        "paired_reproduction": paired_reproduction,
        "robustness_reproduction": robustness_reproduction,
    }


def _exit_codes() -> dict[str, int]:
    return dict.fromkeys(COMMAND_KEYS, 0)


def _refresh(tmp_path: Path, paths: dict[str, Path]) -> dict[str, object]:
    return refresh_release_machine_validation(
        tmp_path,
        record_path=paths["record"],
        junit_path=paths["junit"],
        collection_path=paths["collection"],
        paired_reproduction_path=paths["paired_reproduction"],
        robustness_reproduction_path=paths["robustness_reproduction"],
        command_exit_codes=_exit_codes(),
        source_git_commit=SOURCE_GIT_COMMIT,
    )


def test_refresh_writes_canonical_self_contained_release_record(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)

    record = _refresh(tmp_path, paths)

    assert paths["record"].read_bytes() == canonical_machine_validation_bytes(record)
    assert record["release"] == {
        "date": "2026-08-27",
        "product": "GridGuard Khartoum",
        "release_id": "gridguard-khartoum-v1.2.0",
        "version": "1.2.0",
    }
    assert record["source"] == {
        "git_commit": SOURCE_GIT_COMMIT,
        "uv_lock": {
            "path": UV_LOCK_PATH.as_posix(),
            "sha256": hashlib.sha256((tmp_path / UV_LOCK_PATH).read_bytes()).hexdigest(),
            "size_bytes": (tmp_path / UV_LOCK_PATH).stat().st_size,
        },
    }
    assert record["junit"]["tests"] == 2
    assert record["junit"]["testcase_ids"] == [
        "tests.test_alpha::test_zulu",
        "tests.test_beta::test_alpha",
    ]
    assert record["junit"]["failures"] == 0
    assert record["junit"]["errors"] == 0
    assert record["junit"]["skips"] == 0
    assert record["pytest_collection"]["testcase_ids"] == record["junit"]["testcase_ids"]
    assert record["evidence_reproductions"]["paired"]["byte_identical"] is True
    assert record["evidence_reproductions"]["robustness"]["byte_identical"] is True
    assert record["visual_review"]["pdf"]["page_count"] == 38
    assert record["visual_review"]["presentation"]["slide_count"] == 13
    assert record["visual_review"]["presentation"]["speaker_note_slide_count"] == 13
    assert record["assurance_scope"]["cryptographic_authenticity_claimed"] is False
    assert record["assurance_scope"]["human_execution_claimed"] is False
    assert record["assurance_scope"]["package_runtime_binding"] == (
        "RELEASE_MANIFEST.windows_build"
    )
    assert record["assurance_scope"]["package_runtime_scope"] == (
        "Post-build release-manifest and release_artifact gates, package inventory, "
        "and frozen-runtime facts are separately executed and bound by the final "
        "build workflow and RELEASE_MANIFEST.windows_build."
    )
    assert verify_release_machine_validation(tmp_path, record_path=paths["record"]) == record
    assert (
        verify_release_machine_validation(
            tmp_path,
            record_path=paths["record"],
            verify_inputs=True,
        )
        == record
    )


def test_refresh_rejects_passing_junit_subset_of_exact_collection(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    _write_junit(
        paths["junit"],
        declared_tests=1,
        testcases=(("tests.test_alpha", "test_zulu"),),
    )

    with pytest.raises(MachineValidationError, match="exact pytest collection"):
        _refresh(tmp_path, paths)


def test_verify_rejects_a_stale_lockfile_source_binding(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    _refresh(tmp_path, paths)
    (tmp_path / UV_LOCK_PATH).write_bytes(b"stale lockfile")

    with pytest.raises(MachineValidationError, match="uv.lock.*stale"):
        verify_release_machine_validation(tmp_path, record_path=paths["record"])


@pytest.mark.parametrize(
    "source_git_commit",
    ["", "1" * 39, "A" * 40, "g" * 40],
)
def test_refresh_rejects_an_invalid_source_commit(
    tmp_path: Path,
    source_git_commit: str,
) -> None:
    paths = _workspace(tmp_path)

    with pytest.raises(MachineValidationError, match="source Git commit"):
        refresh_release_machine_validation(
            tmp_path,
            record_path=paths["record"],
            collection_path=paths["collection"],
            junit_path=paths["junit"],
            paired_reproduction_path=paths["paired_reproduction"],
            robustness_reproduction_path=paths["robustness_reproduction"],
            command_exit_codes=_exit_codes(),
            source_git_commit=source_git_commit,
        )


def test_expected_commands_are_exact_and_bound_to_normalized_input_paths(
    tmp_path: Path,
) -> None:
    paths = _workspace(tmp_path)

    commands = expected_machine_validation_commands(
        tmp_path,
        collection_path=paths["collection"],
        junit_path=paths["junit"],
        paired_reproduction_path=paths["paired_reproduction"],
        robustness_reproduction_path=paths["robustness_reproduction"],
    )

    assert tuple(commands) == COMMAND_KEYS
    assert commands == {
        "locked_sync": "uv sync --locked --extra dev --extra docs --offline",
        "ruff_check": "uv run --no-sync ruff check .",
        "ruff_format_check": "uv run --no-sync ruff format --check .",
        "cuda_doctor": "uv run --no-sync gridguard-doctor",
        "evidence_verification": "uv run --no-sync gridguard-evaluate --verify-evidence",
        "paired_reproduction": (
            "uv run --no-sync gridguard-evaluate --reproduce-paired --output "
            '"tmp/release-validation/paired-benchmark-reproduced.json"'
        ),
        "robustness_reproduction": (
            "uv run --no-sync gridguard-evaluate --reproduce-robustness --output "
            '"tmp/release-validation/robustness-supplement-reproduced.json"'
        ),
        "source_pytest_collection": (
            "uv run --no-sync python tools/machine_validation.py "
            "--collect-pytest-inventory --collection-path "
            '"tmp/release-validation/release-tests.collection.json"'
        ),
        "source_pytest": (
            "uv run --no-sync pytest -q --junitxml "
            '"tmp/release-validation/release-tests.xml" '
            '-m "not source_archive and not release_artifact"'
        ),
    }


def test_collection_contract_includes_non_artifact_release_manifest_tests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_pytest_main(arguments, *, plugins):
        captured["arguments"] = arguments
        plugins[0].pytest_collection_finish(
            SimpleNamespace(
                items=[
                    SimpleNamespace(
                        nodeid=(
                            "tests/test_release_manifest.py::"
                            "test_release_manifest_parser_rejects_duplicate_keys"
                        )
                    )
                ]
            )
        )
        return 0

    monkeypatch.setattr(pytest, "main", fake_pytest_main)
    collection_path = tmp_path / "tmp" / "release-tests.collection.json"

    payload = collect_pytest_inventory(tmp_path, collection_path=collection_path)

    assert payload["selection"] == {
        "ignore": [],
        "marker_expression": "not source_archive and not release_artifact",
    }
    assert payload["testcase_ids"] == [
        "tests.test_release_manifest::test_release_manifest_parser_rejects_duplicate_keys"
    ]
    assert not any(str(argument).startswith("--ignore=") for argument in captured["arguments"])


@pytest.mark.parametrize(
    "environment_name",
    [
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
        "PYTEST_FUTURE_SELECTION_CONTROL",
    ],
)
def test_collection_rejects_ambient_pytest_controls(
    tmp_path: Path,
    monkeypatch,
    environment_name: str,
) -> None:
    monkeypatch.setenv(environment_name, "-k narrowed")

    with pytest.raises(MachineValidationError, match="ambient pytest"):
        collect_pytest_inventory(
            tmp_path,
            collection_path=tmp_path / "tmp" / "release-tests.collection.json",
        )


def test_expected_commands_reject_powershell_metacharacters_in_paths(
    tmp_path: Path,
) -> None:
    paths = _workspace(tmp_path)

    with pytest.raises(MachineValidationError, match="safe command-path characters"):
        expected_machine_validation_commands(
            tmp_path,
            collection_path=paths["collection"],
            junit_path=tmp_path / "tmp/$(Get-ChildItem)/release-tests.xml",
            paired_reproduction_path=paths["paired_reproduction"],
            robustness_reproduction_path=paths["robustness_reproduction"],
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda values: values | {"ruff_check": 1}, "exit code"),
        (
            lambda values: {key: value for key, value in values.items() if key != "ruff_check"},
            "missing",
        ),
        (lambda values: values | {"invented": 0}, "unknown"),
        (lambda values: values | {"ruff_check": True}, "integer"),
    ],
)
def test_refresh_rejects_untrustworthy_command_results(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    paths = _workspace(tmp_path)

    with pytest.raises(MachineValidationError, match=match):
        refresh_release_machine_validation(
            tmp_path,
            record_path=paths["record"],
            collection_path=paths["collection"],
            junit_path=paths["junit"],
            paired_reproduction_path=paths["paired_reproduction"],
            robustness_reproduction_path=paths["robustness_reproduction"],
            command_exit_codes=mutate(_exit_codes()),
            source_git_commit=SOURCE_GIT_COMMIT,
        )


@pytest.mark.parametrize(
    ("declared_tests", "result_element", "match"),
    [
        (3, None, "declared totals"),
        (2, "failure", "failures"),
        (2, "error", "errors"),
        (2, "skipped", "skips"),
    ],
)
def test_refresh_rejects_falsified_or_nonpassing_junit(
    tmp_path: Path,
    declared_tests: int,
    result_element: str | None,
    match: str,
) -> None:
    paths = _workspace(tmp_path)
    _write_junit(
        paths["junit"],
        declared_tests=declared_tests,
        result_element=result_element,
    )

    with pytest.raises(MachineValidationError, match=match):
        _refresh(tmp_path, paths)


def test_refresh_rejects_nonidentical_reproduction_and_path_escape(
    tmp_path: Path,
) -> None:
    paths = _workspace(tmp_path)
    paths["paired_reproduction"].write_bytes(b"not the paired evidence")

    with pytest.raises(MachineValidationError, match="byte-identical"):
        _refresh(tmp_path, paths)

    escaped = tmp_path.parent / "escaped-reproduction.json"
    escaped.write_bytes((tmp_path / PAIRED_PATH).read_bytes())
    paths["paired_reproduction"] = escaped
    with pytest.raises(MachineValidationError, match="inside the repository"):
        _refresh(tmp_path, paths)


def test_default_verify_is_self_contained_for_transient_inputs(
    tmp_path: Path,
) -> None:
    paths = _workspace(tmp_path)
    record = _refresh(tmp_path, paths)
    paths["junit"].unlink()
    paths["paired_reproduction"].unlink()
    paths["robustness_reproduction"].unlink()

    assert verify_release_machine_validation(tmp_path, record_path=paths["record"]) == record
    with pytest.raises(MachineValidationError, match="transient input"):
        verify_release_machine_validation(
            tmp_path,
            record_path=paths["record"],
            verify_inputs=True,
        )


def test_transient_input_scope_does_not_promise_nondeterministic_junit_recreation() -> None:
    assert "retained or freshly generated transient files" in TRANSIENT_INPUT_SCOPE
    assert "JUnit testcase IDs and outcomes are compared semantically" in TRANSIENT_INPUT_SCOPE
    assert "rerun the recorded commands before using" not in TRANSIENT_INPUT_SCOPE


def test_verify_inputs_ignores_only_nondeterministic_junit_bytes(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    record = _refresh(tmp_path, paths)
    junit_bytes = paths["junit"].read_bytes()
    paths["junit"].write_bytes(junit_bytes.replace(b'time="0.1"', b'time="9.9"'))

    assert (
        verify_release_machine_validation(
            tmp_path,
            record_path=paths["record"],
            verify_inputs=True,
        )
        == record
    )


def test_verify_inputs_rejects_stale_junit_and_reproduction_bytes(
    tmp_path: Path,
) -> None:
    paths = _workspace(tmp_path)
    _refresh(tmp_path, paths)
    _write_collection(
        paths["collection"],
        testcase_ids=("tests.test_alpha::test_zulu",),
    )

    with pytest.raises(MachineValidationError, match="pytest collection"):
        verify_release_machine_validation(
            tmp_path,
            record_path=paths["record"],
            verify_inputs=True,
        )

    _write_collection(paths["collection"])
    _write_junit(paths["junit"], declared_tests=1)

    with pytest.raises(MachineValidationError, match="JUnit"):
        verify_release_machine_validation(
            tmp_path,
            record_path=paths["record"],
            verify_inputs=True,
        )

    _write_junit(paths["junit"])
    paths["robustness_reproduction"].write_bytes(b"stale")
    with pytest.raises(MachineValidationError, match="robustness reproduction"):
        verify_release_machine_validation(
            tmp_path,
            record_path=paths["record"],
            verify_inputs=True,
        )


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("unknown", "unknown fields"),
        ("release", "release version"),
        ("testcase_total", "testcase"),
        ("command", "command"),
    ],
)
def test_verify_rejects_semantically_tampered_canonical_records(
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    paths = _workspace(tmp_path)
    record = _refresh(tmp_path, paths)
    if tamper == "unknown":
        record["invented"] = True
    elif tamper == "release":
        record["release"]["version"] = "9.9.9"
    elif tamper == "testcase_total":
        record["junit"]["tests"] = 999
    else:
        record["commands"]["ruff_check"]["command"] = "ruff check ."
    paths["record"].write_bytes(canonical_machine_validation_bytes(record))

    with pytest.raises(MachineValidationError, match=match):
        verify_release_machine_validation(tmp_path, record_path=paths["record"])


@pytest.mark.parametrize("tamper", ["noncanonical", "duplicate", "nonfinite"])
def test_verify_rejects_noncanonical_duplicate_or_nonfinite_json(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = _workspace(tmp_path)
    record = _refresh(tmp_path, paths)
    if tamper == "noncanonical":
        encoded = json.dumps(record, sort_keys=True).encode()
    elif tamper == "duplicate":
        encoded = (
            paths["record"]
            .read_bytes()
            .replace(
                b'"schema_name":',
                b'"schema_name": "duplicate",\n  "schema_name":',
                1,
            )
        )
    else:
        encoded = (
            paths["record"]
            .read_bytes()
            .replace(
                b'"schema_version": 3',
                b'"schema_version": NaN',
                1,
            )
        )
    paths["record"].write_bytes(encoded)

    with pytest.raises(MachineValidationError):
        verify_release_machine_validation(tmp_path, record_path=paths["record"])


def test_verify_rejects_missing_or_stale_committed_inputs(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    _refresh(tmp_path, paths)
    (tmp_path / PAIRED_PATH).unlink()

    with pytest.raises(MachineValidationError, match="paired canonical"):
        verify_release_machine_validation(tmp_path, record_path=paths["record"])

    _copy_release_input(PAIRED_PATH, tmp_path)
    (tmp_path / PDF_PATH).write_bytes(b"stale PDF")
    with pytest.raises(MachineValidationError, match="visual-review"):
        verify_release_machine_validation(tmp_path, record_path=paths["record"])


def test_verify_rejects_a_record_that_aliases_its_transient_junit_path(
    tmp_path: Path,
) -> None:
    paths = _workspace(tmp_path)
    record = _refresh(tmp_path, paths)
    record["junit"]["path"] = RECORD_PATH.as_posix()
    record["commands"]["source_pytest"]["command"] = (
        "uv run --no-sync pytest -q --junitxml "
        '"docs/release-machine-validation.json" '
        '-m "not source_archive and not release_artifact"'
    )
    paths["record"].write_bytes(canonical_machine_validation_bytes(record))

    with pytest.raises(MachineValidationError, match="record path must be distinct"):
        verify_release_machine_validation(tmp_path, record_path=paths["record"])


def test_cli_refresh_and_both_verification_modes(tmp_path: Path, capsys) -> None:
    paths = _workspace(tmp_path)
    common = [
        "--repository-root",
        str(tmp_path),
        "--record-path",
        str(paths["record"]),
    ]
    refresh_args = [
        "--refresh",
        *common,
        "--collection-path",
        str(paths["collection"]),
        "--junit-path",
        str(paths["junit"]),
        "--paired-reproduction-path",
        str(paths["paired_reproduction"]),
        "--robustness-reproduction-path",
        str(paths["robustness_reproduction"]),
        "--source-git-commit",
        SOURCE_GIT_COMMIT,
    ]
    for key in COMMAND_KEYS:
        refresh_args.extend(("--command-exit-code", f"{key}=0"))

    assert main(refresh_args) == 0
    assert "unsigned" in capsys.readouterr().out.lower()
    assert main(["--verify", *common]) == 0
    assert "self-contained" in capsys.readouterr().out.lower()
    assert main(["--verify", "--verify-inputs", *common]) == 0
    assert "transient" in capsys.readouterr().out.lower()

    paths["junit"].unlink()
    assert main(["--verify", "--verify-inputs", *common]) == 1
    assert "failed" in capsys.readouterr().err.lower()
