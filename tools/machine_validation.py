"""Create and verify the unsigned release machine-validation summary.

The record is a deterministic reviewer-facing summary of the tested Git/lockfile
snapshot, release command exit codes, reconstructed JUnit results, byte-identical
evidence reproductions, and the separately validated visual-review facts. It is
unsigned and AI-generated: it does not authenticate who ran commands or
independently re-execute them.

The exact pytest collection inventory, JUnit XML, and reproduction outputs are
intentionally transient. Their complete identities and reconstructable facts
are embedded so ordinary verification is self-contained. ``verify_inputs=True``
requires retained or freshly generated workspace files: the deterministic
collection and evidence reproductions are verified byte-for-byte, while JUnit
testcase facts are compared semantically because timing and host fields are not
byte-reproducible. Package inventory and frozen-runtime facts are outside this
source-snapshot-stable record and are bound by ``RELEASE_MANIFEST.windows_build``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

if __package__:
    from tools.release_validation import (
        EXPECTED_PDF_PAGE_COUNT,
        ReleaseVisualReviewError,
        verify_release_visual_review,
    )
else:  # pragma: no cover - exercised by the direct-script CLI in release use
    from release_validation import (  # type: ignore[import-not-found]
        EXPECTED_PDF_PAGE_COUNT,
        ReleaseVisualReviewError,
        verify_release_visual_review,
    )

SCHEMA_NAME = "gridguard.release_machine_validation"
SCHEMA_VERSION = 3
PYTEST_COLLECTION_SCHEMA_NAME = "gridguard.pytest_collection_inventory"
PYTEST_COLLECTION_SCHEMA_VERSION = 1
PYTEST_MARKER_EXPRESSION = "not source_archive and not release_artifact"
PYTEST_IGNORED_PATHS: tuple[str, ...] = ()
RECORD_ROLE = "unsigned_ai_generated_release_machine_validation_summary"
PRODUCT = "GridGuard Khartoum"
RELEASE_VERSION = "1.2.0"
RELEASE_ID = "gridguard-khartoum-v1.2.0"
RELEASE_DATE = "2026-08-27"

COMMAND_KEYS = (
    "locked_sync",
    "ruff_check",
    "ruff_format_check",
    "cuda_doctor",
    "evidence_verification",
    "paired_reproduction",
    "robustness_reproduction",
    "source_pytest_collection",
    "source_pytest",
)

DEFAULT_RECORD_PATH = Path("docs/release-machine-validation.json")
DEFAULT_VISUAL_RECORD_PATH = Path("docs/release-visual-review.json")
PAIRED_CANONICAL_PATH = Path("data/evidence/paired-benchmark.json")
ROBUSTNESS_CANONICAL_PATH = Path("data/evidence/robustness-supplement.json")
UV_LOCK_PATH = Path("uv.lock")
PAIRED_CANONICAL_SHA256 = "7f2009c4f7efbbedf098069ced9218addec51d55a18b969e6c54433027020af5"
ROBUSTNESS_CANONICAL_SHA256 = "6b56d149331ee191302cd9e6f22365988db0ae301f9441190b61214acf832e48"

AUTOMATED_SCOPE = (
    "This unsigned AI-generated record validates a closed schema, the tested Git "
    "commit and uv.lock identity, exact recorded commands and zero caller-supplied "
    "exit codes, exact pytest collection-to-JUnit testcase equality, JUnit arithmetic, "
    "evidence byte identities, and strict visual-record bindings. It does not "
    "authenticate an operator or independently re-execute commands."
)
TRANSIENT_INPUT_SCOPE = (
    "The exact pytest collection inventory, JUnit XML, and reproduction outputs are "
    "transient retained-workspace inputs. Their identities and reconstructed facts are "
    "embedded; default verification is self-contained, while verify_inputs=True rechecks "
    "retained or freshly generated transient files. The deterministic collection and "
    "evidence reproductions are compared byte-for-byte. JUnit testcase IDs and outcomes "
    "are compared semantically because timing and host fields are not byte-reproducible."
)
PACKAGE_RUNTIME_SCOPE = (
    "Post-build release-manifest and release_artifact gates, package inventory, "
    "and frozen-runtime facts are separately executed and bound by the final "
    "build workflow and RELEASE_MANIFEST.windows_build."
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_NONNEGATIVE_INTEGER = re.compile(r"0|[1-9][0-9]*")
_SAFE_COMMAND_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")
_MAX_JUNIT_BYTES = 64 * 1024 * 1024
_JUNIT_OUTCOME_TAGS = frozenset({"failure", "error", "skipped"})
_JUNIT_ALLOWED_TESTCASE_CHILDREN = frozenset(
    {"failure", "error", "skipped", "properties", "system-out", "system-err"}
)
_JUNIT_ALLOWED_SUITE_CHILDREN = frozenset({"testcase", "properties", "system-out", "system-err"})


class MachineValidationError(ValueError):
    """Raised when release machine-validation evidence fails closed."""


def canonical_machine_validation_bytes(payload: object) -> bytes:
    """Return deterministic human-readable JSON bytes for one record."""
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise MachineValidationError(
            f"machine-validation record is not JSON-serializable: {error}"
        ) from error
    return f"{text}\n".encode()


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MachineValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise MachineValidationError(f"invalid non-finite JSON constant: {value}")


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MachineValidationError(f"{label} must be a JSON object")
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise MachineValidationError(f"{label} has unknown fields: {unknown}")
    if missing:
        raise MachineValidationError(f"{label} is missing fields: {missing}")
    return value


def _require_exact_value(value: object, expected: object, *, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise MachineValidationError(f"{label} must be {expected!r}")


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise MachineValidationError(
            f"{label} must be a nonempty string without surrounding whitespace"
        )
    if any(ord(character) < 32 for character in value):
        raise MachineValidationError(f"{label} must not contain control characters")
    return value


def _require_positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise MachineValidationError(f"{label} must be a positive integer")
    return value


def _validate_relative_path(value: object, *, label: str) -> str:
    path = _require_string(value, label=label)
    pure = PurePosixPath(path)
    if (
        "\\" in path
        or ":" in path
        or pure.is_absolute()
        or path == "."
        or ".." in pure.parts
        or pure.as_posix() != path
    ):
        raise MachineValidationError(f"{label} must be a normalized repository-relative POSIX path")
    if _SAFE_COMMAND_PATH.fullmatch(path) is None:
        raise MachineValidationError(
            f"{label} must use only safe command-path characters: A-Z, a-z, 0-9, ._/-"
        )
    return path


def _repository_root(repository_root: str | Path) -> Path:
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise MachineValidationError(f"repository root is not a directory: {root}")
    return root


def _resolve_inside_repository(
    root: Path,
    supplied_path: str | Path,
    *,
    label: str,
) -> tuple[Path, str]:
    requested = Path(supplied_path)
    resolved = (root / requested).resolve() if not requested.is_absolute() else requested.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise MachineValidationError(f"{label} must resolve inside the repository root") from error
    relative_path = relative.as_posix()
    _validate_relative_path(relative_path, label=f"{label} path")
    return resolved, relative_path


def _resolve_recorded_path(root: Path, path: object, *, label: str) -> Path:
    relative_path = _validate_relative_path(path, label=f"{label} path")
    resolved, current_relative = _resolve_inside_repository(root, relative_path, label=label)
    if current_relative != relative_path:
        raise MachineValidationError(f"{label} path is not canonical")
    return resolved


def _sha256(path: Path, *, label: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MachineValidationError(f"cannot read {label}: {error}") from error
    return digest.hexdigest()


def _identity(path: Path, relative_path: str, *, label: str) -> dict[str, object]:
    try:
        size_bytes = path.stat().st_size
    except OSError as error:
        raise MachineValidationError(f"cannot inspect {label}: {error}") from error
    if not path.is_file() or size_bytes <= 0:
        raise MachineValidationError(f"{label} must be a nonempty file: {path}")
    return {
        "path": relative_path,
        "sha256": _sha256(path, label=label),
        "size_bytes": size_bytes,
    }


def _validate_identity(value: object, *, label: str) -> Mapping[str, object]:
    binding = _require_exact_fields(
        value,
        {"path", "sha256", "size_bytes"},
        label=label,
    )
    _validate_relative_path(binding["path"], label=f"{label} path")
    sha256 = binding["sha256"]
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise MachineValidationError(f"{label} sha256 must be 64 lowercase hexadecimal characters")
    _require_positive_integer(binding["size_bytes"], label=f"{label} size_bytes")
    return binding


def _require_current_identity(
    root: Path,
    recorded: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    path = _resolve_recorded_path(root, recorded["path"], label=label)
    current = _identity(path, str(recorded["path"]), label=label)
    if dict(recorded) != current:
        raise MachineValidationError(
            f"{label} identity is stale: recorded {dict(recorded)!r}, current {current!r}"
        )
    return current


def _files_are_identical(first: Path, second: Path) -> bool:
    try:
        if first.stat().st_size != second.stat().st_size:
            return False
        with first.open("rb") as first_stream, second.open("rb") as second_stream:
            while True:
                first_chunk = first_stream.read(1024 * 1024)
                second_chunk = second_stream.read(1024 * 1024)
                if first_chunk != second_chunk:
                    return False
                if not first_chunk:
                    return True
    except OSError as error:
        raise MachineValidationError(
            f"cannot compare evidence reproduction bytes: {error}"
        ) from error


def _commands_for_paths(
    *,
    collection_path: str,
    junit_path: str,
    paired_reproduction_path: str,
    robustness_reproduction_path: str,
) -> dict[str, str]:
    return {
        "locked_sync": "uv sync --locked --extra dev --extra docs --offline",
        "ruff_check": "uv run --no-sync ruff check .",
        "ruff_format_check": "uv run --no-sync ruff format --check .",
        "cuda_doctor": "uv run --no-sync gridguard-doctor",
        "evidence_verification": ("uv run --no-sync gridguard-evaluate --verify-evidence"),
        "paired_reproduction": (
            "uv run --no-sync gridguard-evaluate --reproduce-paired --output "
            f'"{paired_reproduction_path}"'
        ),
        "robustness_reproduction": (
            "uv run --no-sync gridguard-evaluate --reproduce-robustness --output "
            f'"{robustness_reproduction_path}"'
        ),
        "source_pytest_collection": (
            "uv run --no-sync python tools/machine_validation.py "
            "--collect-pytest-inventory --collection-path "
            f'"{collection_path}"'
        ),
        "source_pytest": (
            f'uv run --no-sync pytest -q --junitxml "{junit_path}" -m "{PYTEST_MARKER_EXPRESSION}"'
        ),
    }


def expected_machine_validation_commands(
    repository_root: str | Path,
    *,
    collection_path: str | Path,
    junit_path: str | Path,
    paired_reproduction_path: str | Path,
    robustness_reproduction_path: str | Path,
) -> dict[str, str]:
    """Return the exact commands whose exit codes the record accepts."""
    root = _repository_root(repository_root)
    _, collection_relative = _resolve_inside_repository(
        root, collection_path, label="pytest collection inventory"
    )
    _, junit_relative = _resolve_inside_repository(root, junit_path, label="JUnit XML")
    _, paired_relative = _resolve_inside_repository(
        root, paired_reproduction_path, label="paired reproduction"
    )
    _, robustness_relative = _resolve_inside_repository(
        root, robustness_reproduction_path, label="robustness reproduction"
    )
    if len({collection_relative, junit_relative, paired_relative, robustness_relative}) != 4:
        raise MachineValidationError(
            "pytest collection, JUnit, and evidence reproduction paths must be distinct"
        )
    return _commands_for_paths(
        collection_path=collection_relative,
        junit_path=junit_relative,
        paired_reproduction_path=paired_relative,
        robustness_reproduction_path=robustness_relative,
    )


def _validate_exit_codes(value: object) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise MachineValidationError("command exit codes must be a mapping")
    actual = set(value)
    expected = set(COMMAND_KEYS)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise MachineValidationError(f"command exit codes have unknown gates: {unknown}")
    if missing:
        raise MachineValidationError(f"command exit codes are missing gates: {missing}")
    validated: dict[str, int] = {}
    for key in COMMAND_KEYS:
        exit_code = value[key]
        if type(exit_code) is not int:
            raise MachineValidationError(f"{key} exit code must be an integer")
        if exit_code != 0:
            raise MachineValidationError(f"{key} exit code must be zero, found {exit_code}")
        validated[key] = exit_code
    return validated


def _junit_integer_attribute(
    element: ElementTree.Element,
    name: str,
    *,
    label: str,
) -> int:
    value = element.attrib.get(name)
    if value is None or _NONNEGATIVE_INTEGER.fullmatch(value) is None:
        raise MachineValidationError(f"{label} {name} must be a canonical nonnegative integer")
    return int(value)


def _junit_identifier(testcase: ElementTree.Element, *, suite_index: int) -> str:
    classname = _require_string(
        testcase.attrib.get("classname"),
        label=f"JUnit suite {suite_index} testcase classname",
    )
    name = _require_string(
        testcase.attrib.get("name"),
        label=f"JUnit suite {suite_index} testcase name",
    )
    return f"{classname}::{name}"


def _parse_junit(path: Path, relative_path: str) -> dict[str, object]:
    identity = _identity(path, relative_path, label="JUnit XML")
    size_bytes = int(identity["size_bytes"])
    if size_bytes > _MAX_JUNIT_BYTES:
        raise MachineValidationError(
            f"JUnit XML exceeds the {_MAX_JUNIT_BYTES}-byte validation limit"
        )
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise MachineValidationError(f"cannot read JUnit XML: {error}") from error
    if b"<!doctype" in encoded.lower():
        raise MachineValidationError("JUnit XML must not contain a document type")
    try:
        root = ElementTree.fromstring(encoded)
    except ElementTree.ParseError as error:
        raise MachineValidationError(f"JUnit XML is invalid: {error}") from error
    if root.tag != "testsuites":
        raise MachineValidationError("JUnit XML root must be testsuites")
    suites = list(root)
    if not suites or any(suite.tag != "testsuite" for suite in suites):
        raise MachineValidationError(
            "JUnit XML must contain one or more direct testsuite elements only"
        )

    testcase_ids: list[str] = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skips = 0
    for suite_index, suite in enumerate(suites, start=1):
        children = list(suite)
        unknown_children = sorted({child.tag for child in children} - _JUNIT_ALLOWED_SUITE_CHILDREN)
        if unknown_children:
            raise MachineValidationError(
                f"JUnit suite {suite_index} has unsupported elements: {unknown_children}"
            )
        testcases = [child for child in children if child.tag == "testcase"]
        suite_failures = 0
        suite_errors = 0
        suite_skips = 0
        for testcase in testcases:
            testcase_ids.append(_junit_identifier(testcase, suite_index=suite_index))
            testcase_children = list(testcase)
            unsupported = sorted(
                {child.tag for child in testcase_children} - _JUNIT_ALLOWED_TESTCASE_CHILDREN
            )
            if unsupported:
                raise MachineValidationError(
                    f"JUnit testcase has unsupported elements: {unsupported}"
                )
            outcomes = [
                child.tag for child in testcase_children if child.tag in _JUNIT_OUTCOME_TAGS
            ]
            if len(outcomes) > 1:
                raise MachineValidationError("JUnit testcase must not declare multiple outcomes")
            if outcomes == ["failure"]:
                suite_failures += 1
            elif outcomes == ["error"]:
                suite_errors += 1
            elif outcomes == ["skipped"]:
                suite_skips += 1

        computed = {
            "tests": len(testcases),
            "failures": suite_failures,
            "errors": suite_errors,
            "skipped": suite_skips,
        }
        declared = {
            name: _junit_integer_attribute(suite, name, label=f"JUnit suite {suite_index}")
            for name in ("tests", "failures", "errors", "skipped")
        }
        if declared != computed:
            differences = [
                f"{('skips' if name == 'skipped' else name)}="
                f"{declared[name]} declared/{computed[name]} reconstructed"
                for name in computed
                if declared[name] != computed[name]
            ]
            raise MachineValidationError(
                "JUnit suite declared totals do not match testcase outcomes: "
                + ", ".join(differences)
            )
        total_tests += computed["tests"]
        total_failures += computed["failures"]
        total_errors += computed["errors"]
        total_skips += computed["skipped"]

    root_count_fields = {"tests", "failures", "errors", "skipped"}
    present_root_counts = root_count_fields.intersection(root.attrib)
    if present_root_counts and present_root_counts != root_count_fields:
        raise MachineValidationError(
            "JUnit testsuites aggregate counts must be all present or all absent"
        )
    if present_root_counts:
        reconstructed = {
            "tests": total_tests,
            "failures": total_failures,
            "errors": total_errors,
            "skipped": total_skips,
        }
        declared_root = {
            name: _junit_integer_attribute(root, name, label="JUnit testsuites")
            for name in ("tests", "failures", "errors", "skipped")
        }
        if declared_root != reconstructed:
            raise MachineValidationError(
                "JUnit testsuites declared totals do not match reconstructed suite totals"
            )

    if total_tests <= 0:
        raise MachineValidationError("JUnit report must contain at least one testcase")
    if total_failures:
        raise MachineValidationError(f"JUnit report contains {total_failures} failures")
    if total_errors:
        raise MachineValidationError(f"JUnit report contains {total_errors} errors")
    if total_skips:
        raise MachineValidationError(f"JUnit report contains {total_skips} skips")
    sorted_ids = sorted(testcase_ids)
    if len(set(sorted_ids)) != len(sorted_ids):
        raise MachineValidationError("JUnit testcase identifiers must be unique")
    return {
        **identity,
        "tests": total_tests,
        "failures": total_failures,
        "errors": total_errors,
        "skips": total_skips,
        "testcase_ids": sorted_ids,
    }


def _validate_pytest_collection_payload(value: object) -> Mapping[str, object]:
    collection = _require_exact_fields(
        value,
        {"schema_name", "schema_version", "selection", "testcase_ids", "tests"},
        label="pytest collection inventory",
    )
    _require_exact_value(
        collection["schema_name"],
        PYTEST_COLLECTION_SCHEMA_NAME,
        label="pytest collection schema_name",
    )
    _require_exact_value(
        collection["schema_version"],
        PYTEST_COLLECTION_SCHEMA_VERSION,
        label="pytest collection schema_version",
    )
    selection = _require_exact_fields(
        collection["selection"],
        {"ignore", "marker_expression"},
        label="pytest collection selection",
    )
    _require_exact_value(
        selection["marker_expression"],
        PYTEST_MARKER_EXPRESSION,
        label="pytest collection marker_expression",
    )
    _require_exact_value(
        selection["ignore"],
        list(PYTEST_IGNORED_PATHS),
        label="pytest collection ignore",
    )
    tests = _require_positive_integer(collection["tests"], label="pytest collection tests")
    testcase_ids = collection["testcase_ids"]
    if not isinstance(testcase_ids, list):
        raise MachineValidationError("pytest collection testcase_ids must be a list")
    validated_ids = [
        _require_string(identifier, label="pytest collection testcase identifier")
        for identifier in testcase_ids
    ]
    if validated_ids != sorted(validated_ids):
        raise MachineValidationError(
            "pytest collection testcase identifiers must be canonically sorted"
        )
    if len(set(validated_ids)) != len(validated_ids):
        raise MachineValidationError("pytest collection testcase identifiers must be unique")
    if len(validated_ids) != tests:
        raise MachineValidationError(
            "pytest collection testcase identifier count does not equal its testcase total"
        )
    return collection


def _parse_pytest_collection(path: Path, relative_path: str) -> dict[str, object]:
    identity = _identity(path, relative_path, label="pytest collection inventory")
    try:
        encoded = path.read_bytes()
        loaded = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except MachineValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MachineValidationError(f"pytest collection inventory is invalid: {error}") from error
    if canonical_machine_validation_bytes(loaded) != encoded:
        raise MachineValidationError("pytest collection inventory is not canonical JSON")
    collection = _validate_pytest_collection_payload(loaded)
    return {**identity, **dict(collection)}


def _require_exact_pytest_collection(
    collection: Mapping[str, object], junit: Mapping[str, object]
) -> None:
    if collection["tests"] != junit["tests"] or collection["testcase_ids"] != junit["testcase_ids"]:
        raise MachineValidationError(
            "JUnit testcase IDs/count must equal the exact pytest collection inventory"
        )


def _visual_facts(root: Path, visual_record_path: Path, relative_path: str) -> dict[str, object]:
    try:
        visual = verify_release_visual_review(root, record_path=visual_record_path)
    except ReleaseVisualReviewError as error:
        raise MachineValidationError(f"visual-review verification failed: {error}") from error
    artifacts = visual.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise MachineValidationError("visual-review record has no artifacts mapping")
    pdf = artifacts.get("pdf")
    presentation = artifacts.get("presentation")
    if not isinstance(pdf, Mapping) or not isinstance(presentation, Mapping):
        raise MachineValidationError(
            "visual-review record does not contain PDF and presentation facts"
        )
    return {
        "record": _identity(
            visual_record_path,
            relative_path,
            label="visual-review JSON",
        ),
        "pdf": {
            "path": pdf["path"],
            "sha256": pdf["sha256"],
            "size_bytes": pdf["size_bytes"],
            "page_count": pdf["detected_page_count"],
        },
        "presentation": {
            "path": presentation["path"],
            "sha256": presentation["sha256"],
            "size_bytes": presentation["size_bytes"],
            "slide_count": presentation["detected_slide_count"],
            "speaker_note_slide_count": presentation["detected_speaker_note_slide_count"],
        },
    }


def _validate_junit_summary(value: object) -> Mapping[str, object]:
    junit = _require_exact_fields(
        value,
        {
            "errors",
            "failures",
            "path",
            "sha256",
            "size_bytes",
            "skips",
            "testcase_ids",
            "tests",
        },
        label="JUnit summary",
    )
    _validate_identity(
        {field: junit[field] for field in ("path", "sha256", "size_bytes")},
        label="JUnit input",
    )
    tests = _require_positive_integer(junit["tests"], label="JUnit tests")
    for field in ("failures", "errors", "skips"):
        _require_exact_value(junit[field], 0, label=f"JUnit {field}")
    testcase_ids = junit["testcase_ids"]
    if not isinstance(testcase_ids, list):
        raise MachineValidationError("JUnit testcase_ids must be a list")
    validated_ids = [
        _require_string(identifier, label="JUnit testcase identifier")
        for identifier in testcase_ids
    ]
    if validated_ids != sorted(validated_ids):
        raise MachineValidationError("JUnit testcase identifiers must be canonically sorted")
    if len(set(validated_ids)) != len(validated_ids):
        raise MachineValidationError("JUnit testcase identifiers must be unique")
    if len(validated_ids) != tests:
        raise MachineValidationError(
            "JUnit testcase identifier count does not equal the recorded testcase total"
        )
    return junit


def _validate_pytest_collection_summary(value: object) -> Mapping[str, object]:
    collection = _require_exact_fields(
        value,
        {
            "path",
            "schema_name",
            "schema_version",
            "selection",
            "sha256",
            "size_bytes",
            "testcase_ids",
            "tests",
        },
        label="pytest collection summary",
    )
    _validate_identity(
        {field: collection[field] for field in ("path", "sha256", "size_bytes")},
        label="pytest collection input",
    )
    _validate_pytest_collection_payload(
        {
            field: collection[field]
            for field in (
                "schema_name",
                "schema_version",
                "selection",
                "testcase_ids",
                "tests",
            )
        }
    )
    return collection


def _validate_reproduction(
    value: object,
    *,
    label: str,
    expected_canonical_path: str,
    expected_sha256: str,
) -> Mapping[str, object]:
    reproduction = _require_exact_fields(
        value,
        {"byte_identical", "canonical", "reproduction"},
        label=f"{label} reproduction",
    )
    canonical = _validate_identity(reproduction["canonical"], label=f"{label} canonical")
    output = _validate_identity(reproduction["reproduction"], label=f"{label} reproduction")
    _require_exact_value(
        canonical["path"], expected_canonical_path, label=f"{label} canonical path"
    )
    _require_exact_value(canonical["sha256"], expected_sha256, label=f"{label} canonical sha256")
    _require_exact_value(reproduction["byte_identical"], True, label=f"{label} byte_identical")
    if output["path"] == canonical["path"]:
        raise MachineValidationError(
            f"{label} reproduction path must differ from the trusted canonical path"
        )
    for field in ("sha256", "size_bytes"):
        if output[field] != canonical[field]:
            raise MachineValidationError(
                f"{label} reproduction {field} must equal the canonical identity"
            )
    return reproduction


def _validate_visual_facts(value: object) -> Mapping[str, object]:
    visual = _require_exact_fields(
        value,
        {"pdf", "presentation", "record"},
        label="visual_review",
    )
    _validate_identity(visual["record"], label="visual-review record")
    pdf = _require_exact_fields(
        visual["pdf"],
        {"page_count", "path", "sha256", "size_bytes"},
        label="visual-review PDF",
    )
    _validate_identity(
        {field: pdf[field] for field in ("path", "sha256", "size_bytes")},
        label="visual-review PDF",
    )
    _require_exact_value(
        pdf["page_count"], EXPECTED_PDF_PAGE_COUNT, label="visual-review PDF page_count"
    )
    presentation = _require_exact_fields(
        visual["presentation"],
        {
            "path",
            "sha256",
            "size_bytes",
            "slide_count",
            "speaker_note_slide_count",
        },
        label="visual-review presentation",
    )
    _validate_identity(
        {field: presentation[field] for field in ("path", "sha256", "size_bytes")},
        label="visual-review presentation",
    )
    _require_exact_value(
        presentation["slide_count"],
        13,
        label="visual-review presentation slide_count",
    )
    _require_exact_value(
        presentation["speaker_note_slide_count"],
        13,
        label="visual-review presentation speaker_note_slide_count",
    )
    return visual


def _validate_record(record: object) -> Mapping[str, object]:
    top = _require_exact_fields(
        record,
        {
            "assurance_scope",
            "commands",
            "evidence_reproductions",
            "junit",
            "pytest_collection",
            "record_role",
            "release",
            "schema_name",
            "schema_version",
            "source",
            "visual_review",
        },
        label="machine-validation record",
    )
    _require_exact_value(top["schema_name"], SCHEMA_NAME, label="schema_name")
    _require_exact_value(top["schema_version"], SCHEMA_VERSION, label="schema_version")
    _require_exact_value(top["record_role"], RECORD_ROLE, label="record_role")

    release = _require_exact_fields(
        top["release"],
        {"date", "product", "release_id", "version"},
        label="release",
    )
    for field, expected in (
        ("product", PRODUCT),
        ("version", RELEASE_VERSION),
        ("release_id", RELEASE_ID),
        ("date", RELEASE_DATE),
    ):
        _require_exact_value(release[field], expected, label=f"release {field}")

    source = _require_exact_fields(
        top["source"],
        {"git_commit", "uv_lock"},
        label="source",
    )
    source_git_commit = _require_string(
        source["git_commit"],
        label="source Git commit",
    )
    if _GIT_COMMIT.fullmatch(source_git_commit) is None:
        raise MachineValidationError(
            "source Git commit must be a lowercase 40-character hexadecimal ID"
        )
    source_uv_lock = _validate_identity(source["uv_lock"], label="source uv.lock")
    _require_exact_value(
        source_uv_lock["path"],
        UV_LOCK_PATH.as_posix(),
        label="source uv.lock path",
    )

    scope = _require_exact_fields(
        top["assurance_scope"],
        {
            "automated_scope",
            "cryptographic_authenticity_claimed",
            "human_execution_claimed",
            "package_runtime_binding",
            "package_runtime_facts_included",
            "package_runtime_scope",
            "transient_input_scope",
        },
        label="assurance_scope",
    )
    _require_exact_value(scope["automated_scope"], AUTOMATED_SCOPE, label="automated_scope")
    _require_exact_value(
        scope["cryptographic_authenticity_claimed"],
        False,
        label="cryptographic_authenticity_claimed",
    )
    _require_exact_value(scope["human_execution_claimed"], False, label="human_execution_claimed")
    _require_exact_value(
        scope["package_runtime_binding"],
        "RELEASE_MANIFEST.windows_build",
        label="package_runtime_binding",
    )
    _require_exact_value(
        scope["package_runtime_facts_included"],
        False,
        label="package_runtime_facts_included",
    )
    _require_exact_value(
        scope["package_runtime_scope"],
        PACKAGE_RUNTIME_SCOPE,
        label="package_runtime_scope",
    )
    _require_exact_value(
        scope["transient_input_scope"],
        TRANSIENT_INPUT_SCOPE,
        label="transient_input_scope",
    )

    collection = _validate_pytest_collection_summary(top["pytest_collection"])
    junit = _validate_junit_summary(top["junit"])
    _require_exact_pytest_collection(collection, junit)
    evidence = _require_exact_fields(
        top["evidence_reproductions"],
        {"paired", "robustness"},
        label="evidence_reproductions",
    )
    paired = _validate_reproduction(
        evidence["paired"],
        label="paired",
        expected_canonical_path=PAIRED_CANONICAL_PATH.as_posix(),
        expected_sha256=PAIRED_CANONICAL_SHA256,
    )
    robustness = _validate_reproduction(
        evidence["robustness"],
        label="robustness",
        expected_canonical_path=ROBUSTNESS_CANONICAL_PATH.as_posix(),
        expected_sha256=ROBUSTNESS_CANONICAL_SHA256,
    )
    visual = _validate_visual_facts(top["visual_review"])

    paired_output = paired["reproduction"]
    robustness_output = robustness["reproduction"]
    assert isinstance(paired_output, Mapping)
    assert isinstance(robustness_output, Mapping)
    expected_commands = _commands_for_paths(
        collection_path=str(collection["path"]),
        junit_path=str(junit["path"]),
        paired_reproduction_path=str(paired_output["path"]),
        robustness_reproduction_path=str(robustness_output["path"]),
    )
    commands = _require_exact_fields(top["commands"], set(COMMAND_KEYS), label="commands")
    for key in COMMAND_KEYS:
        command = _require_exact_fields(
            commands[key], {"command", "exit_code"}, label=f"commands.{key}"
        )
        _require_exact_value(command["command"], expected_commands[key], label=f"{key} command")
        _require_exact_value(command["exit_code"], 0, label=f"{key} exit_code")

    path_values = [
        str(source_uv_lock["path"]),
        str(collection["path"]),
        str(junit["path"]),
        str(paired["canonical"]["path"]),
        str(paired_output["path"]),
        str(robustness["canonical"]["path"]),
        str(robustness_output["path"]),
        str(visual["record"]["path"]),
        str(visual["pdf"]["path"]),
        str(visual["presentation"]["path"]),
    ]
    if len(set(path_values)) != len(path_values):
        raise MachineValidationError("machine-validation input and artifact paths must be distinct")
    return top


def _load_record(path: Path) -> Mapping[str, object]:
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise MachineValidationError(
            f"cannot read machine-validation record {path}: {error}"
        ) from error
    try:
        decoded = encoded.decode("utf-8")
        loaded = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except MachineValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MachineValidationError(
            f"machine-validation record is invalid JSON: {error}"
        ) from error
    if canonical_machine_validation_bytes(loaded) != encoded:
        raise MachineValidationError("machine-validation record is not canonical JSON")
    return _validate_record(loaded)


def _write_atomic(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except OSError as error:
        raise MachineValidationError(
            f"cannot write machine-validation record {path}: {error}"
        ) from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _pytest_nodeid_to_junit_identifier(nodeid: object) -> str:
    normalized = _require_string(nodeid, label="collected pytest nodeid")
    path_and_test, bracket, parameters = normalized.partition("[")
    names = path_and_test.split("::")
    if len(names) < 2 or not names[0].endswith(".py"):
        raise MachineValidationError(
            f"collected pytest nodeid cannot map to JUnit identity: {normalized!r}"
        )
    names[0] = names[0].replace("\\", ".").replace("/", ".")[:-3]
    names[-1] += bracket + parameters
    identifier = f"{'.'.join(names[:-1])}::{names[-1]}"
    return _require_string(identifier, label="collected pytest JUnit identifier")


class _PytestCollectionPlugin:
    def __init__(self) -> None:
        self.testcase_ids: list[str] = []

    def pytest_collection_finish(self, session: object) -> None:
        items = getattr(session, "items", None)
        if not isinstance(items, list):
            raise MachineValidationError("pytest collection did not expose collected items")
        self.testcase_ids = [
            _pytest_nodeid_to_junit_identifier(getattr(item, "nodeid", None)) for item in items
        ]


def _reject_ambient_pytest_controls() -> None:
    controls = sorted(
        name
        for name, value in os.environ.items()
        if name.startswith("PYTEST_")
        and name not in {"PYTEST_CURRENT_TEST", "PYTEST_VERSION"}
        and value.strip()
    )
    if controls:
        raise MachineValidationError(
            "ambient pytest controls are not permitted for the attested collection: "
            + ", ".join(controls)
        )


def collect_pytest_inventory(
    repository_root: str | Path,
    *,
    collection_path: str | Path,
) -> dict[str, object]:
    """Collect the exact release pytest selection and write its canonical inventory."""
    root = _repository_root(repository_root)
    _reject_ambient_pytest_controls()
    resolved_collection, _ = _resolve_inside_repository(
        root, collection_path, label="pytest collection inventory"
    )
    try:
        import pytest
    except ImportError as error:  # pragma: no cover - release environment contract
        raise MachineValidationError(
            "pytest is required to collect the release inventory"
        ) from error

    plugin = _PytestCollectionPlugin()
    original_working_directory = Path.cwd()
    try:
        os.chdir(root)
        exit_code = pytest.main(
            [
                "--collect-only",
                "-q",
                "-m",
                PYTEST_MARKER_EXPRESSION,
                *(f"--ignore={path}" for path in PYTEST_IGNORED_PATHS),
            ],
            plugins=[plugin],
        )
    finally:
        os.chdir(original_working_directory)
    if int(exit_code) != 0:
        raise MachineValidationError(f"pytest collection failed with exit code {int(exit_code)}")
    testcase_ids = sorted(plugin.testcase_ids)
    if not testcase_ids:
        raise MachineValidationError("pytest collection must contain at least one testcase")
    if len(set(testcase_ids)) != len(testcase_ids):
        raise MachineValidationError("pytest collection testcase identifiers must be unique")
    payload: dict[str, object] = {
        "schema_name": PYTEST_COLLECTION_SCHEMA_NAME,
        "schema_version": PYTEST_COLLECTION_SCHEMA_VERSION,
        "selection": {
            "marker_expression": PYTEST_MARKER_EXPRESSION,
            "ignore": list(PYTEST_IGNORED_PATHS),
        },
        "testcase_ids": testcase_ids,
        "tests": len(testcase_ids),
    }
    _validate_pytest_collection_payload(payload)
    _write_atomic(resolved_collection, canonical_machine_validation_bytes(payload))
    return payload


def _canonical_evidence_identity(
    root: Path,
    relative_path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[Path, dict[str, object]]:
    resolved, relative = _resolve_inside_repository(root, relative_path, label=label)
    identity = _identity(resolved, relative, label=label)
    if identity["sha256"] != expected_sha256:
        raise MachineValidationError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, found {identity['sha256']}"
        )
    return resolved, identity


def refresh_release_machine_validation(
    repository_root: str | Path,
    *,
    collection_path: str | Path,
    junit_path: str | Path,
    paired_reproduction_path: str | Path,
    robustness_reproduction_path: str | Path,
    command_exit_codes: Mapping[str, int],
    source_git_commit: str,
    record_path: str | Path = DEFAULT_RECORD_PATH,
    visual_record_path: str | Path = DEFAULT_VISUAL_RECORD_PATH,
) -> dict[str, object]:
    """Write a canonical record from completed release-workspace inputs."""
    root = _repository_root(repository_root)
    if not isinstance(source_git_commit, str) or _GIT_COMMIT.fullmatch(source_git_commit) is None:
        raise MachineValidationError(
            "source Git commit must be a lowercase 40-character hexadecimal ID"
        )
    resolved_record, _ = _resolve_inside_repository(
        root, record_path, label="machine-validation record"
    )
    resolved_visual, relative_visual = _resolve_inside_repository(
        root, visual_record_path, label="visual-review record"
    )
    resolved_collection, relative_collection = _resolve_inside_repository(
        root, collection_path, label="pytest collection inventory"
    )
    resolved_junit, relative_junit = _resolve_inside_repository(root, junit_path, label="JUnit XML")
    resolved_paired_output, relative_paired_output = _resolve_inside_repository(
        root, paired_reproduction_path, label="paired reproduction"
    )
    resolved_robustness_output, relative_robustness_output = _resolve_inside_repository(
        root, robustness_reproduction_path, label="robustness reproduction"
    )
    resolved_uv_lock, relative_uv_lock = _resolve_inside_repository(
        root, UV_LOCK_PATH, label="source uv.lock"
    )
    uv_lock_identity = _identity(
        resolved_uv_lock,
        relative_uv_lock,
        label="source uv.lock",
    )
    resolved_paired, paired_identity = _canonical_evidence_identity(
        root,
        PAIRED_CANONICAL_PATH,
        PAIRED_CANONICAL_SHA256,
        label="paired canonical evidence",
    )
    resolved_robustness, robustness_identity = _canonical_evidence_identity(
        root,
        ROBUSTNESS_CANONICAL_PATH,
        ROBUSTNESS_CANONICAL_SHA256,
        label="robustness canonical evidence",
    )
    all_paths = {
        resolved_record,
        resolved_visual,
        resolved_collection,
        resolved_junit,
        resolved_paired_output,
        resolved_robustness_output,
        resolved_paired,
        resolved_robustness,
        resolved_uv_lock,
    }
    if len(all_paths) != 9:
        raise MachineValidationError(
            "record, source, evidence, visual-review, pytest collection, JUnit, and reproduction "
            "paths must be distinct"
        )

    exit_codes = _validate_exit_codes(command_exit_codes)
    commands = _commands_for_paths(
        collection_path=relative_collection,
        junit_path=relative_junit,
        paired_reproduction_path=relative_paired_output,
        robustness_reproduction_path=relative_robustness_output,
    )
    collection = _parse_pytest_collection(resolved_collection, relative_collection)
    junit = _parse_junit(resolved_junit, relative_junit)
    _require_exact_pytest_collection(collection, junit)
    paired_output_identity = _identity(
        resolved_paired_output,
        relative_paired_output,
        label="paired reproduction",
    )
    if not _files_are_identical(resolved_paired, resolved_paired_output):
        raise MachineValidationError(
            "paired reproduction must be byte-identical to paired canonical evidence"
        )
    robustness_output_identity = _identity(
        resolved_robustness_output,
        relative_robustness_output,
        label="robustness reproduction",
    )
    if not _files_are_identical(resolved_robustness, resolved_robustness_output):
        raise MachineValidationError(
            "robustness reproduction must be byte-identical to robustness canonical evidence"
        )
    visual = _visual_facts(root, resolved_visual, relative_visual)

    record: dict[str, object] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "record_role": RECORD_ROLE,
        "release": {
            "product": PRODUCT,
            "version": RELEASE_VERSION,
            "release_id": RELEASE_ID,
            "date": RELEASE_DATE,
        },
        "source": {
            "git_commit": source_git_commit,
            "uv_lock": uv_lock_identity,
        },
        "assurance_scope": {
            "automated_scope": AUTOMATED_SCOPE,
            "cryptographic_authenticity_claimed": False,
            "human_execution_claimed": False,
            "transient_input_scope": TRANSIENT_INPUT_SCOPE,
            "package_runtime_binding": "RELEASE_MANIFEST.windows_build",
            "package_runtime_facts_included": False,
            "package_runtime_scope": PACKAGE_RUNTIME_SCOPE,
        },
        "commands": {
            key: {"command": commands[key], "exit_code": exit_codes[key]} for key in COMMAND_KEYS
        },
        "pytest_collection": collection,
        "junit": junit,
        "evidence_reproductions": {
            "paired": {
                "canonical": paired_identity,
                "reproduction": paired_output_identity,
                "byte_identical": True,
            },
            "robustness": {
                "canonical": robustness_identity,
                "reproduction": robustness_output_identity,
                "byte_identical": True,
            },
        },
        "visual_review": visual,
    }
    _validate_record(record)
    _write_atomic(resolved_record, canonical_machine_validation_bytes(record))
    return record


def _require_same_mapping(
    recorded: Mapping[str, object],
    current: Mapping[str, object],
    *,
    label: str,
) -> None:
    if dict(recorded) != dict(current):
        raise MachineValidationError(f"{label} does not match the recorded release input")


def _require_same_junit_facts(
    recorded: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    semantic_fields = ("path", "tests", "failures", "errors", "skips", "testcase_ids")
    recorded_facts = {field: recorded[field] for field in semantic_fields}
    current_facts = {field: current[field] for field in semantic_fields}
    if recorded_facts != current_facts:
        raise MachineValidationError(
            "JUnit transient input semantic facts do not match the recorded release run"
        )


def verify_release_machine_validation(
    repository_root: str | Path,
    *,
    record_path: str | Path = DEFAULT_RECORD_PATH,
    verify_inputs: bool = False,
) -> dict[str, object]:
    """Verify the record, committed artifacts, and optionally transient inputs."""
    if type(verify_inputs) is not bool:
        raise MachineValidationError("verify_inputs must be a boolean")
    root = _repository_root(repository_root)
    resolved_record, _ = _resolve_inside_repository(
        root, record_path, label="machine-validation record"
    )
    record = _load_record(resolved_record)

    evidence = record["evidence_reproductions"]
    source = record["source"]
    visual = record["visual_review"]
    collection = record["pytest_collection"]
    junit = record["junit"]
    assert isinstance(evidence, Mapping)
    assert isinstance(source, Mapping)
    assert isinstance(visual, Mapping)
    assert isinstance(collection, Mapping)
    assert isinstance(junit, Mapping)
    paired = evidence["paired"]
    robustness = evidence["robustness"]
    assert isinstance(paired, Mapping)
    assert isinstance(robustness, Mapping)
    paired_canonical = paired["canonical"]
    robustness_canonical = robustness["canonical"]
    paired_output = paired["reproduction"]
    robustness_output = robustness["reproduction"]
    visual_record = visual["record"]
    visual_pdf = visual["pdf"]
    visual_presentation = visual["presentation"]
    source_uv_lock = source["uv_lock"]
    assert isinstance(paired_canonical, Mapping)
    assert isinstance(robustness_canonical, Mapping)
    assert isinstance(paired_output, Mapping)
    assert isinstance(robustness_output, Mapping)
    assert isinstance(visual_record, Mapping)
    assert isinstance(visual_pdf, Mapping)
    assert isinstance(visual_presentation, Mapping)
    assert isinstance(source_uv_lock, Mapping)
    for label, binding in (
        ("pytest collection", collection),
        ("JUnit", junit),
        ("source uv.lock", source_uv_lock),
        ("paired canonical", paired_canonical),
        ("paired reproduction", paired_output),
        ("robustness canonical", robustness_canonical),
        ("robustness reproduction", robustness_output),
        ("visual-review record", visual_record),
        ("visual-review PDF", visual_pdf),
        ("visual-review presentation", visual_presentation),
    ):
        if _resolve_recorded_path(root, binding["path"], label=label) == resolved_record:
            raise MachineValidationError(
                "machine-validation record path must be distinct from every recorded input"
            )
    _require_current_identity(root, paired_canonical, label="paired canonical evidence")
    _require_current_identity(root, robustness_canonical, label="robustness canonical evidence")
    _require_current_identity(root, source_uv_lock, label="source uv.lock")

    visual_path = _resolve_recorded_path(root, visual_record["path"], label="visual-review record")
    current_visual = _visual_facts(root, visual_path, str(visual_record["path"]))
    _require_same_mapping(visual, current_visual, label="visual-review facts")

    if verify_inputs:
        collection_path = _resolve_recorded_path(
            root,
            collection["path"],
            label="transient input pytest collection",
        )
        try:
            current_collection = _parse_pytest_collection(collection_path, str(collection["path"]))
        except MachineValidationError as error:
            raise MachineValidationError(
                f"pytest collection transient input mismatch: {error}"
            ) from error
        _require_same_mapping(
            collection,
            current_collection,
            label="pytest collection transient input",
        )
        junit_path = _resolve_recorded_path(root, junit["path"], label="transient input JUnit")
        try:
            current_junit = _parse_junit(junit_path, str(junit["path"]))
        except MachineValidationError as error:
            raise MachineValidationError(f"JUnit transient input mismatch: {error}") from error
        _require_same_junit_facts(junit, current_junit)
        _require_exact_pytest_collection(current_collection, current_junit)
        for label, reproduction in (
            ("paired", paired),
            ("robustness", robustness),
        ):
            canonical = reproduction["canonical"]
            output = reproduction["reproduction"]
            assert isinstance(canonical, Mapping)
            assert isinstance(output, Mapping)
            canonical_path = _resolve_recorded_path(
                root, canonical["path"], label=f"{label} canonical evidence"
            )
            output_path = _resolve_recorded_path(
                root, output["path"], label=f"transient input {label} reproduction"
            )
            try:
                _require_current_identity(
                    root, output, label=f"{label} reproduction transient input"
                )
            except MachineValidationError as error:
                raise MachineValidationError(
                    f"{label} reproduction transient input mismatch: {error}"
                ) from error
            if not _files_are_identical(canonical_path, output_path):
                raise MachineValidationError(
                    f"{label} reproduction transient input is not byte-identical"
                )
    return dict(record)


def _parse_command_exit_codes(values: Sequence[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        key, separator, raw_exit_code = value.partition("=")
        if not separator or not key or not _NONNEGATIVE_INTEGER.fullmatch(raw_exit_code):
            raise MachineValidationError(
                "--command-exit-code values must use GATE=NONNEGATIVE_INTEGER"
            )
        if key in parsed:
            raise MachineValidationError(f"duplicate command exit-code gate: {key}")
        parsed[key] = int(raw_exit_code)
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh or verify the unsigned GridGuard release machine-validation "
            "summary. Package/runtime facts remain in RELEASE_MANIFEST.windows_build."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--refresh", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--collect-pytest-inventory", action="store_true")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--record-path", type=Path, default=DEFAULT_RECORD_PATH)
    parser.add_argument("--visual-record-path", type=Path)
    parser.add_argument("--collection-path", type=Path)
    parser.add_argument("--junit-path", type=Path)
    parser.add_argument("--paired-reproduction-path", type=Path)
    parser.add_argument("--robustness-reproduction-path", type=Path)
    parser.add_argument("--source-git-commit")
    parser.add_argument(
        "--command-exit-code",
        action="append",
        default=[],
        metavar="GATE=CODE",
    )
    parser.add_argument(
        "--verify-inputs",
        action="store_true",
        help=(
            "Recheck a retained or fresh exact collection, semantic JUnit facts, "
            "and byte-identical reproductions."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the machine-validation refresh/verify CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.collect_pytest_inventory:
            collection_only_invalid = (
                args.visual_record_path is not None
                or args.junit_path is not None
                or args.paired_reproduction_path is not None
                or args.robustness_reproduction_path is not None
                or args.source_git_commit is not None
                or bool(args.command_exit_code)
                or args.verify_inputs
            )
            if collection_only_invalid:
                raise MachineValidationError(
                    "release record inputs are not valid with --collect-pytest-inventory"
                )
            if args.collection_path is None:
                raise MachineValidationError(
                    "--collect-pytest-inventory requires --collection-path"
                )
            collection = collect_pytest_inventory(
                args.repository_root,
                collection_path=args.collection_path,
            )
            print(
                "Exact release pytest collection inventory written: "
                f"{collection['tests']} testcases."
            )
        elif args.refresh:
            if args.verify_inputs:
                raise MachineValidationError("--verify-inputs is valid only with --verify")
            missing = [
                option
                for option, value in (
                    ("--collection-path", args.collection_path),
                    ("--junit-path", args.junit_path),
                    ("--paired-reproduction-path", args.paired_reproduction_path),
                    (
                        "--robustness-reproduction-path",
                        args.robustness_reproduction_path,
                    ),
                    ("--source-git-commit", args.source_git_commit),
                )
                if value is None
            ]
            if missing:
                raise MachineValidationError(
                    "--refresh is missing required inputs: " + ", ".join(missing)
                )
            exit_codes = _parse_command_exit_codes(args.command_exit_code)
            refresh_release_machine_validation(
                args.repository_root,
                record_path=args.record_path,
                visual_record_path=(
                    DEFAULT_VISUAL_RECORD_PATH
                    if args.visual_record_path is None
                    else args.visual_record_path
                ),
                collection_path=args.collection_path,
                junit_path=args.junit_path,
                paired_reproduction_path=args.paired_reproduction_path,
                robustness_reproduction_path=args.robustness_reproduction_path,
                command_exit_codes=exit_codes,
                source_git_commit=args.source_git_commit,
            )
            print(
                "Unsigned AI-generated release machine-validation record refreshed; "
                "command exit codes are caller-supplied and package/runtime facts remain "
                "in RELEASE_MANIFEST.windows_build."
            )
        else:
            refresh_only_supplied = (
                args.visual_record_path is not None
                or args.collection_path is not None
                or args.junit_path is not None
                or args.paired_reproduction_path is not None
                or args.robustness_reproduction_path is not None
                or args.source_git_commit is not None
                or bool(args.command_exit_code)
            )
            if refresh_only_supplied:
                raise MachineValidationError(
                    "visual/collection/JUnit/reproduction paths and command exit codes are valid "
                    "only with --refresh"
                )
            verify_release_machine_validation(
                args.repository_root,
                record_path=args.record_path,
                verify_inputs=args.verify_inputs,
            )
            if args.verify_inputs:
                print(
                    "Historical record/artifact verification: release machine-validation record, "
                    "committed artifacts, and fresh/retained transient inputs verified; "
                    "the record remains unsigned; current-source release requires "
                    "scripts/build-presentation.ps1 -VerifyOnly."
                )
            else:
                print(
                    "Historical record/artifact verification: release machine-validation record "
                    "and committed artifacts verified "
                    "in self-contained mode; --verify-inputs compares an exact collection, "
                    "semantic JUnit facts, and byte-identical reproductions; "
                    "current-source release requires scripts/build-presentation.ps1 -VerifyOnly."
                )
    except MachineValidationError as error:
        print(f"Release machine-validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
