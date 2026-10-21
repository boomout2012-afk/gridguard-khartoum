"""Create and verify the final release visual-review attestation.

The automated checks in this module bind a prior AI-agent content and visual
inspection to specific release artifacts. They validate hashes, paths, and
document structure; they do not perform, repeat, or independently prove that
content or visual inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

if __package__:
    from . import release_artifact_inspection as _artifact_inspection
else:
    import release_artifact_inspection as _artifact_inspection

SCHEMA_NAME = "gridguard.release_visual_review"
SCHEMA_VERSION = 2
RECORD_ROLE = "ai_agent_visual_and_content_inspection_attestation"
EXPECTED_PDF_PAGE_COUNT = 39
EXPECTED_PRESENTATION_SLIDE_COUNT = 13
EXPECTED_SPEAKER_NOTE_SLIDE_COUNT = 13
EXPECTED_APPLICATION_SCREENSHOT_COUNT = 3
AGENT_INSPECTION_STATUS = "agent_inspected_pass"
AGENT_CONTENT_INSPECTION_STATUS = "agent_content_inspected_pass"
AUTOMATED_NOTE_STATUS = "automated_present_nonempty"
COMPLETED_INSPECTION_STATUS = "completed_pass"
DEFAULT_REVIEWER = "Codex root agent"

DEFAULT_PDF_PATH = Path("output/pdf/gridguard-khartoum-final-report.pdf")
DEFAULT_PRESENTATION_PATH = Path("output/presentation/gridguard-khartoum-defense.pptx")
DEFAULT_RECORD_PATH = Path("docs/release-visual-review.json")
APPLICATION_SCREENSHOT_PATHS = (
    Path("docs/assets/screenshots/overview.png"),
    Path("docs/assets/screenshots/live.png"),
    Path("docs/assets/screenshots/results.png"),
)

AUTOMATED_VALIDATION_SCOPE = (
    "This validator checks artifact hashes, PDF/PPTX structure and counts, speaker-note "
    "presence and raw-content bindings, application-screenshot path/hash bindings, canonical "
    "schema, and complete recorded statuses; it does not inspect meaning or appearance and "
    "does not perform or independently prove AI-agent content or visual inspection."
)
INSPECTION_METHOD = (
    "rendered PDF pages, rendered presentation slides, all speaker-note contents, and all three "
    "application screenshots reviewed one-by-one in the active release session"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SPEAKER_NOTE_PART_PATH = re.compile(r"^ppt/notesSlides/notesSlide([1-9][0-9]*)\.xml$")


class ReleaseVisualReviewError(ValueError):
    """Raised when the release visual-review record cannot be trusted."""


def canonical_release_visual_review_bytes(payload: object) -> bytes:
    """Encode one deterministic, human-readable canonical JSON document."""
    try:
        text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ReleaseVisualReviewError(
            f"release visual-review record is not JSON-serializable: {error}"
        ) from error
    return f"{text}\n".encode()


def _reject_duplicate_json_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseVisualReviewError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ReleaseVisualReviewError(f"invalid non-finite JSON constant: {value}")


def _require_exact_fields(
    value: object,
    expected: set[str],
    *,
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReleaseVisualReviewError(f"{label} must be a JSON object")
    actual = set(value)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ReleaseVisualReviewError(f"{label} has unknown fields: {unknown}")
    if missing:
        raise ReleaseVisualReviewError(f"{label} is missing fields: {missing}")
    return value


def _require_exact_value(
    value: object,
    expected: object,
    *,
    label: str,
) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ReleaseVisualReviewError(f"{label} must be {expected!r}")


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ReleaseVisualReviewError(
            f"{label} must be a nonempty string without surrounding whitespace"
        )
    if any(ord(character) < 32 for character in value):
        raise ReleaseVisualReviewError(f"{label} must not contain control characters")
    return value


def _require_positive_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReleaseVisualReviewError(f"{label} must be a positive integer")
    return value


def _validate_inspection_rows(
    rows: object,
    *,
    count: int,
    item_label: str,
    index_field: str,
    status_field: str,
    status: str,
) -> None:
    if not isinstance(rows, list):
        raise ReleaseVisualReviewError(f"{item_label} inspection rows must be a list")
    if len(rows) != count:
        raise ReleaseVisualReviewError(
            f"{item_label} inspection rows must contain exactly {count} entries"
        )
    for expected_index, row in enumerate(rows, start=1):
        mapping = _require_exact_fields(
            row,
            {index_field, status_field},
            label=f"{item_label} inspection row {expected_index}",
        )
        _require_exact_value(
            mapping[index_field],
            expected_index,
            label=f"{item_label} inspection index {expected_index}",
        )
        _require_exact_value(
            mapping[status_field],
            status,
            label=f"{item_label} inspection status {expected_index}",
        )


def _validate_binding_fields(
    binding: Mapping[str, object],
    *,
    label: str,
) -> None:
    path = _require_nonempty_string(binding["path"], label=f"{label} path")
    if "\\" in path or path.startswith("/") or ".." in Path(path).parts:
        raise ReleaseVisualReviewError(
            f"{label} path must be a normalized repository-relative POSIX path"
        )
    sha256 = binding["sha256"]
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise ReleaseVisualReviewError(f"{label} sha256 must be lowercase hexadecimal")
    _require_positive_integer(binding["size_bytes"], label=f"{label} size_bytes")


def _validate_speaker_note_rows(rows: object) -> None:
    if not isinstance(rows, list):
        raise ReleaseVisualReviewError("speaker-note inspection rows must be a list")
    if len(rows) != EXPECTED_SPEAKER_NOTE_SLIDE_COUNT:
        raise ReleaseVisualReviewError(
            "speaker-note inspection rows must contain exactly "
            f"{EXPECTED_SPEAKER_NOTE_SLIDE_COUNT} entries"
        )
    note_paths: list[str] = []
    for expected_index, row in enumerate(rows, start=1):
        label = f"speaker-note inspection row {expected_index}"
        mapping = _require_exact_fields(
            row,
            {
                "content_inspection_status",
                "path",
                "sha256",
                "size_bytes",
                "slide_index",
                "structural_inspection_status",
            },
            label=label,
        )
        _require_exact_value(
            mapping["slide_index"],
            expected_index,
            label=f"speaker-note inspection index {expected_index}",
        )
        _validate_binding_fields(mapping, label=f"speaker-note {expected_index}")
        note_path = mapping["path"]
        assert isinstance(note_path, str)
        if _SPEAKER_NOTE_PART_PATH.fullmatch(note_path) is None:
            raise ReleaseVisualReviewError(
                f"speaker-note path {expected_index} must be a canonical note-slide part"
            )
        note_paths.append(note_path)
        _require_exact_value(
            mapping["structural_inspection_status"],
            AUTOMATED_NOTE_STATUS,
            label=f"speaker-note structural inspection status {expected_index}",
        )
        _require_exact_value(
            mapping["content_inspection_status"],
            AGENT_CONTENT_INSPECTION_STATUS,
            label=f"speaker-note content inspection status {expected_index}",
        )
    expected_note_paths = {
        f"ppt/notesSlides/notesSlide{index}.xml"
        for index in range(1, EXPECTED_SPEAKER_NOTE_SLIDE_COUNT + 1)
    }
    if len(set(note_paths)) != len(note_paths) or set(note_paths) != expected_note_paths:
        raise ReleaseVisualReviewError(
            "speaker-note paths must be unique and cover every canonical note-slide part"
        )


def _validate_application_screenshot_rows(rows: object) -> None:
    if not isinstance(rows, list):
        raise ReleaseVisualReviewError("application screenshot inspection rows must be a list")
    if len(rows) != EXPECTED_APPLICATION_SCREENSHOT_COUNT:
        raise ReleaseVisualReviewError(
            "application screenshot inspection rows must contain exactly "
            f"{EXPECTED_APPLICATION_SCREENSHOT_COUNT} entries"
        )
    for expected_index, (row, expected_path) in enumerate(
        zip(rows, APPLICATION_SCREENSHOT_PATHS, strict=True),
        start=1,
    ):
        label = f"application screenshot inspection row {expected_index}"
        mapping = _require_exact_fields(
            row,
            {
                "path",
                "screenshot_index",
                "sha256",
                "size_bytes",
                "visual_inspection_status",
            },
            label=label,
        )
        _require_exact_value(
            mapping["screenshot_index"],
            expected_index,
            label=f"application screenshot inspection index {expected_index}",
        )
        _validate_binding_fields(mapping, label=f"application screenshot {expected_index}")
        _require_exact_value(
            mapping["path"],
            expected_path.as_posix(),
            label=f"application screenshot path {expected_index}",
        )
        _require_exact_value(
            mapping["visual_inspection_status"],
            AGENT_INSPECTION_STATUS,
            label=f"application screenshot visual inspection status {expected_index}",
        )


def _validate_release_visual_review(record: object) -> Mapping[str, object]:
    top = _require_exact_fields(
        record,
        {
            "artifacts",
            "automated_validation_scope",
            "inspection_attestation",
            "record_role",
            "schema_name",
            "schema_version",
        },
        label="release visual-review record",
    )
    _require_exact_value(top["schema_name"], SCHEMA_NAME, label="schema_name")
    _require_exact_value(top["schema_version"], SCHEMA_VERSION, label="schema_version")
    _require_exact_value(top["record_role"], RECORD_ROLE, label="record_role")
    _require_exact_value(
        top["automated_validation_scope"],
        AUTOMATED_VALIDATION_SCOPE,
        label="automated_validation_scope",
    )

    attestation = _require_exact_fields(
        top["inspection_attestation"],
        {
            "human_visual_inspection_claimed",
            "inspection_method",
            "inspection_status",
            "reviewer",
            "reviewer_kind",
        },
        label="inspection_attestation",
    )
    _require_nonempty_string(attestation["reviewer"], label="attestation reviewer")
    _require_exact_value(
        attestation["reviewer_kind"], "ai_agent", label="attestation reviewer_kind"
    )
    _require_exact_value(
        attestation["inspection_method"],
        INSPECTION_METHOD,
        label="attestation inspection_method",
    )
    _require_exact_value(
        attestation["inspection_status"],
        COMPLETED_INSPECTION_STATUS,
        label="attestation inspection_status",
    )
    _require_exact_value(
        attestation["human_visual_inspection_claimed"],
        False,
        label="attestation human_visual_inspection_claimed",
    )

    artifacts = _require_exact_fields(
        top["artifacts"],
        {"application_screenshots", "pdf", "presentation"},
        label="artifacts",
    )
    pdf = _require_exact_fields(
        artifacts["pdf"],
        {"detected_page_count", "pages", "path", "sha256", "size_bytes"},
        label="PDF artifact",
    )
    _validate_binding_fields(pdf, label="PDF artifact")
    _require_exact_value(
        pdf["detected_page_count"],
        EXPECTED_PDF_PAGE_COUNT,
        label="PDF detected_page_count",
    )
    _validate_inspection_rows(
        pdf["pages"],
        count=EXPECTED_PDF_PAGE_COUNT,
        item_label="PDF page",
        index_field="page_index",
        status_field="visual_inspection_status",
        status=AGENT_INSPECTION_STATUS,
    )

    presentation = _require_exact_fields(
        artifacts["presentation"],
        {
            "detected_slide_count",
            "detected_speaker_note_slide_count",
            "path",
            "sha256",
            "size_bytes",
            "slides",
            "speaker_note_slides",
        },
        label="presentation artifact",
    )
    _validate_binding_fields(presentation, label="presentation artifact")
    _require_exact_value(
        presentation["detected_slide_count"],
        EXPECTED_PRESENTATION_SLIDE_COUNT,
        label="presentation detected_slide_count",
    )
    _require_exact_value(
        presentation["detected_speaker_note_slide_count"],
        EXPECTED_SPEAKER_NOTE_SLIDE_COUNT,
        label="presentation detected_speaker_note_slide_count",
    )
    _validate_inspection_rows(
        presentation["slides"],
        count=EXPECTED_PRESENTATION_SLIDE_COUNT,
        item_label="presentation slide",
        index_field="slide_index",
        status_field="visual_inspection_status",
        status=AGENT_INSPECTION_STATUS,
    )
    _validate_speaker_note_rows(presentation["speaker_note_slides"])

    application_screenshots = _require_exact_fields(
        artifacts["application_screenshots"],
        {"detected_screenshot_count", "screenshots"},
        label="application screenshots",
    )
    _require_exact_value(
        application_screenshots["detected_screenshot_count"],
        EXPECTED_APPLICATION_SCREENSHOT_COUNT,
        label="detected application screenshot count",
    )
    _validate_application_screenshot_rows(application_screenshots["screenshots"])
    return top


def _resolve_repository_path(
    repository_root: Path,
    supplied_path: str | Path | None,
    default_path: Path,
    *,
    label: str,
) -> tuple[Path, str]:
    root = repository_root.resolve()
    requested = default_path if supplied_path is None else Path(supplied_path)
    resolved = (root / requested).resolve() if not requested.is_absolute() else requested.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ReleaseVisualReviewError(
            f"{label} must resolve inside the repository root"
        ) from error
    return resolved, relative.as_posix()


def _inspect_pdf(path: Path, relative_path: str) -> dict[str, object]:
    try:
        return _artifact_inspection.inspect_pdf_artifact(
            path,
            relative_path,
            expected_page_count=EXPECTED_PDF_PAGE_COUNT,
        )
    except _artifact_inspection.ReleaseArtifactInspectionError as error:
        raise ReleaseVisualReviewError(str(error)) from error


def _speaker_note_bindings(
    path: Path,
    presentation: Mapping[str, object],
    resolved_note_parts: Sequence[str],
) -> list[dict[str, object]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReleaseVisualReviewError(f"cannot read final presentation {path}: {error}") from error
    if (
        len(raw) != presentation["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != presentation["sha256"]
    ):
        raise ReleaseVisualReviewError("final presentation changed during inspection")
    expected_note_parts = {
        f"ppt/notesSlides/notesSlide{index}.xml"
        for index in range(1, EXPECTED_SPEAKER_NOTE_SLIDE_COUNT + 1)
    }
    if (
        len(resolved_note_parts) != EXPECTED_SPEAKER_NOTE_SLIDE_COUNT
        or any(
            _SPEAKER_NOTE_PART_PATH.fullmatch(part_path) is None
            for part_path in resolved_note_parts
        )
        or len(set(resolved_note_parts)) != len(resolved_note_parts)
        or set(resolved_note_parts) != expected_note_parts
    ):
        raise ReleaseVisualReviewError(
            "validated speaker-note paths must be canonical, unique, and complete"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            bindings = []
            for slide_index, part_path in enumerate(resolved_note_parts, start=1):
                note_raw = archive.read(part_path)
                bindings.append(
                    {
                        "slide_index": slide_index,
                        "path": part_path,
                        "sha256": hashlib.sha256(note_raw).hexdigest(),
                        "size_bytes": len(note_raw),
                    }
                )
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ReleaseVisualReviewError(
            f"cannot bind final presentation speaker-note contents: {error}"
        ) from error
    return bindings


def _inspect_presentation(
    path: Path,
    relative_path: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        presentation = _artifact_inspection.inspect_presentation_artifact(
            path,
            relative_path,
            expected_slide_count=EXPECTED_PRESENTATION_SLIDE_COUNT,
            expected_note_count=EXPECTED_SPEAKER_NOTE_SLIDE_COUNT,
        )
    except _artifact_inspection.ReleaseArtifactInspectionError as error:
        raise ReleaseVisualReviewError(str(error)) from error
    resolved_note_parts = presentation.pop("resolved_speaker_note_parts", None)
    if not isinstance(resolved_note_parts, list) or not all(
        isinstance(part_path, str) for part_path in resolved_note_parts
    ):
        raise ReleaseVisualReviewError(
            "presentation inspection did not return validated speaker-note relationship paths"
        )
    return presentation, _speaker_note_bindings(
        path,
        presentation,
        resolved_note_parts,
    )


def _inspect_application_screenshots(
    repository_root: Path,
) -> tuple[list[dict[str, object]], list[Path]]:
    bindings: list[dict[str, object]] = []
    resolved_paths: list[Path] = []
    for screenshot_index, expected_path in enumerate(APPLICATION_SCREENSHOT_PATHS, start=1):
        resolved, relative_path = _resolve_repository_path(
            repository_root,
            expected_path,
            expected_path,
            label=f"application screenshot {screenshot_index}",
        )
        if relative_path != expected_path.as_posix():
            raise ReleaseVisualReviewError(
                f"application screenshot {screenshot_index} must resolve exactly to "
                f"{expected_path.as_posix()}"
            )
        try:
            raw = resolved.read_bytes()
        except OSError as error:
            raise ReleaseVisualReviewError(
                f"cannot read application screenshot {resolved}: {error}"
            ) from error
        if not raw:
            raise ReleaseVisualReviewError(
                f"application screenshot must be a nonempty file: {resolved}"
            )
        bindings.append(
            {
                "screenshot_index": screenshot_index,
                "path": relative_path,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
        resolved_paths.append(resolved)
    return bindings, resolved_paths


def _record_paths(
    repository_root: str | Path,
    *,
    record_path: str | Path | None,
    pdf_path: str | Path | None,
    presentation_path: str | Path | None,
) -> tuple[Path, Path, str, Path, str]:
    root = Path(repository_root).resolve()
    if not root.is_dir():
        raise ReleaseVisualReviewError(f"repository root is not a directory: {root}")
    resolved_record, _ = _resolve_repository_path(
        root, record_path, DEFAULT_RECORD_PATH, label="record path"
    )
    resolved_pdf, relative_pdf = _resolve_repository_path(
        root, pdf_path, DEFAULT_PDF_PATH, label="PDF path"
    )
    resolved_presentation, relative_presentation = _resolve_repository_path(
        root,
        presentation_path,
        DEFAULT_PRESENTATION_PATH,
        label="presentation path",
    )
    resolved_paths = {
        resolved_record,
        resolved_pdf,
        resolved_presentation,
    }
    if len(resolved_paths) != 3:
        raise ReleaseVisualReviewError(
            "record, PDF, and presentation paths must resolve to distinct files"
        )
    return (
        resolved_record,
        resolved_pdf,
        relative_pdf,
        resolved_presentation,
        relative_presentation,
    )


def _require_distinct_release_paths(paths: Sequence[Path]) -> None:
    if len(set(paths)) != len(paths):
        raise ReleaseVisualReviewError(
            "record, PDF, presentation, and application screenshot paths must resolve to "
            "distinct files"
        )


def _write_atomic(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except OSError as error:
        raise ReleaseVisualReviewError(f"cannot write record {path}: {error}") from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def refresh_release_visual_review(
    repository_root: str | Path,
    *,
    record_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    presentation_path: str | Path | None = None,
    reviewer: str = DEFAULT_REVIEWER,
    inspection_confirmed: bool = False,
) -> dict[str, object]:
    """Write a record only after explicit confirmation of the prior AI review."""
    if inspection_confirmed is not True:
        raise ReleaseVisualReviewError(
            "refresh requires explicit confirmation that the AI-agent content and visual "
            "inspection was completed"
        )
    reviewer = _require_nonempty_string(reviewer, label="reviewer")
    (
        resolved_record,
        resolved_pdf,
        relative_pdf,
        resolved_presentation,
        relative_presentation,
    ) = _record_paths(
        repository_root,
        record_path=record_path,
        pdf_path=pdf_path,
        presentation_path=presentation_path,
    )
    pdf = _inspect_pdf(resolved_pdf, relative_pdf)
    presentation, speaker_note_bindings = _inspect_presentation(
        resolved_presentation, relative_presentation
    )
    screenshot_bindings, screenshot_paths = _inspect_application_screenshots(
        Path(repository_root).resolve()
    )
    _require_distinct_release_paths(
        [resolved_record, resolved_pdf, resolved_presentation, *screenshot_paths]
    )
    record: dict[str, object] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "record_role": RECORD_ROLE,
        "automated_validation_scope": AUTOMATED_VALIDATION_SCOPE,
        "inspection_attestation": {
            "reviewer": reviewer,
            "reviewer_kind": "ai_agent",
            "inspection_method": INSPECTION_METHOD,
            "inspection_status": COMPLETED_INSPECTION_STATUS,
            "human_visual_inspection_claimed": False,
        },
        "artifacts": {
            "pdf": {
                **pdf,
                "pages": [
                    {
                        "page_index": index,
                        "visual_inspection_status": AGENT_INSPECTION_STATUS,
                    }
                    for index in range(1, EXPECTED_PDF_PAGE_COUNT + 1)
                ],
            },
            "presentation": {
                **presentation,
                "slides": [
                    {
                        "slide_index": index,
                        "visual_inspection_status": AGENT_INSPECTION_STATUS,
                    }
                    for index in range(1, EXPECTED_PRESENTATION_SLIDE_COUNT + 1)
                ],
                "speaker_note_slides": [
                    {
                        **binding,
                        "structural_inspection_status": AUTOMATED_NOTE_STATUS,
                        "content_inspection_status": AGENT_CONTENT_INSPECTION_STATUS,
                    }
                    for binding in speaker_note_bindings
                ],
            },
            "application_screenshots": {
                "detected_screenshot_count": EXPECTED_APPLICATION_SCREENSHOT_COUNT,
                "screenshots": [
                    {
                        **binding,
                        "visual_inspection_status": AGENT_INSPECTION_STATUS,
                    }
                    for binding in screenshot_bindings
                ],
            },
        },
    }
    _validate_release_visual_review(record)
    _write_atomic(resolved_record, canonical_release_visual_review_bytes(record))
    return record


def _load_record(path: Path) -> Mapping[str, object]:
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise ReleaseVisualReviewError(f"cannot read record {path}: {error}") from error
    try:
        decoded = encoded.decode("utf-8")
        loaded = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_json_fields,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except ReleaseVisualReviewError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseVisualReviewError(
            f"release visual-review record is invalid JSON: {error}"
        ) from error
    if canonical_release_visual_review_bytes(loaded) != encoded:
        raise ReleaseVisualReviewError("release visual-review record is not canonical JSON")
    return _validate_release_visual_review(loaded)


def _require_current_binding(
    recorded: Mapping[str, object],
    current: Mapping[str, object],
    *,
    label: str,
) -> None:
    for field in current:
        if recorded[field] != current[field]:
            raise ReleaseVisualReviewError(
                f"{label} binding mismatch for {field}: "
                f"recorded {recorded[field]!r}, current {current[field]!r}"
            )


def _require_current_row_bindings(
    recorded: object,
    current: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> None:
    if not isinstance(recorded, list) or len(recorded) != len(current):
        raise ReleaseVisualReviewError(f"{label} row count mismatch")
    for row_index, (recorded_row, current_row) in enumerate(
        zip(recorded, current, strict=True),
        start=1,
    ):
        if not isinstance(recorded_row, Mapping):
            raise ReleaseVisualReviewError(f"{label} row {row_index} must be a JSON object")
        for field, current_value in current_row.items():
            if recorded_row[field] != current_value:
                raise ReleaseVisualReviewError(
                    f"{label} mismatch at row {row_index} for {field}: "
                    f"recorded {recorded_row[field]!r}, current {current_value!r}"
                )


def verify_release_visual_review(
    repository_root: str | Path,
    *,
    record_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    presentation_path: str | Path | None = None,
) -> dict[str, object]:
    """Verify record bytes, review coverage, artifact bindings, and structure."""
    (
        resolved_record,
        resolved_pdf,
        relative_pdf,
        resolved_presentation,
        relative_presentation,
    ) = _record_paths(
        repository_root,
        record_path=record_path,
        pdf_path=pdf_path,
        presentation_path=presentation_path,
    )
    screenshot_bindings, screenshot_paths = _inspect_application_screenshots(
        Path(repository_root).resolve()
    )
    _require_distinct_release_paths(
        [resolved_record, resolved_pdf, resolved_presentation, *screenshot_paths]
    )
    record = _load_record(resolved_record)
    artifacts = record["artifacts"]
    assert isinstance(artifacts, Mapping)
    recorded_pdf = artifacts["pdf"]
    recorded_presentation = artifacts["presentation"]
    recorded_screenshots = artifacts["application_screenshots"]
    assert isinstance(recorded_pdf, Mapping)
    assert isinstance(recorded_presentation, Mapping)
    assert isinstance(recorded_screenshots, Mapping)
    _require_current_binding(
        recorded_pdf,
        _inspect_pdf(resolved_pdf, relative_pdf),
        label="PDF",
    )
    current_presentation, current_speaker_notes = _inspect_presentation(
        resolved_presentation, relative_presentation
    )
    _require_current_binding(
        recorded_presentation,
        current_presentation,
        label="presentation",
    )
    _require_current_row_bindings(
        recorded_presentation["speaker_note_slides"],
        current_speaker_notes,
        label="speaker-note binding",
    )
    _require_current_row_bindings(
        recorded_screenshots["screenshots"],
        screenshot_bindings,
        label="application screenshot binding",
    )
    return dict(record)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh or verify the final release visual-review attestation. "
            "Automated verification does not perform content or visual inspection."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--refresh", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--record-path", type=Path)
    parser.add_argument("--pdf-path", type=Path)
    parser.add_argument("--presentation-path", type=Path)
    parser.add_argument(
        "--reviewer",
        help=f"AI-agent reviewer label for --refresh (default: {DEFAULT_REVIEWER}).",
    )
    parser.add_argument(
        "--confirm-agent-content-and-visual-inspection-completed",
        action="store_true",
        help=(
            "Required for refresh; confirms prior AI-agent review of pages, slides, "
            "speaker-note contents, and application screenshots, not a human claim."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the refresh/verify command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.refresh:
            reviewer = DEFAULT_REVIEWER if args.reviewer is None else args.reviewer
            refresh_release_visual_review(
                args.repository_root,
                record_path=args.record_path,
                pdf_path=args.pdf_path,
                presentation_path=args.presentation_path,
                reviewer=reviewer,
                inspection_confirmed=args.confirm_agent_content_and_visual_inspection_completed,
            )
            print(
                "Release visual-review record refreshed: recorded the confirmed "
                "AI-agent content and visual inspection; this is not a human "
                "visual-inspection claim."
            )
        else:
            if args.confirm_agent_content_and_visual_inspection_completed:
                raise ReleaseVisualReviewError(
                    "the inspection-confirmation flag is valid only with --refresh"
                )
            if args.reviewer is not None:
                raise ReleaseVisualReviewError("--reviewer is valid only with --refresh")
            verify_release_visual_review(
                args.repository_root,
                record_path=args.record_path,
                pdf_path=args.pdf_path,
                presentation_path=args.presentation_path,
            )
            print(
                "Historical record/artifact verification: release visual-review hashes, paths, "
                "structure, counts, and recorded statuses verified; automated verification does "
                "not independently repeat content or visual inspection; current-source release "
                "requires scripts/build-presentation.ps1 -VerifyOnly."
            )
    except ReleaseVisualReviewError as error:
        print(f"Release visual-review validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
