from __future__ import annotations

import hashlib
import json
import re
import stat
import struct
import zipfile
import zlib
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
    RectangleObject,
    TextStringObject,
)

from tools.release_artifact_inspection import (
    inspect_pdf_artifact,
    inspect_presentation_artifact,
)
from tools.release_validation import (
    AGENT_CONTENT_INSPECTION_STATUS,
    AGENT_INSPECTION_STATUS,
    APPLICATION_SCREENSHOT_PATHS,
    AUTOMATED_NOTE_STATUS,
    EXPECTED_APPLICATION_SCREENSHOT_COUNT,
    EXPECTED_PDF_PAGE_COUNT,
    EXPECTED_PRESENTATION_SLIDE_COUNT,
    ReleaseVisualReviewError,
    canonical_release_visual_review_bytes,
    main,
    refresh_release_visual_review,
    verify_release_visual_review,
)

PDF_RELATIVE_PATH = Path("output/pdf/gridguard-khartoum-final-report.pdf")
PRESENTATION_RELATIVE_PATH = Path("output/presentation/gridguard-khartoum-defense.pptx")
RECORD_RELATIVE_PATH = Path("docs/release-visual-review.json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_RELATIONSHIPS_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_RELATIONSHIPS_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PRESENTATION_NAMESPACE = "http://schemas.openxmlformats.org/presentationml/2006/main"
DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


VALID_PNG = (
    b"\x89PNG\r\n\x1a\n"
    + _png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0),
    )
    + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00\x00"))
    + _png_chunk(b"IEND", b"")
)


def _write_pdf(
    repository_root: Path,
    *,
    page_count: int = EXPECTED_PDF_PAGE_COUNT,
    attachment: bool = False,
    javascript: bool = False,
    uri_action: bool = False,
    orphan_javascript: bool = False,
    orphan_filespec: bool = False,
    orphan_postscript: bool = False,
    external_stream: bool = False,
) -> Path:
    path = repository_root / PDF_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    if attachment:
        writer.add_attachment("reviewer-payload.txt", b"unexpected attachment")
    if javascript:
        writer.add_js("app.alert('unexpected active content')")
    if uri_action:
        writer.add_uri(
            0,
            "https://attacker.invalid/payload",
            RectangleObject((0, 0, 10, 10)),
        )
    if orphan_javascript:
        writer._add_object(  # noqa: SLF001 - constructs an intentionally orphaned PDF object.
            DictionaryObject(
                {
                    NameObject("/S"): NameObject("/JavaScript"),
                    NameObject("/JS"): TextStringObject("unexpected active content"),
                }
            )
        )
    if orphan_filespec:
        writer._add_object(  # noqa: SLF001 - constructs an intentionally orphaned PDF object.
            DictionaryObject({NameObject("/Type"): NameObject("/Filespec")})
        )
    if orphan_postscript:
        stream = DecodedStreamObject()
        stream.set_data(b"showpage")
        stream[NameObject("/Type")] = NameObject("/XObject")
        stream[NameObject("/Subtype")] = NameObject("/PS")
        writer._add_object(stream)  # noqa: SLF001 - intentionally orphaned PostScript.
    if external_stream:
        stream = DecodedStreamObject()
        stream.set_data(b"")
        stream[NameObject("/F")] = TextStringObject("C:/Windows/System32/calc.exe")
        writer.pages[0][NameObject("/Contents")] = writer._add_object(  # noqa: SLF001
            stream
        )
    with path.open("wb") as stream:
        writer.write(stream)
    return path


def _relationship_xml(
    note_index: int,
    *,
    namespace: str = PACKAGE_RELATIONSHIPS_NAMESPACE,
    relationship_type: str | None = None,
) -> str:
    relationship_type = (
        f"{OFFICE_RELATIONSHIPS_NAMESPACE}/notesSlide"
        if relationship_type is None
        else relationship_type
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<Relationships xmlns="{namespace}">\n'
        f'  <Relationship Id="rIdNotes" Type="{relationship_type}" '
        f'Target="../notesSlides/notesSlide{note_index}.xml"/>\n'
        "</Relationships>\n"
    )


def _note_relationships_xml(note_index: int, *, slide_index: int | None = None) -> str:
    slide_index = note_index if slide_index is None else slide_index
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<Relationships xmlns="{PACKAGE_RELATIONSHIPS_NAMESPACE}">\n'
        f'  <Relationship Id="rIdSlide" Type="{OFFICE_RELATIONSHIPS_NAMESPACE}/slide" '
        f'Target="../slides/slide{slide_index}.xml"/>\n'
        "</Relationships>\n"
    )


def _notes_xml(
    note_index: int,
    *,
    presentation_namespace: str = PRESENTATION_NAMESPACE,
    drawing_namespace: str = DRAWING_NAMESPACE,
    placeholder_type: str = "body",
) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<p:notes xmlns:p="{presentation_namespace}" '
        f'xmlns:a="{drawing_namespace}">\n'
        "  <p:cSld><p:spTree><p:sp><p:nvSpPr><p:nvPr>"
        f'<p:ph type="{placeholder_type}"/>'
        "</p:nvPr></p:nvSpPr><p:txBody><a:p><a:r>"
        f"<a:t>Speaker note {note_index}</a:t>"
        "</a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>\n"
        "</p:notes>\n"
    )


def _content_types_xml(slide_count: int) -> str:
    overrides = [
        (
            "/ppt/presentation.xml",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml",
        )
    ]
    overrides.extend(
        (
            f"/ppt/slides/slide{index}.xml",
            "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
        )
        for index in range(1, slide_count + 1)
    )
    overrides.extend(
        (
            f"/ppt/notesSlides/notesSlide{index}.xml",
            "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml",
        )
        for index in range(1, slide_count + 1)
    )
    serialized = "".join(
        f'<Override PartName="{part_name}" ContentType="{content_type}"/>'
        for part_name, content_type in overrides
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{serialized}</Types>"
    )


def _package_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Relationships xmlns="{PACKAGE_RELATIONSHIPS_NAMESPACE}">'
        f'<Relationship Id="rId1" Type="{OFFICE_RELATIONSHIPS_NAMESPACE}/'
        'officeDocument" Target="ppt/presentation.xml"/>'
        "</Relationships>"
    )


def _presentation_xml(referenced_slide_count: int) -> str:
    slide_ids = "".join(
        f'<p:sldId id="{255 + index}" r:id="rId{index}"/>'
        for index in range(1, referenced_slide_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<p:presentation xmlns:p="{PRESENTATION_NAMESPACE}" '
        f'xmlns:r="{OFFICE_RELATIONSHIPS_NAMESPACE}">'
        f"<p:sldIdLst>{slide_ids}</p:sldIdLst></p:presentation>"
    )


def _presentation_relationships_xml(referenced_slide_count: int) -> str:
    relationships = "".join(
        f'<Relationship Id="rId{index}" '
        f'Type="{OFFICE_RELATIONSHIPS_NAMESPACE}/slide" '
        f'Target="slides/slide{index}.xml"/>'
        for index in range(1, referenced_slide_count + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Relationships xmlns="{PACKAGE_RELATIONSHIPS_NAMESPACE}">'
        f"{relationships}</Relationships>"
    )


def _write_presentation(
    repository_root: Path,
    *,
    slide_count: int = EXPECTED_PRESENTATION_SLIDE_COUNT,
    referenced_slide_count: int | None = None,
    missing_note_relationship: int | None = None,
    empty_note: int | None = None,
    invalid_note_relationship: int | None = None,
    invalid_note_namespace: int | None = None,
    slide_number_only_note: int | None = None,
    omit_presentation_graph: bool = False,
) -> Path:
    _write_application_screenshots(repository_root)
    referenced_slide_count = (
        slide_count if referenced_slide_count is None else referenced_slide_count
    )
    path = repository_root / PRESENTATION_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml(slide_count))
        archive.writestr("_rels/.rels", _package_relationships_xml())
        if not omit_presentation_graph:
            archive.writestr(
                "ppt/presentation.xml",
                _presentation_xml(referenced_slide_count),
            )
            archive.writestr(
                "ppt/_rels/presentation.xml.rels",
                _presentation_relationships_xml(referenced_slide_count),
            )
        for index in range(1, slide_count + 1):
            archive.writestr(
                f"ppt/slides/slide{index}.xml",
                '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>',
            )
            if index != missing_note_relationship:
                archive.writestr(
                    f"ppt/slides/_rels/slide{index}.xml.rels",
                    _relationship_xml(
                        index,
                        namespace=(
                            "urn:not-opc"
                            if index == invalid_note_relationship
                            else PACKAGE_RELATIONSHIPS_NAMESPACE
                        ),
                        relationship_type=(
                            "https://attacker.invalid/notesSlide"
                            if index == invalid_note_relationship
                            else None
                        ),
                    ),
                )
            archive.writestr(
                f"ppt/notesSlides/notesSlide{index}.xml",
                (
                    '<p:notes xmlns:p="http://schemas.openxmlformats.org/'
                    'presentationml/2006/main"/>'
                    if index == empty_note
                    else _notes_xml(
                        index,
                        presentation_namespace=(
                            "urn:not-presentationml"
                            if index == invalid_note_namespace
                            else PRESENTATION_NAMESPACE
                        ),
                        drawing_namespace=(
                            "urn:not-drawingml"
                            if index == invalid_note_namespace
                            else DRAWING_NAMESPACE
                        ),
                        placeholder_type=("sldNum" if index == slide_number_only_note else "body"),
                    )
                ),
            )
            archive.writestr(
                f"ppt/notesSlides/_rels/notesSlide{index}.xml.rels",
                _note_relationships_xml(index),
            )
    return path


def _write_application_screenshots(repository_root: Path) -> list[Path]:
    paths: list[Path] = []
    for relative_path in APPLICATION_SCREENSHOT_PATHS:
        path = repository_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(VALID_PNG)
        paths.append(path)
    return paths


def _write_release_artifacts(repository_root: Path) -> tuple[Path, Path]:
    return _write_pdf(repository_root), _write_presentation(repository_root)


def _append_presentation_member(
    path: Path,
    member_name: str,
    content: bytes = b"unexpected payload",
    *,
    symbolic_link: bool = False,
) -> None:
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        if not symbolic_link:
            if "\\" not in member_name:
                archive.writestr(member_name, content)
            else:
                member = zipfile.ZipInfo("placeholder")
                member.filename = member_name
                member.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(member, content)
            return
        member = zipfile.ZipInfo(member_name)
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, content)


def _replace_presentation_member(path: Path, member_name: str, content: bytes) -> None:
    replacement = path.with_suffix(".replacement.pptx")
    with (
        zipfile.ZipFile(path) as source,
        zipfile.ZipFile(
            replacement,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as target,
    ):
        for member in source.infolist():
            target.writestr(
                member,
                content if member.filename == member_name else source.read(member),
            )
    replacement.replace(path)


def _rewrite_presentation_compression(path: Path, compression: int) -> None:
    replacement = path.with_suffix(".replacement.pptx")
    with (
        zipfile.ZipFile(path) as source,
        zipfile.ZipFile(replacement, "w", compression=compression) as target,
    ):
        for member in source.infolist():
            target.writestr(
                member.filename,
                source.read(member),
                compress_type=compression,
            )
    replacement.replace(path)


def _add_external_slide_relationship(path: Path) -> None:
    relationship_part = "ppt/slides/_rels/slide1.xml.rels"
    with zipfile.ZipFile(path) as archive:
        original = archive.read(relationship_part)
    external = (
        f'<Relationship Id="rIdExternal" Type="{OFFICE_RELATIONSHIPS_NAMESPACE}/hyperlink" '
        'Target="https://attacker.invalid/payload" TargetMode="External"/>'
    ).encode()
    _replace_presentation_member(
        path,
        relationship_part,
        original.replace(b"</Relationships>", external + b"</Relationships>"),
    )


def _add_referenced_slide_image(path: Path, content: bytes) -> None:
    relationship_part = "ppt/slides/_rels/slide1.xml.rels"
    with zipfile.ZipFile(path) as archive:
        original = archive.read(relationship_part)
    image_relationship = (
        f'<Relationship Id="rIdImage" Type="{OFFICE_RELATIONSHIPS_NAMESPACE}/image" '
        'Target="../media/image1.png"/>'
    ).encode()
    _replace_presentation_member(
        path,
        relationship_part,
        original.replace(
            b"</Relationships>",
            image_relationship + b"</Relationships>",
        ),
    )
    _append_presentation_member(path, "ppt/media/image1.png", content)


def _replace_local_header_member_name(
    path: Path,
    member_name: str,
    replacement: str,
) -> None:
    with zipfile.ZipFile(path) as archive:
        header_offset = archive.getinfo(member_name).header_offset
    raw = bytearray(path.read_bytes())
    assert raw[header_offset : header_offset + 4] == b"PK\x03\x04"
    name_length = struct.unpack_from("<H", raw, header_offset + 26)[0]
    replacement_bytes = replacement.encode("ascii")
    assert len(replacement_bytes) == name_length
    raw[header_offset + 30 : header_offset + 30 + name_length] = replacement_bytes
    path.write_bytes(raw)


def _tamper_local_header_padding(path: Path, member_name: str) -> None:
    with zipfile.ZipFile(path) as archive:
        header_offset = archive.getinfo(member_name).header_offset
    raw = bytearray(path.read_bytes())
    name_length, extra_length = struct.unpack_from("<HH", raw, header_offset + 26)
    assert extra_length > 8
    raw[header_offset + 30 + name_length + 8] = 1
    path.write_bytes(raw)


def _insert_unused_deflate_member_payload(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path) as archive:
        member = max(archive.infolist(), key=lambda candidate: candidate.header_offset)
        central_directory_offset = archive.start_dir
    assert member.compress_type == zipfile.ZIP_DEFLATED
    raw = bytearray(path.read_bytes())
    local_compressed_size = struct.unpack_from("<I", raw, member.header_offset + 18)[0]
    struct.pack_into(
        "<I",
        raw,
        member.header_offset + 18,
        local_compressed_size + len(payload),
    )
    raw[central_directory_offset:central_directory_offset] = payload

    central_cursor = central_directory_offset + len(payload)
    found = False
    while raw[central_cursor : central_cursor + 4] == b"PK\x01\x02":
        filename_length, extra_length, comment_length = struct.unpack_from(
            "<HHH",
            raw,
            central_cursor + 28,
        )
        filename_start = central_cursor + 46
        filename_end = filename_start + filename_length
        if raw[filename_start:filename_end].decode("ascii") == member.filename:
            central_compressed_size = struct.unpack_from("<I", raw, central_cursor + 20)[0]
            struct.pack_into(
                "<I",
                raw,
                central_cursor + 20,
                central_compressed_size + len(payload),
            )
            found = True
        central_cursor = filename_end + extra_length + comment_length
    assert found
    end_record_offset = len(raw) - 22
    assert raw[end_record_offset : end_record_offset + 4] == b"PK\x05\x06"
    struct.pack_into(
        "<I",
        raw,
        end_record_offset + 16,
        central_directory_offset + len(payload),
    )
    path.write_bytes(raw)


def _insert_unbound_pdf_bytes_before_xref(path: Path, payload: bytes) -> None:
    raw = path.read_bytes()
    match = re.search(rb"startxref\s+([0-9]+)\s+%%EOF\s*\Z", raw)
    assert match is not None
    old_xref_offset = int(match.group(1))
    new_xref_offset = old_xref_offset + len(payload)
    mutated = raw[:old_xref_offset] + payload + raw[old_xref_offset:]
    old_startxref = f"startxref\n{old_xref_offset}\n".encode()
    new_startxref = f"startxref\n{new_xref_offset}\n".encode()
    assert mutated.count(old_startxref) == 1
    path.write_bytes(mutated.replace(old_startxref, new_startxref, 1))


def _load_record(repository_root: Path) -> dict[str, object]:
    return json.loads((repository_root / RECORD_RELATIVE_PATH).read_text(encoding="utf-8"))


def _write_canonical_record(repository_root: Path, record: dict[str, object]) -> None:
    (repository_root / RECORD_RELATIVE_PATH).write_bytes(
        canonical_release_visual_review_bytes(record)
    )


def _speaker_note_binding(
    presentation_path: Path,
    slide_index: int,
    *,
    note_index: int | None = None,
) -> dict[str, object]:
    note_index = slide_index if note_index is None else note_index
    part_path = f"ppt/notesSlides/notesSlide{note_index}.xml"
    with zipfile.ZipFile(presentation_path) as archive:
        raw = archive.read(part_path)
    return {
        "slide_index": slide_index,
        "path": part_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "structural_inspection_status": AUTOMATED_NOTE_STATUS,
        "content_inspection_status": AGENT_CONTENT_INSPECTION_STATUS,
    }


def test_refresh_requires_explicit_agent_inspection_confirmation(tmp_path: Path) -> None:
    _write_release_artifacts(tmp_path)

    with pytest.raises(ReleaseVisualReviewError, match="confirmation"):
        refresh_release_visual_review(tmp_path)


def test_release_review_tracks_only_the_three_current_application_pages() -> None:
    assert EXPECTED_APPLICATION_SCREENSHOT_COUNT == 3
    expected_paths = (
        Path("docs/assets/screenshots/overview.png"),
        Path("docs/assets/screenshots/live.png"),
        Path("docs/assets/screenshots/results.png"),
    )
    assert expected_paths == APPLICATION_SCREENSHOT_PATHS


@pytest.mark.parametrize(
    "path_options",
    [
        {"record_path": PDF_RELATIVE_PATH},
        {"record_path": PRESENTATION_RELATIVE_PATH},
        {"record_path": APPLICATION_SCREENSHOT_PATHS[0]},
        {"pdf_path": PRESENTATION_RELATIVE_PATH},
    ],
)
def test_refresh_rejects_path_alias_without_overwriting_artifacts(
    tmp_path: Path, path_options: dict[str, Path]
) -> None:
    pdf_path, presentation_path = _write_release_artifacts(tmp_path)
    before = {
        pdf_path: pdf_path.read_bytes(),
        presentation_path: presentation_path.read_bytes(),
        **{
            tmp_path / relative_path: (tmp_path / relative_path).read_bytes()
            for relative_path in APPLICATION_SCREENSHOT_PATHS
        },
    }

    with pytest.raises(ReleaseVisualReviewError, match="distinct"):
        refresh_release_visual_review(
            tmp_path,
            inspection_confirmed=True,
            **path_options,
        )

    assert pdf_path.read_bytes() == before[pdf_path]
    assert presentation_path.read_bytes() == before[presentation_path]


def test_refresh_and_verify_bind_complete_honest_visual_review_record(
    tmp_path: Path,
) -> None:
    pdf_path, presentation_path = _write_release_artifacts(tmp_path)

    refreshed = refresh_release_visual_review(
        tmp_path,
        reviewer="Codex root agent",
        inspection_confirmed=True,
    )
    verified = verify_release_visual_review(tmp_path)
    record_path = tmp_path / RECORD_RELATIVE_PATH

    assert verified == refreshed
    assert record_path.read_bytes() == canonical_release_visual_review_bytes(refreshed)
    assert refreshed["schema_name"] == "gridguard.release_visual_review"
    assert refreshed["schema_version"] == 2
    assert refreshed["record_role"] == "ai_agent_visual_and_content_inspection_attestation"
    assert "does not inspect meaning or appearance" in refreshed["automated_validation_scope"]
    assert (
        "speaker-note presence and raw-content bindings" in refreshed["automated_validation_scope"]
    )
    assert "application-screenshot path/hash bindings" in refreshed["automated_validation_scope"]

    attestation = refreshed["inspection_attestation"]
    assert attestation == {
        "human_visual_inspection_claimed": False,
        "inspection_method": (
            "rendered PDF pages, rendered presentation slides, all speaker-note contents, "
            "and all three application screenshots reviewed one-by-one in the active "
            "release session"
        ),
        "inspection_status": "completed_pass",
        "reviewer": "Codex root agent",
        "reviewer_kind": "ai_agent",
    }

    artifacts = refreshed["artifacts"]
    pdf = artifacts["pdf"]
    assert pdf["path"] == PDF_RELATIVE_PATH.as_posix()
    assert pdf["sha256"] == hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    assert pdf["size_bytes"] == pdf_path.stat().st_size
    assert pdf["detected_page_count"] == EXPECTED_PDF_PAGE_COUNT
    assert pdf["pages"] == [
        {"page_index": index, "visual_inspection_status": AGENT_INSPECTION_STATUS}
        for index in range(1, EXPECTED_PDF_PAGE_COUNT + 1)
    ]

    presentation = artifacts["presentation"]
    assert presentation["path"] == PRESENTATION_RELATIVE_PATH.as_posix()
    assert presentation["sha256"] == hashlib.sha256(presentation_path.read_bytes()).hexdigest()
    assert presentation["size_bytes"] == presentation_path.stat().st_size
    assert presentation["detected_slide_count"] == EXPECTED_PRESENTATION_SLIDE_COUNT
    assert presentation["detected_speaker_note_slide_count"] == EXPECTED_PRESENTATION_SLIDE_COUNT
    assert presentation["slides"] == [
        {"slide_index": index, "visual_inspection_status": AGENT_INSPECTION_STATUS}
        for index in range(1, EXPECTED_PRESENTATION_SLIDE_COUNT + 1)
    ]
    assert presentation["speaker_note_slides"] == [
        _speaker_note_binding(presentation_path, index)
        for index in range(1, EXPECTED_PRESENTATION_SLIDE_COUNT + 1)
    ]

    screenshots = artifacts["application_screenshots"]
    assert screenshots["detected_screenshot_count"] == EXPECTED_APPLICATION_SCREENSHOT_COUNT
    assert screenshots["screenshots"] == [
        {
            "screenshot_index": index,
            "path": relative_path.as_posix(),
            "sha256": hashlib.sha256((tmp_path / relative_path).read_bytes()).hexdigest(),
            "size_bytes": (tmp_path / relative_path).stat().st_size,
            "visual_inspection_status": AGENT_INSPECTION_STATUS,
        }
        for index, relative_path in enumerate(APPLICATION_SCREENSHOT_PATHS, start=1)
    ]


def test_refresh_binds_speaker_notes_by_slide_relationship_order(tmp_path: Path) -> None:
    _write_pdf(tmp_path)
    presentation_path = _write_presentation(tmp_path)
    _replace_presentation_member(
        presentation_path,
        "ppt/slides/_rels/slide1.xml.rels",
        _relationship_xml(2).encode(),
    )
    _replace_presentation_member(
        presentation_path,
        "ppt/slides/_rels/slide2.xml.rels",
        _relationship_xml(1).encode(),
    )
    _replace_presentation_member(
        presentation_path,
        "ppt/notesSlides/_rels/notesSlide1.xml.rels",
        _note_relationships_xml(1, slide_index=2).encode(),
    )
    _replace_presentation_member(
        presentation_path,
        "ppt/notesSlides/_rels/notesSlide2.xml.rels",
        _note_relationships_xml(2, slide_index=1).encode(),
    )

    refreshed = refresh_release_visual_review(tmp_path, inspection_confirmed=True)
    speaker_notes = refreshed["artifacts"]["presentation"]["speaker_note_slides"]

    assert speaker_notes == [
        _speaker_note_binding(
            presentation_path,
            slide_index,
            note_index=2 if slide_index == 1 else 1 if slide_index == 2 else slide_index,
        )
        for slide_index in range(1, EXPECTED_PRESENTATION_SLIDE_COUNT + 1)
    ]
    assert verify_release_visual_review(tmp_path) == refreshed


@pytest.mark.parametrize("page_count", [EXPECTED_PDF_PAGE_COUNT - 1, EXPECTED_PDF_PAGE_COUNT + 1])
def test_refresh_rejects_wrong_pdf_page_count(tmp_path: Path, page_count: int) -> None:
    _write_pdf(tmp_path, page_count=page_count)
    _write_presentation(tmp_path)

    with pytest.raises(
        ReleaseVisualReviewError, match=f"exactly {EXPECTED_PDF_PAGE_COUNT} PDF pages"
    ):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    ("pdf_options", "message"),
    [
        ({"attachment": True}, "attachment|embedded file"),
        ({"javascript": True}, "active content|JavaScript"),
        ({"uri_action": True}, "active content action /URI"),
    ],
)
def test_refresh_rejects_pdf_attachments_and_active_content(
    tmp_path: Path,
    pdf_options: dict[str, bool],
    message: str,
) -> None:
    _write_pdf(tmp_path, **pdf_options)
    _write_presentation(tmp_path)

    with pytest.raises(ReleaseVisualReviewError, match=message):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    ("pdf_options", "message"),
    [
        ({"orphan_javascript": True}, "active content|JavaScript"),
        ({"orphan_filespec": True}, "attachment|embedded file"),
        ({"orphan_postscript": True}, "PostScript|/PS|active content"),
        ({"external_stream": True}, "external file|external stream"),
    ],
)
def test_refresh_rejects_orphan_or_external_pdf_objects(
    tmp_path: Path,
    pdf_options: dict[str, bool],
    message: str,
) -> None:
    _write_pdf(tmp_path, **pdf_options)
    _write_presentation(tmp_path)

    with pytest.raises(ReleaseVisualReviewError, match=message):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    "suffix",
    [b"\nMZ-HIDDEN-PAYLOAD", b"\nMZ-HIDDEN-PAYLOAD\n%%EOF\n"],
)
def test_refresh_rejects_pdf_bytes_after_terminal_eof(
    tmp_path: Path,
    suffix: bytes,
) -> None:
    pdf = _write_pdf(tmp_path)
    _write_presentation(tmp_path)
    pdf.write_bytes(pdf.read_bytes() + suffix)

    with pytest.raises(ReleaseVisualReviewError, match="PDF.*container|terminal EOF|trailing"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_pdf_bytes_before_header(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path)
    _write_presentation(tmp_path)
    pdf.write_bytes(b"MZ-HIDDEN-PAYLOAD\n" + pdf.read_bytes())

    with pytest.raises(ReleaseVisualReviewError, match="PDF.*header|byte zero"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_unbound_pdf_bytes_before_xref(tmp_path: Path) -> None:
    pdf = _write_pdf(tmp_path)
    _write_presentation(tmp_path)
    _insert_unbound_pdf_bytes_before_xref(pdf, b"MZ-HIDDEN-PAYLOAD\n")

    with pytest.raises(ReleaseVisualReviewError, match="unbound bytes"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize("slide_count", [12, 14])
def test_refresh_rejects_wrong_presentation_slide_count(tmp_path: Path, slide_count: int) -> None:
    _write_pdf(tmp_path)
    _write_presentation(tmp_path, slide_count=slide_count)

    with pytest.raises(ReleaseVisualReviewError, match="exactly 13 presentation slides"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    "member_name",
    [
        "/absolute.xml",
        "C:/absolute.xml",
        "../outside.xml",
        "ppt/./media/image99.png",
        "ppt/media/../image99.png",
        r"ppt\media\image99.png",
    ],
)
def test_refresh_rejects_unsafe_presentation_zip_member_paths(
    tmp_path: Path,
    member_name: str,
) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _append_presentation_member(presentation, member_name)

    with pytest.raises(ReleaseVisualReviewError, match="ZIP member path"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_presentation_symbolic_link_member(tmp_path: Path) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _append_presentation_member(
        presentation,
        "ppt/media/image99.png",
        b"image1.png",
        symbolic_link=True,
    )

    with pytest.raises(ReleaseVisualReviewError, match="symbolic link"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize("mutation", ["prefix", "suffix", "comment"])
def test_refresh_rejects_noncanonical_presentation_container_bytes(
    tmp_path: Path,
    mutation: str,
) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    payload = b"MZ-HIDDEN-PAYLOAD"
    if mutation == "prefix":
        presentation.write_bytes(payload + presentation.read_bytes())
    elif mutation == "suffix":
        presentation.write_bytes(presentation.read_bytes() + payload)
    else:
        with zipfile.ZipFile(presentation, "a") as archive:
            archive.comment = payload

    with pytest.raises(
        ReleaseVisualReviewError,
        match="presentation.*container|ZIP.*comment|prefix|trailing",
    ):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize("compression", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_refresh_rejects_non_opc_zip_compression(
    tmp_path: Path,
    compression: int,
) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _rewrite_presentation_compression(presentation, compression)

    with pytest.raises(ReleaseVisualReviewError, match="unsupported ZIP metadata"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_unused_bytes_inside_deflate_member(tmp_path: Path) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _insert_unused_deflate_member_payload(presentation, b"MZ-HIDDEN-IN-DEFLATE-MEMBER")

    with pytest.raises(ReleaseVisualReviewError, match="exact compression stream"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_mismatched_local_zip_member_name(tmp_path: Path) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _add_referenced_slide_image(presentation, VALID_PNG)
    _replace_local_header_member_name(
        presentation,
        "ppt/media/image1.png",
        "../" + "x" * 17,
    )

    with pytest.raises(
        ReleaseVisualReviewError,
        match="local.*filename|ZIP.*member|presentation.*container",
    ):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_nonzero_office_zip_growth_hint_payload(tmp_path: Path) -> None:
    _write_pdf(tmp_path)
    source = REPOSITORY_ROOT / PRESENTATION_RELATIVE_PATH
    presentation = tmp_path / PRESENTATION_RELATIVE_PATH
    presentation.parent.mkdir(parents=True, exist_ok=True)
    presentation.write_bytes(source.read_bytes())
    _tamper_local_header_padding(presentation, "[Content_Types].xml")

    with pytest.raises(ReleaseVisualReviewError, match="local ZIP filename/header"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_external_presentation_relationship(tmp_path: Path) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _add_external_slide_relationship(presentation)

    with pytest.raises(ReleaseVisualReviewError, match="external relationship"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    ("member_name", "message"),
    [
        ("ppt/embeddings/oleObject1.bin", "embedded/OLE|active payload"),
        ("customXml/item1.xml", "outside the generated presentation contract"),
    ],
)
def test_refresh_rejects_embedded_or_unexpected_presentation_payload(
    tmp_path: Path,
    member_name: str,
    message: str,
) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _append_presentation_member(presentation, member_name)

    with pytest.raises(ReleaseVisualReviewError, match=message):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_unreferenced_presentation_payload(tmp_path: Path) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _append_presentation_member(presentation, "ppt/media/image99.png", b"not really a PNG")

    with pytest.raises(ReleaseVisualReviewError, match="unreferenced"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_executable_bytes_disguised_as_referenced_image(
    tmp_path: Path,
) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _add_referenced_slide_image(presentation, b"MZ-HIDDEN-PAYLOAD")

    with pytest.raises(ReleaseVisualReviewError, match="PNG|image encoding|media"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (b"MZ-HIDDEN-PAYLOAD", "invalid XML|XML payload"),
        (
            f'<p:notes xmlns:p="{PRESENTATION_NAMESPACE}"/>'.encode(),
            "unexpected root",
        ),
        (
            (
                f'<p:sld xmlns:p="{PRESENTATION_NAMESPACE}" '
                f'xmlns:a="{DRAWING_NAMESPACE}">'
                '<p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="1" name="payload">'
                '<a:hlinkClick action="ppaction://program?file:///C:/Windows/System32/'
                'calc.exe"/>'
                "</p:cNvPr></p:nvSpPr></p:sp></p:spTree></p:cSld></p:sld>"
            ).encode(),
            "active.*XML|action|hlinkClick",
        ),
        (
            (
                f'<p:sld xmlns:p="{PRESENTATION_NAMESPACE}">'
                '<p:timing><p:tnLst><p:cmd type="evt" cmd="onstopaudio"/>'
                "</p:tnLst></p:timing></p:sld>"
            ).encode(),
            "active.*XML|cmd",
        ),
    ],
)
def test_refresh_rejects_malformed_or_active_presentation_xml(
    tmp_path: Path,
    replacement: bytes,
    message: str,
) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _replace_presentation_member(
        presentation,
        "ppt/slides/slide1.xml",
        replacement,
    )

    with pytest.raises(ReleaseVisualReviewError, match=message):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


def test_refresh_rejects_wrong_note_to_slide_back_reference(tmp_path: Path) -> None:
    _write_pdf(tmp_path)
    presentation = _write_presentation(tmp_path)
    _replace_presentation_member(
        presentation,
        "ppt/notesSlides/_rels/notesSlide1.xml.rels",
        _note_relationships_xml(1, slide_index=2).encode(),
    )

    with pytest.raises(ReleaseVisualReviewError, match="back-reference|slide 1"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    ("presentation_options", "message"),
    [
        ({"missing_note_relationship": 7}, "notes relationship"),
        ({"empty_note": 7}, "nonempty speaker-note text"),
    ],
)
def test_refresh_rejects_invalid_speaker_notes(
    tmp_path: Path,
    presentation_options: dict[str, int],
    message: str,
) -> None:
    _write_pdf(tmp_path)
    _write_presentation(tmp_path, **presentation_options)

    with pytest.raises(ReleaseVisualReviewError, match=message):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    "presentation_options",
    [
        {"referenced_slide_count": 12},
        {"omit_presentation_graph": True},
    ],
)
def test_refresh_rejects_missing_or_incomplete_presentation_graph(
    tmp_path: Path,
    presentation_options: dict[str, int | bool],
) -> None:
    _write_pdf(tmp_path)
    _write_presentation(tmp_path, **presentation_options)

    with pytest.raises(ReleaseVisualReviewError, match="presentation graph"):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    ("presentation_options", "message"),
    [
        ({"invalid_note_relationship": 7}, "supported notes relationship"),
        ({"invalid_note_namespace": 7}, "PresentationML notes root"),
        ({"slide_number_only_note": 7}, "notes body"),
    ],
)
def test_refresh_rejects_loose_or_placeholder_only_note_content(
    tmp_path: Path,
    presentation_options: dict[str, int],
    message: str,
) -> None:
    _write_pdf(tmp_path)
    _write_presentation(tmp_path, **presentation_options)

    with pytest.raises(ReleaseVisualReviewError, match=message):
        refresh_release_visual_review(tmp_path, inspection_confirmed=True)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda record: record.update({"unknown": True}),
        lambda record: record["inspection_attestation"].update({"unknown": True}),
        lambda record: record["artifacts"]["pdf"]["pages"][0].update({"unknown": True}),
        lambda record: record["artifacts"]["presentation"]["slides"][0].update({"unknown": True}),
        lambda record: record["artifacts"]["presentation"]["speaker_note_slides"][0].update(
            {"unknown": True}
        ),
        lambda record: record["artifacts"]["application_screenshots"]["screenshots"][0].update(
            {"unknown": True}
        ),
    ],
)
def test_verifier_rejects_unknown_fields(tmp_path: Path, mutator) -> None:
    _write_release_artifacts(tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)
    record = _load_record(tmp_path)
    mutator(record)
    _write_canonical_record(tmp_path, record)

    with pytest.raises(ReleaseVisualReviewError, match="unknown fields"):
        verify_release_visual_review(tmp_path)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda record: record["artifacts"]["pdf"]["pages"][0].update({"page_index": 2}),
        lambda record: record["artifacts"]["pdf"]["pages"][0].update(
            {"visual_inspection_status": "pending"}
        ),
        lambda record: record["artifacts"]["presentation"]["slides"][0].update({"slide_index": 2}),
        lambda record: record["artifacts"]["presentation"]["speaker_note_slides"][0].update(
            {"structural_inspection_status": "unchecked"}
        ),
        lambda record: record["artifacts"]["presentation"]["speaker_note_slides"][0].update(
            {"content_inspection_status": "unchecked"}
        ),
        lambda record: record["artifacts"]["application_screenshots"]["screenshots"][0].update(
            {"screenshot_index": 2}
        ),
        lambda record: record["artifacts"]["application_screenshots"]["screenshots"][0].update(
            {"visual_inspection_status": "unchecked"}
        ),
    ],
)
def test_verifier_rejects_tampered_indices_or_statuses(tmp_path: Path, mutator) -> None:
    _write_release_artifacts(tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)
    record = _load_record(tmp_path)
    mutator(record)
    _write_canonical_record(tmp_path, record)

    with pytest.raises(ReleaseVisualReviewError, match="inspection"):
        verify_release_visual_review(tmp_path)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda record: record["artifacts"]["presentation"]["speaker_note_slides"][0].update(
                {"path": "ppt/notesSlides/notesSlide2.xml"}
            ),
            "speaker-note paths",
        ),
        (
            lambda record: record["artifacts"]["application_screenshots"]["screenshots"][0].update(
                {"path": APPLICATION_SCREENSHOT_PATHS[1].as_posix()}
            ),
            "application screenshot path 1",
        ),
        (
            lambda record: record["artifacts"]["presentation"]["speaker_note_slides"].pop(),
            "exactly 13 entries",
        ),
        (
            lambda record: record["artifacts"]["application_screenshots"]["screenshots"].pop(),
            "exactly 3 entries",
        ),
    ],
)
def test_verifier_rejects_incomplete_or_misbound_review_rows(
    tmp_path: Path,
    mutator,
    message: str,
) -> None:
    _write_release_artifacts(tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)
    record = _load_record(tmp_path)
    mutator(record)
    _write_canonical_record(tmp_path, record)

    with pytest.raises(ReleaseVisualReviewError, match=message):
        verify_release_visual_review(tmp_path)


def test_verifier_rejects_artifact_hash_tampering(tmp_path: Path) -> None:
    _, presentation_path = _write_release_artifacts(tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)
    with presentation_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(ReleaseVisualReviewError, match="presentation binding"):
        verify_release_visual_review(tmp_path)


def test_verifier_rejects_speaker_note_binding_tampering(tmp_path: Path) -> None:
    _write_release_artifacts(tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)
    record = _load_record(tmp_path)
    record["artifacts"]["presentation"]["speaker_note_slides"][0]["sha256"] = "0" * 64
    _write_canonical_record(tmp_path, record)

    with pytest.raises(ReleaseVisualReviewError, match="speaker-note binding"):
        verify_release_visual_review(tmp_path)


def test_verifier_rejects_application_screenshot_tampering(tmp_path: Path) -> None:
    _write_release_artifacts(tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)
    screenshot = tmp_path / APPLICATION_SCREENSHOT_PATHS[2]
    screenshot.write_bytes(VALID_PNG + b"tampered")

    with pytest.raises(ReleaseVisualReviewError, match="application screenshot binding"):
        verify_release_visual_review(tmp_path)


def test_verifier_rejects_noncanonical_and_duplicate_json(tmp_path: Path) -> None:
    _write_release_artifacts(tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)
    record_path = tmp_path / RECORD_RELATIVE_PATH
    record = _load_record(tmp_path)

    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ReleaseVisualReviewError, match="canonical JSON"):
        verify_release_visual_review(tmp_path)

    canonical = canonical_release_visual_review_bytes(record)
    duplicate = canonical.replace(
        b'{\n  "artifacts"',
        b'{\n  "schema_version": 2,\n  "artifacts"',
        1,
    )
    record_path.write_bytes(duplicate)
    with pytest.raises(ReleaseVisualReviewError, match="duplicate JSON field"):
        verify_release_visual_review(tmp_path)


def test_cli_refresh_and_verify_report_limited_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_release_artifacts(tmp_path)

    assert (
        main(
            [
                "--refresh",
                "--repository-root",
                str(tmp_path),
                "--confirm-agent-content-and-visual-inspection-completed",
                "--reviewer",
                "Codex root agent",
            ]
        )
        == 0
    )
    refresh_output = capsys.readouterr().out
    assert "recorded the confirmed AI-agent content and visual inspection" in refresh_output
    assert "not a human visual-inspection claim" in refresh_output

    assert main(["--verify", "--repository-root", str(tmp_path)]) == 0
    verify_output = capsys.readouterr().out
    assert "hashes, paths, structure, counts, and recorded statuses verified" in verify_output
    assert "does not independently repeat content or visual inspection" in verify_output
    assert "Historical record/artifact verification" in verify_output
    assert (
        "current-source release requires scripts/build-presentation.ps1 -VerifyOnly"
        in verify_output
    )


def test_cli_rejects_refresh_only_reviewer_option_in_verify_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_release_artifacts(tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)

    assert (
        main(
            [
                "--verify",
                "--repository-root",
                str(tmp_path),
                "--reviewer",
                "Misleading ignored reviewer",
            ]
        )
        == 1
    )
    assert "--reviewer is valid only with --refresh" in capsys.readouterr().err


def test_cli_rejects_review_confirmation_in_verify_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_release_artifacts(tmp_path)
    refresh_release_visual_review(tmp_path, inspection_confirmed=True)

    assert (
        main(
            [
                "--verify",
                "--repository-root",
                str(tmp_path),
                "--confirm-agent-content-and-visual-inspection-completed",
            ]
        )
        == 1
    )
    assert "confirmation flag is valid only with --refresh" in capsys.readouterr().err


def test_committed_release_visual_review_verifies_current_final_artifacts() -> None:
    verified = verify_release_visual_review(REPOSITORY_ROOT)

    assert verified["artifacts"]["pdf"]["detected_page_count"] == EXPECTED_PDF_PAGE_COUNT
    assert verified["artifacts"]["presentation"]["detected_slide_count"] == 13
    assert verified["artifacts"]["presentation"]["detected_speaker_note_slide_count"] == 13
    assert len(verified["artifacts"]["presentation"]["speaker_note_slides"]) == 13
    assert verified["artifacts"]["application_screenshots"]["detected_screenshot_count"] == 3
    assert len(verified["artifacts"]["application_screenshots"]["screenshots"]) == 3


def test_current_real_release_artifacts_pass_direct_structural_inspection() -> None:
    pdf = inspect_pdf_artifact(
        REPOSITORY_ROOT / PDF_RELATIVE_PATH,
        PDF_RELATIVE_PATH.as_posix(),
        expected_page_count=EXPECTED_PDF_PAGE_COUNT,
    )
    presentation = inspect_presentation_artifact(
        REPOSITORY_ROOT / PRESENTATION_RELATIVE_PATH,
        PRESENTATION_RELATIVE_PATH.as_posix(),
        expected_slide_count=EXPECTED_PRESENTATION_SLIDE_COUNT,
        expected_note_count=EXPECTED_PRESENTATION_SLIDE_COUNT,
    )

    assert pdf["detected_page_count"] == EXPECTED_PDF_PAGE_COUNT
    assert presentation["detected_slide_count"] == EXPECTED_PRESENTATION_SLIDE_COUNT
    assert presentation["detected_speaker_note_slide_count"] == EXPECTED_PRESENTATION_SLIDE_COUNT


def test_direct_inspectors_use_one_immutable_artifact_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf, presentation = _write_release_artifacts(tmp_path)
    inspected_paths = {pdf, presentation}
    open_counts = {path: 0 for path in inspected_paths}
    original_open = Path.open

    def counting_open(path: Path, *args, **kwargs):
        if path in inspected_paths:
            open_counts[path] += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    inspect_pdf_artifact(
        pdf,
        PDF_RELATIVE_PATH.as_posix(),
        expected_page_count=EXPECTED_PDF_PAGE_COUNT,
    )
    inspect_presentation_artifact(
        presentation,
        PRESENTATION_RELATIVE_PATH.as_posix(),
        expected_slide_count=EXPECTED_PRESENTATION_SLIDE_COUNT,
        expected_note_count=EXPECTED_PRESENTATION_SLIDE_COUNT,
    )

    assert open_counts == {pdf: 1, presentation: 1}
