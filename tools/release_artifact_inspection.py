"""Fail-closed structural inspection for final PDF and PPTX artifacts."""

from __future__ import annotations

import hashlib
import io
import posixpath
import re
import stat
import struct
import zipfile
import zlib
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree

from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject, StreamObject

_SLIDE_NAME = re.compile(r"^ppt/slides/slide([1-9][0-9]*)\.xml$")
_NOTE_NAME = re.compile(r"^ppt/notesSlides/notesSlide([1-9][0-9]*)\.xml$")
_IMAGE_NAME = re.compile(r"^ppt/media/image[1-9][0-9]*\.png$")
_NOTES_MASTER_NAME = re.compile(r"^ppt/notesMasters/notesMaster[1-9][0-9]*\.xml$")
_SLIDE_LAYOUT_NAME = re.compile(r"^ppt/slideLayouts/slideLayout[1-9][0-9]*\.xml$")
_SLIDE_MASTER_NAME = re.compile(r"^ppt/slideMasters/slideMaster[1-9][0-9]*\.xml$")
_THEME_NAME = re.compile(r"^ppt/theme/theme[1-9][0-9]*\.xml$")
_CONTENT_TYPES_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/content-types"
_PACKAGE_RELATIONSHIPS_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
_PRESENTATION_NAMESPACES = frozenset(
    {
        "http://schemas.openxmlformats.org/presentationml/2006/main",
        "http://purl.oclc.org/ooxml/presentationml/main",
    }
)
_DRAWING_NAMESPACES = frozenset(
    {
        "http://schemas.openxmlformats.org/drawingml/2006/main",
        "http://purl.oclc.org/ooxml/drawingml/main",
    }
)
_CORE_PROPERTIES_NAMESPACE = (
    "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
)
_EXTENDED_PROPERTIES_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
)
_OFFICE_RELATIONSHIPS_NAMESPACES = frozenset(
    {
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "http://purl.oclc.org/ooxml/officeDocument/relationships",
    }
)
_OFFICE_DOCUMENT_RELATIONSHIP_TYPES = frozenset(
    f"{namespace}/officeDocument" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
)
_SLIDE_RELATIONSHIP_TYPES = frozenset(
    f"{namespace}/slide" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
)
_NOTES_RELATIONSHIP_TYPES = frozenset(
    f"{namespace}/notesSlide" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
)
_RELATIONSHIP_TYPES_BY_FAMILY = {
    "core_properties": frozenset({f"{_PACKAGE_RELATIONSHIPS_NAMESPACE}/metadata/core-properties"}),
    "extended_properties": frozenset(
        f"{namespace}/extended-properties" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
    ),
    "image": frozenset(f"{namespace}/image" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES),
    "notes_master": frozenset(
        f"{namespace}/notesMaster" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
    ),
    "notes_slide": _NOTES_RELATIONSHIP_TYPES,
    "office_document": _OFFICE_DOCUMENT_RELATIONSHIP_TYPES,
    "presentation_properties": frozenset(
        f"{namespace}/presProps" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
    ),
    "slide": _SLIDE_RELATIONSHIP_TYPES,
    "slide_layout": frozenset(
        f"{namespace}/slideLayout" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
    ),
    "slide_master": frozenset(
        f"{namespace}/slideMaster" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
    ),
    "table_styles": frozenset(
        f"{namespace}/tableStyles" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
    ),
    "theme": frozenset(f"{namespace}/theme" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES),
    "thumbnail": frozenset({f"{_PACKAGE_RELATIONSHIPS_NAMESPACE}/metadata/thumbnail"}),
    "view_properties": frozenset(
        f"{namespace}/viewProps" for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
    ),
}
_RELATIONSHIP_FAMILY_BY_TYPE = {
    relationship_type: family
    for family, relationship_types in _RELATIONSHIP_TYPES_BY_FAMILY.items()
    for relationship_type in relationship_types
}
_RELATIONSHIP_TARGET_PATTERNS = {
    "core_properties": re.compile(r"^docProps/core\.xml$"),
    "extended_properties": re.compile(r"^docProps/app\.xml$"),
    "image": _IMAGE_NAME,
    "notes_master": _NOTES_MASTER_NAME,
    "notes_slide": _NOTE_NAME,
    "office_document": re.compile(r"^ppt/presentation\.xml$"),
    "presentation_properties": re.compile(r"^ppt/presProps\.xml$"),
    "slide": _SLIDE_NAME,
    "slide_layout": _SLIDE_LAYOUT_NAME,
    "slide_master": _SLIDE_MASTER_NAME,
    "table_styles": re.compile(r"^ppt/tableStyles\.xml$"),
    "theme": _THEME_NAME,
    "thumbnail": re.compile(r"^docProps/thumbnail\.jpeg$"),
    "view_properties": re.compile(r"^ppt/viewProps\.xml$"),
}
_PRESENTATION_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
)
_SLIDE_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
_NOTES_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.notesSlide+xml"
_RELATIONSHIPS_CONTENT_TYPE = "application/vnd.openxmlformats-package.relationships+xml"
_PRESENTATION_EXACT_PART_CONTENT_TYPES = {
    "docProps/app.xml": "application/vnd.openxmlformats-officedocument.extended-properties+xml",
    "docProps/core.xml": "application/vnd.openxmlformats-package.core-properties+xml",
    "docProps/thumbnail.jpeg": "image/jpeg",
    "ppt/presentation.xml": _PRESENTATION_CONTENT_TYPE,
    "ppt/presProps.xml": (
        "application/vnd.openxmlformats-officedocument.presentationml.presProps+xml"
    ),
    "ppt/tableStyles.xml": (
        "application/vnd.openxmlformats-officedocument.presentationml.tableStyles+xml"
    ),
    "ppt/viewProps.xml": (
        "application/vnd.openxmlformats-officedocument.presentationml.viewProps+xml"
    ),
}
_PRESENTATION_NUMBERED_PART_CONTENT_TYPES = (
    (
        _IMAGE_NAME,
        "image/png",
    ),
    (
        _NOTES_MASTER_NAME,
        "application/vnd.openxmlformats-officedocument.presentationml.notesMaster+xml",
    ),
    (_NOTE_NAME, _NOTES_CONTENT_TYPE),
    (
        _SLIDE_LAYOUT_NAME,
        "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml",
    ),
    (
        _SLIDE_MASTER_NAME,
        "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml",
    ),
    (_SLIDE_NAME, _SLIDE_CONTENT_TYPE),
    (
        _THEME_NAME,
        "application/vnd.openxmlformats-officedocument.theme+xml",
    ),
)
_PRESENTATION_DEFAULT_CONTENT_TYPES = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "rels": _RELATIONSHIPS_CONTENT_TYPE,
    "xml": "application/xml",
}
_ACTIVE_PRESENTATION_PATH_MARKERS = (
    "/activex/",
    "/controls/",
    "/embeddings/",
    "/macros/",
    "/oleobjects/",
    "/vbaproject",
)
_PDF_ATTACHMENT_KEYS = frozenset({"/AF", "/EF", "/EmbeddedFiles"})
_PDF_EXTERNAL_STREAM_KEYS = frozenset({"/F", "/FDecodeParms", "/FFilter"})
_PDF_ACTIVE_STREAM_SUBTYPES = frozenset({"/PS"})
_PDF_ACTIVE_CONTENT_KEYS = frozenset(
    {
        "/AA",
        "/AcroForm",
        "/JavaScript",
        "/JS",
        "/OpenAction",
        "/RichMediaContent",
        "/RichMediaExecute",
        "/RichMediaSettings",
        "/XFA",
    }
)
_PDF_ACTIVE_ACTION_TYPES = frozenset(
    {
        "/GoTo3DView",
        "/GoToE",
        "/GoToR",
        "/Hide",
        "/ImportData",
        "/JavaScript",
        "/Launch",
        "/Movie",
        "/Named",
        "/Rendition",
        "/ResetForm",
        "/SetOCGState",
        "/Sound",
        "/SubmitForm",
        "/Thread",
        "/Trans",
        "/URI",
    }
)
_PDF_ACTIVE_ANNOTATION_SUBTYPES = frozenset(
    {
        "/3D",
        "/FileAttachment",
        "/Movie",
        "/RichMedia",
        "/Screen",
        "/Sound",
    }
)
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:")
_PDF_TERMINAL_EOF = re.compile(rb"%%EOF[\t ]*(?:\r\n|\n|\r)?\Z")
_PDF_CLASSIC_HEADER = re.compile(rb"%PDF-1\.[0-9]\r?\n%[^\r\n]{4,96}\r?\n")
_PDF_STARTXREF = re.compile(rb"startxref[\t \r\n]+([0-9]+)[\t \r\n]+%%EOF[\t ]*(?:\r\n|\n|\r)?\Z")
_ZIP_LOCAL_FILE_HEADER = struct.Struct("<4s5H3L2H")
_ZIP_END_CENTRAL_DIRECTORY = struct.Struct("<4s4H2LH")
_MAX_PRESENTATION_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_PRESENTATION_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_PRESENTATION_XML_BYTES = 8 * 1024 * 1024
_MAX_PRESENTATION_IMAGE_PIXELS = 4_194_304
_MAX_RELEASE_ARTIFACT_BYTES = 64 * 1024 * 1024
_ACTIVE_PRESENTATION_XML_ELEMENTS = frozenset(
    {
        "audio",
        "audiofile",
        "cmd",
        "contentpart",
        "control",
        "custdata",
        "custdatalst",
        "hlinkclick",
        "hlinkhover",
        "media",
        "oleobj",
        "quicktimefile",
        "snd",
        "video",
        "videofile",
        "wavaudiofile",
        "webvideo",
        "webvideopr",
    }
)
_ACTIVE_PRESENTATION_ATTRIBUTE_SCHEMES = (
    "data:",
    "file:",
    "ftp:",
    "http:",
    "https:",
    "javascript:",
    "mailto:",
    "ms-officecmd:",
    "ppaction:",
    "vbscript:",
)


class ReleaseArtifactInspectionError(ValueError):
    """Raised when a final release artifact fails structural inspection."""


def _artifact_snapshot(path: Path, relative_path: str) -> tuple[dict[str, object], bytes]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_RELEASE_ARTIFACT_BYTES + 1)
    except OSError as error:
        raise ReleaseArtifactInspectionError(f"cannot read artifact {path}: {error}") from error
    if not raw:
        raise ReleaseArtifactInspectionError(f"artifact must be a nonempty file: {path}")
    if len(raw) > _MAX_RELEASE_ARTIFACT_BYTES:
        raise ReleaseArtifactInspectionError(
            f"artifact exceeds the release inspection size limit: {path}"
        )
    return (
        {
            "path": relative_path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
        raw,
    )


def _require_canonical_pdf_container(raw: bytes) -> None:
    if not raw.startswith(b"%PDF-"):
        raise ReleaseArtifactInspectionError(
            "final PDF container must begin with its PDF header at byte zero"
        )
    if (
        raw.count(b"%%EOF") != 1
        or raw.count(b"startxref") != 1
        or _PDF_TERMINAL_EOF.search(raw) is None
    ):
        raise ReleaseArtifactInspectionError(
            "final PDF container must have exactly one terminal xref/EOF marker and no "
            "trailing bytes"
        )


def _require_classic_pdf_byte_coverage(raw: bytes, reader: PdfReader) -> None:
    startxref_match = _PDF_STARTXREF.search(raw)
    if startxref_match is None:
        raise ReleaseArtifactInspectionError(
            "final PDF container must use one canonical terminal startxref record"
        )
    xref_offset = int(startxref_match.group(1))
    if reader.xref_objStm or set(reader.xref) - {0, 65535}:
        raise ReleaseArtifactInspectionError(
            "final PDF container must use the generated classic xref layout"
        )
    object_offsets = reader.xref.get(0, {})
    try:
        declared_size = int(reader.trailer["/Size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseArtifactInspectionError(
            "final PDF trailer must declare its complete object count"
        ) from error
    object_ids = sorted(object_offsets)
    if object_ids != list(range(1, declared_size)):
        raise ReleaseArtifactInspectionError(
            "final PDF xref must bind every generated indirect object exactly once"
        )
    physical_offsets = [object_offsets[object_id] for object_id in object_ids]
    if physical_offsets != sorted(set(physical_offsets)) or not physical_offsets:
        raise ReleaseArtifactInspectionError(
            "final PDF indirect objects must use a contiguous canonical order"
        )
    header = raw[: physical_offsets[0]]
    if _PDF_CLASSIC_HEADER.fullmatch(header) is None:
        raise ReleaseArtifactInspectionError(
            "final PDF header contains bytes outside the generated container contract"
        )
    boundaries = [*physical_offsets[1:], xref_offset]
    for object_id, object_offset, next_offset in zip(
        object_ids,
        physical_offsets,
        boundaries,
        strict=True,
    ):
        if not (object_offset < next_offset <= len(raw)):
            raise ReleaseArtifactInspectionError("final PDF indirect-object boundaries are invalid")
        object_bytes = raw[object_offset:next_offset]
        object_header = re.compile(
            rb"^" + str(object_id).encode("ascii") + rb" 0 obj(?:\r\n|\n|\r)"
        )
        if (
            object_header.search(object_bytes) is None
            or re.search(
                rb"endobj[\t ]*(?:\r\n|\n|\r)\Z",
                object_bytes,
            )
            is None
        ):
            raise ReleaseArtifactInspectionError(
                "final PDF has unbound bytes between indirect objects or before its xref: "
                f"object {object_id}"
            )

    xref_lines = [
        b"xref\n",
        f"0 {declared_size}\n".encode("ascii"),
        b"0000000000 65535 f \n",
    ]
    xref_lines.extend(
        f"{object_offsets[object_id]:010d} 00000 n \n".encode("ascii") for object_id in object_ids
    )
    expected_xref = b"".join(xref_lines)
    if raw[xref_offset : xref_offset + len(expected_xref)] != expected_xref:
        raise ReleaseArtifactInspectionError(
            "final PDF xref table is outside the generated canonical contract"
        )
    trailer_bytes = raw[xref_offset + len(expected_xref) : startxref_match.start()]
    trailer_keys = {str(key) for key in reader.trailer}
    if (
        not {"/Root", "/Size"}.issubset(trailer_keys)
        or not trailer_keys.issubset({"/ID", "/Info", "/Root", "/Size"})
        or not trailer_bytes.startswith(b"trailer\n<<")
        or re.search(rb">>[\t ]*(?:\r\n|\n|\r)\Z", trailer_bytes) is None
        or any(byte not in {9, 10, 13} and not 32 <= byte <= 126 for byte in trailer_bytes)
        or any(
            token in trailer_bytes
            for token in (b" obj", b"stream", b"endobj", b"startxref", b"%%EOF")
        )
    ):
        raise ReleaseArtifactInspectionError(
            "final PDF trailer contains bytes outside the generated container contract"
        )
    comment_lines = [
        line.strip() for line in trailer_bytes.splitlines() if line.lstrip().startswith(b"%")
    ]
    if comment_lines not in (
        [],
        [b"% ReportLab generated PDF document -- digest (opensource)"],
    ):
        raise ReleaseArtifactInspectionError(
            "final PDF trailer contains an unsupported comment payload"
        )


def _pdf_indirect_references(reader: PdfReader) -> list[IndirectObject]:
    references = {
        (object_id, generation)
        for generation, objects in reader.xref.items()
        for object_id in objects
        if object_id > 0
    }
    references.update((object_id, 0) for object_id in reader.xref_objStm if object_id > 0)
    return [
        IndirectObject(object_id, generation, reader)
        for object_id, generation in sorted(references)
    ]


def _require_passive_pdf(reader: PdfReader) -> None:
    if next(reader.attachment_list, None) is not None:
        raise ReleaseArtifactInspectionError("final PDF must not contain file attachments")

    pending: list[object] = [reader.trailer, *_pdf_indirect_references(reader)]
    visited_indirect: set[tuple[int, int]] = set()
    visited_direct: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, IndirectObject):
            reference = (value.idnum, value.generation)
            if reference in visited_indirect:
                continue
            visited_indirect.add(reference)
            value = value.get_object()
        if isinstance(value, DictionaryObject):
            object_identity = id(value)
            if object_identity in visited_direct:
                continue
            visited_direct.add(object_identity)
            keys = {str(key) for key in value}
            if isinstance(value, StreamObject):
                external_stream_keys = sorted(keys & _PDF_EXTERNAL_STREAM_KEYS)
                if external_stream_keys:
                    raise ReleaseArtifactInspectionError(
                        "final PDF must not contain an external file stream: "
                        f"{external_stream_keys}"
                    )
                stream_subtype = str(value.get("/Subtype", ""))
                if stream_subtype in _PDF_ACTIVE_STREAM_SUBTYPES:
                    raise ReleaseArtifactInspectionError(
                        "final PDF must not contain active PostScript stream content: "
                        f"{stream_subtype}"
                    )
            attachment_keys = sorted(keys & _PDF_ATTACHMENT_KEYS)
            if attachment_keys:
                raise ReleaseArtifactInspectionError(
                    f"final PDF must not contain embedded file attachments: {attachment_keys}"
                )
            active_keys = sorted(keys & _PDF_ACTIVE_CONTENT_KEYS)
            if active_keys:
                raise ReleaseArtifactInspectionError(
                    f"final PDF must not contain active content: {active_keys}"
                )
            object_type = str(value.get("/Type", ""))
            subtype = str(value.get("/Subtype", ""))
            if object_type == "/Filespec" or subtype == "/EmbeddedFile":
                raise ReleaseArtifactInspectionError(
                    "final PDF must not contain embedded file attachments"
                )
            if subtype in _PDF_ACTIVE_ANNOTATION_SUBTYPES:
                raise ReleaseArtifactInspectionError(
                    f"final PDF must not contain active {subtype} annotations"
                )
            action_type = str(value.get("/S", ""))
            if action_type in _PDF_ACTIVE_ACTION_TYPES:
                raise ReleaseArtifactInspectionError(
                    f"final PDF must not contain active content action {action_type}"
                )
            pending.extend(value.values())
        elif isinstance(value, ArrayObject):
            object_identity = id(value)
            if object_identity in visited_direct:
                continue
            visited_direct.add(object_identity)
            pending.extend(value)


def inspect_pdf_artifact(
    path: Path,
    relative_path: str,
    *,
    expected_page_count: int,
) -> dict[str, object]:
    """Return the bound identity and strict page count for one final PDF."""
    identity, raw = _artifact_snapshot(path, relative_path)
    try:
        _require_canonical_pdf_container(raw)
        reader = PdfReader(io.BytesIO(raw), strict=True)
        if reader.is_encrypted:
            raise ReleaseArtifactInspectionError("final PDF must not be encrypted")
        _require_classic_pdf_byte_coverage(raw, reader)
        page_count = len(reader.pages)
        _require_passive_pdf(reader)
    except ReleaseArtifactInspectionError:
        raise
    except (OSError, PdfReadError, ValueError) as error:
        raise ReleaseArtifactInspectionError(f"cannot parse final PDF: {error}") from error
    if page_count != expected_page_count:
        raise ReleaseArtifactInspectionError(
            f"release requires exactly {expected_page_count} PDF pages; detected {page_count}"
        )
    return {**identity, "detected_page_count": page_count}


def _indexed_parts(names: Sequence[str], pattern: re.Pattern[str]) -> list[int]:
    return sorted(
        int(match.group(1)) for name in names if (match := pattern.fullmatch(name)) is not None
    )


def _validated_presentation_member_names(archive: zipfile.ZipFile) -> list[str]:
    members = archive.infolist()
    member_names = [member.filename for member in members]
    member_counts: dict[str, int] = {}
    casefolded_names: dict[str, str] = {}
    for member in members:
        name = member.filename
        original_name = member.orig_filename
        member_counts[name] = member_counts.get(name, 0) + 1
        if (
            not name
            or original_name != name
            or "\x00" in name
            or "\\" in name
            or name.startswith("/")
            or _WINDOWS_DRIVE_PATH.match(name) is not None
            or member.is_dir()
            or any(segment in {"", ".", ".."} for segment in name.split("/"))
            or posixpath.normpath(name) != name
        ):
            raise ReleaseArtifactInspectionError(
                f"presentation ZIP member path is unsafe or noncanonical: {original_name!r}"
            )
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise ReleaseArtifactInspectionError(
                f"presentation ZIP member must not be a symbolic link: {name}"
            )
        if member.flag_bits & 0x1:
            raise ReleaseArtifactInspectionError(
                f"presentation ZIP member must not be encrypted: {name}"
            )
        folded = name.casefold()
        previous = casefolded_names.setdefault(folded, name)
        if previous != name:
            raise ReleaseArtifactInspectionError(
                "presentation contains case-insensitive ZIP member aliases: "
                f"{previous!r} and {name!r}"
            )
    duplicate_names = sorted(name for name, count in member_counts.items() if count > 1)
    if duplicate_names:
        raise ReleaseArtifactInspectionError(
            f"presentation contains duplicate ZIP members: {duplicate_names}"
        )
    return member_names


def _is_zero_filled_office_zip_padding(extra: bytes) -> bool:
    if not extra:
        return True
    if len(extra) < 8:
        return False
    outer_id, outer_size, inner_id, padding_size = struct.unpack_from("<4H", extra)
    return (
        outer_id == 0xA220
        and outer_size == len(extra) - 4
        and inner_id == 0xA028
        and padding_size == len(extra) - 8
        and not any(extra[8:])
    )


def _require_canonical_presentation_container(
    raw: bytes,
    archive: zipfile.ZipFile,
) -> None:
    members = archive.infolist()
    if archive.comment:
        raise ReleaseArtifactInspectionError("final presentation ZIP comment must be empty")
    if len(raw) < _ZIP_END_CENTRAL_DIRECTORY.size or not raw.startswith(b"PK\x03\x04"):
        raise ReleaseArtifactInspectionError(
            "final presentation container must begin with a local ZIP member at byte zero"
        )
    try:
        (
            signature,
            disk_number,
            central_directory_disk,
            entries_on_disk,
            entry_count,
            central_directory_size,
            central_directory_offset,
            comment_length,
        ) = _ZIP_END_CENTRAL_DIRECTORY.unpack_from(
            raw,
            len(raw) - _ZIP_END_CENTRAL_DIRECTORY.size,
        )
    except struct.error as error:
        raise ReleaseArtifactInspectionError(
            f"final presentation has an invalid ZIP end record: {error}"
        ) from error
    end_record_offset = len(raw) - _ZIP_END_CENTRAL_DIRECTORY.size
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or central_directory_disk != 0
        or entries_on_disk != len(members)
        or entry_count != len(members)
        or comment_length != 0
        or central_directory_offset != archive.start_dir
        or central_directory_offset + central_directory_size != end_record_offset
    ):
        raise ReleaseArtifactInspectionError(
            "final presentation binding container has a noncanonical prefix, trailing bytes, "
            "ZIP comment, or central-directory boundary"
        )

    total_uncompressed_bytes = sum(member.file_size for member in members)
    if total_uncompressed_bytes > _MAX_PRESENTATION_TOTAL_BYTES:
        raise ReleaseArtifactInspectionError(
            "final presentation uncompressed payload exceeds the generated deck limit"
        )
    cursor = 0
    for member in sorted(members, key=lambda candidate: candidate.header_offset):
        if (
            member.header_offset != cursor
            or member.file_size > _MAX_PRESENTATION_MEMBER_BYTES
            or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or member.extra
            or member.comment
        ):
            raise ReleaseArtifactInspectionError(
                "final presentation container has a hidden gap, oversized member, or "
                f"unsupported ZIP metadata at {member.filename}"
            )
        try:
            (
                local_signature,
                _extract_version,
                flags,
                compression,
                _modified_time,
                _modified_date,
                crc32,
                compressed_size,
                uncompressed_size,
                filename_length,
                extra_length,
            ) = _ZIP_LOCAL_FILE_HEADER.unpack_from(raw, cursor)
        except struct.error as error:
            raise ReleaseArtifactInspectionError(
                f"final presentation local ZIP header is truncated: {member.filename}"
            ) from error
        filename_start = cursor + _ZIP_LOCAL_FILE_HEADER.size
        filename_end = filename_start + filename_length
        payload_start = filename_end + extra_length
        payload_end = payload_start + compressed_size
        if payload_end > archive.start_dir:
            raise ReleaseArtifactInspectionError(
                f"final presentation ZIP member crosses its container boundary: {member.filename}"
            )
        filename_bytes = raw[filename_start:filename_end]
        local_extra = raw[filename_end:payload_start]
        filename_encoding = "utf-8" if flags & 0x800 else "cp437"
        try:
            local_filename = filename_bytes.decode(filename_encoding)
        except UnicodeDecodeError as error:
            raise ReleaseArtifactInspectionError(
                f"final presentation local ZIP filename is invalid: {member.filename}"
            ) from error
        if (
            local_signature != b"PK\x03\x04"
            or flags & 0x08
            or flags & ~0x806
            or (compression == zipfile.ZIP_STORED and flags & 0x6)
            or not _is_zero_filled_office_zip_padding(local_extra)
            or flags != member.flag_bits
            or compression != member.compress_type
            or crc32 != member.CRC
            or compressed_size != member.compress_size
            or uncompressed_size != member.file_size
            or local_filename != member.orig_filename
        ):
            raise ReleaseArtifactInspectionError(
                "final presentation local ZIP filename/header does not match its central "
                f"directory entry: {member.filename}"
            )
        try:
            content = archive.read(member)
        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
            raise ReleaseArtifactInspectionError(
                f"final presentation ZIP member cannot be decoded: {member.filename}: {error}"
            ) from error
        if len(content) != member.file_size:
            raise ReleaseArtifactInspectionError(
                f"final presentation ZIP member has an invalid decoded size: {member.filename}"
            )
        compressed_payload = raw[payload_start:payload_end]
        if member.compress_type == zipfile.ZIP_STORED:
            exact_decoded_content = compressed_payload
            stream_is_exhausted = len(compressed_payload) == member.file_size
        else:
            try:
                decompressor = zlib.decompressobj(-15)
                exact_decoded_content = decompressor.decompress(
                    compressed_payload,
                    member.file_size + 1,
                )
                remaining = member.file_size + 1 - len(exact_decoded_content)
                if remaining > 0:
                    exact_decoded_content += decompressor.flush(remaining)
            except zlib.error as error:
                raise ReleaseArtifactInspectionError(
                    "final presentation DEFLATE member cannot be decoded exactly: "
                    f"{member.filename}: {error}"
                ) from error
            stream_is_exhausted = (
                decompressor.eof
                and not decompressor.unused_data
                and not decompressor.unconsumed_tail
            )
        if not stream_is_exhausted or exact_decoded_content != content:
            raise ReleaseArtifactInspectionError(
                "final presentation ZIP member contains bytes outside its exact compression "
                f"stream: {member.filename}"
            )
        cursor = payload_end
    if cursor != archive.start_dir:
        raise ReleaseArtifactInspectionError(
            "final presentation container has bytes outside its contiguous local members"
        )
    central_cursor = archive.start_dir
    for member in members:
        if (
            central_cursor + 46 > end_record_offset
            or raw[central_cursor : central_cursor + 4] != b"PK\x01\x02"
        ):
            raise ReleaseArtifactInspectionError(
                "final presentation central directory is truncated or noncanonical"
            )
        central_flags, central_compression = struct.unpack_from("<HH", raw, central_cursor + 8)
        central_crc32, central_compressed_size, central_uncompressed_size = struct.unpack_from(
            "<III",
            raw,
            central_cursor + 16,
        )
        central_filename_length, central_extra_length, central_comment_length = struct.unpack_from(
            "<HHH",
            raw,
            central_cursor + 28,
        )
        central_local_offset = struct.unpack_from("<I", raw, central_cursor + 42)[0]
        central_filename_start = central_cursor + 46
        central_filename_end = central_filename_start + central_filename_length
        central_entry_end = central_filename_end + central_extra_length + central_comment_length
        if central_entry_end > end_record_offset:
            raise ReleaseArtifactInspectionError(
                f"final presentation central-directory entry is truncated: {member.filename}"
            )
        central_filename_bytes = raw[central_filename_start:central_filename_end]
        central_filename_encoding = "utf-8" if central_flags & 0x800 else "cp437"
        try:
            central_filename = central_filename_bytes.decode(central_filename_encoding)
        except UnicodeDecodeError as error:
            raise ReleaseArtifactInspectionError(
                f"final presentation central ZIP filename is invalid: {member.filename}"
            ) from error
        if (
            central_extra_length != 0
            or central_comment_length != 0
            or central_filename != member.orig_filename
            or central_flags != member.flag_bits
            or central_compression != member.compress_type
            or central_crc32 != member.CRC
            or central_compressed_size != member.compress_size
            or central_uncompressed_size != member.file_size
            or central_local_offset != member.header_offset
        ):
            raise ReleaseArtifactInspectionError(
                "final presentation central-directory entry does not match its decoded "
                f"member: {member.filename}"
            )
        central_cursor = central_entry_end
    if central_cursor != end_record_offset:
        raise ReleaseArtifactInspectionError(
            "final presentation container has unbound bytes in its central directory"
        )


def _normalized_relationship_target(source_part: str, target: str) -> str:
    if (
        not target
        or "\x00" in target
        or "\\" in target
        or "#" in target
        or "?" in target
        or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) is not None
    ):
        raise ReleaseArtifactInspectionError(
            "presentation relationship target is not a valid internal OPC path"
        )
    if target.startswith("/"):
        normalized = posixpath.normpath(target.lstrip("/"))
    else:
        normalized = posixpath.normpath(posixpath.join(posixpath.dirname(source_part), target))
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise ReleaseArtifactInspectionError(
            "presentation relationship escapes the presentation package"
        )
    return normalized


def _relationship_part_for_source(source_part: str) -> str:
    if not source_part:
        return "_rels/.rels"
    directory, filename = posixpath.split(source_part)
    return posixpath.join(directory, "_rels", f"{filename}.rels")


def _allowed_relationship_families(source_part: str) -> frozenset[str]:
    if not source_part:
        return frozenset(
            {
                "core_properties",
                "extended_properties",
                "office_document",
                "thumbnail",
            }
        )
    if source_part == "ppt/presentation.xml":
        return frozenset(
            {
                "notes_master",
                "presentation_properties",
                "slide",
                "slide_master",
                "table_styles",
                "theme",
                "view_properties",
            }
        )
    if _SLIDE_NAME.fullmatch(source_part) is not None:
        return frozenset({"image", "notes_slide", "slide_layout"})
    if _NOTE_NAME.fullmatch(source_part) is not None:
        return frozenset({"notes_master", "slide"})
    if _NOTES_MASTER_NAME.fullmatch(source_part) is not None:
        return frozenset({"theme"})
    if _SLIDE_LAYOUT_NAME.fullmatch(source_part) is not None:
        return frozenset({"slide_master"})
    if _SLIDE_MASTER_NAME.fullmatch(source_part) is not None:
        return frozenset({"slide_layout", "theme"})
    return frozenset()


def _validate_presentation_relationship_graph(
    archive: zipfile.ZipFile,
    names: set[str],
) -> None:
    relationship_parts = {name for name in names if name.endswith(".rels")}
    payload_parts = names - relationship_parts - {"[Content_Types].xml"}
    reachable_parts: set[str] = set()
    processed_relationship_parts: set[str] = set()
    visited_sources: set[str] = set()
    pending_sources = [""]
    relationship_tag = _qualified(_PACKAGE_RELATIONSHIPS_NAMESPACE, "Relationship")

    while pending_sources:
        source_part = pending_sources.pop()
        if source_part in visited_sources:
            continue
        visited_sources.add(source_part)
        relationship_part = _relationship_part_for_source(source_part)
        if relationship_part not in relationship_parts:
            continue
        allowed_families = _allowed_relationship_families(source_part)
        if not allowed_families:
            raise ReleaseArtifactInspectionError(
                "presentation contains an unexpected relationship part outside the generated "
                f"contract: {relationship_part}"
            )
        root = _parse_xml_part(
            archive,
            names,
            relationship_part,
            label="presentation relationship graph",
        )
        _require_relationship_root(
            root,
            label=f"presentation relationship part {relationship_part}",
        )
        if root.attrib:
            raise ReleaseArtifactInspectionError(
                f"presentation relationship part {relationship_part} has unexpected metadata"
            )
        processed_relationship_parts.add(relationship_part)
        relationship_ids: set[str] = set()
        for relationship in root:
            if relationship.tag != relationship_tag:
                raise ReleaseArtifactInspectionError(
                    f"presentation relationship part {relationship_part} has an unexpected entry"
                )
            if not set(relationship.attrib).issubset({"Id", "Target", "TargetMode", "Type"}):
                raise ReleaseArtifactInspectionError(
                    f"presentation relationship part {relationship_part} has unexpected fields"
                )
            relationship_id = relationship.attrib.get("Id", "")
            relationship_type = relationship.attrib.get("Type", "")
            target = relationship.attrib.get("Target", "")
            if not relationship_id or relationship_id in relationship_ids:
                raise ReleaseArtifactInspectionError(
                    f"presentation relationship part {relationship_part} has invalid or "
                    "duplicate relationship ids"
                )
            relationship_ids.add(relationship_id)
            target_mode = relationship.attrib.get("TargetMode")
            if target_mode is not None and target_mode != "Internal":
                raise ReleaseArtifactInspectionError(
                    "presentation must not contain an external relationship: "
                    f"{relationship_part} {relationship_id}"
                )
            family = _RELATIONSHIP_FAMILY_BY_TYPE.get(relationship_type)
            if family is None or family not in allowed_families:
                raise ReleaseArtifactInspectionError(
                    "presentation relationship is outside the generated deck contract: "
                    f"{relationship_part} {relationship_type!r}"
                )
            normalized_target = _normalized_relationship_target(source_part, target)
            if _RELATIONSHIP_TARGET_PATTERNS[family].fullmatch(normalized_target) is None:
                raise ReleaseArtifactInspectionError(
                    "presentation relationship target does not match its declared type: "
                    f"{relationship_part} -> {normalized_target}"
                )
            if normalized_target not in payload_parts:
                raise ReleaseArtifactInspectionError(
                    "presentation relationship target is missing from the package: "
                    f"{normalized_target}"
                )
            if normalized_target not in reachable_parts:
                reachable_parts.add(normalized_target)
                pending_sources.append(normalized_target)

    unreferenced_parts = sorted(payload_parts - reachable_parts)
    if unreferenced_parts:
        raise ReleaseArtifactInspectionError(
            f"presentation contains unreferenced payload parts: {unreferenced_parts}"
        )
    unexpected_relationship_parts = sorted(relationship_parts - processed_relationship_parts)
    if unexpected_relationship_parts:
        raise ReleaseArtifactInspectionError(
            "presentation contains unreferenced or unexpected relationship parts: "
            f"{unexpected_relationship_parts}"
        )


def _qualified(namespace: str, local_name: str) -> str:
    return f"{{{namespace}}}{local_name}"


def _parse_xml_part(
    archive: zipfile.ZipFile,
    names: set[str],
    part_name: str,
    *,
    label: str,
) -> ElementTree.Element:
    if part_name not in names:
        raise ReleaseArtifactInspectionError(f"{label} is missing required part {part_name}")
    try:
        return ElementTree.fromstring(archive.read(part_name))
    except (ElementTree.ParseError, KeyError, OSError) as error:
        raise ReleaseArtifactInspectionError(
            f"{label} part {part_name} is invalid XML: {error}"
        ) from error


def _require_relationship_root(
    root: ElementTree.Element,
    *,
    label: str,
) -> None:
    if root.tag != _qualified(_PACKAGE_RELATIONSHIPS_NAMESPACE, "Relationships"):
        raise ReleaseArtifactInspectionError(
            f"{label} must use the supported OPC relationships namespace"
        )


def _expected_presentation_xml_roots(part_name: str) -> frozenset[str]:
    if part_name == "[Content_Types].xml":
        return frozenset({_qualified(_CONTENT_TYPES_NAMESPACE, "Types")})
    if part_name.endswith(".rels"):
        return frozenset({_qualified(_PACKAGE_RELATIONSHIPS_NAMESPACE, "Relationships")})
    exact_roots = {
        "docProps/app.xml": frozenset({_qualified(_EXTENDED_PROPERTIES_NAMESPACE, "Properties")}),
        "docProps/core.xml": frozenset({_qualified(_CORE_PROPERTIES_NAMESPACE, "coreProperties")}),
        "ppt/presentation.xml": frozenset(
            _qualified(namespace, "presentation") for namespace in _PRESENTATION_NAMESPACES
        ),
        "ppt/presProps.xml": frozenset(
            _qualified(namespace, "presentationPr") for namespace in _PRESENTATION_NAMESPACES
        ),
        "ppt/tableStyles.xml": frozenset(
            _qualified(namespace, "tblStyleLst") for namespace in _DRAWING_NAMESPACES
        ),
        "ppt/viewProps.xml": frozenset(
            _qualified(namespace, "viewPr") for namespace in _PRESENTATION_NAMESPACES
        ),
    }
    if roots := exact_roots.get(part_name):
        return roots
    numbered_roots = (
        (_NOTES_MASTER_NAME, _PRESENTATION_NAMESPACES, "notesMaster"),
        (_NOTE_NAME, _PRESENTATION_NAMESPACES, "notes"),
        (_SLIDE_LAYOUT_NAME, _PRESENTATION_NAMESPACES, "sldLayout"),
        (_SLIDE_MASTER_NAME, _PRESENTATION_NAMESPACES, "sldMaster"),
        (_SLIDE_NAME, _PRESENTATION_NAMESPACES, "sld"),
        (_THEME_NAME, _DRAWING_NAMESPACES, "theme"),
    )
    for pattern, namespaces, local_name in numbered_roots:
        if pattern.fullmatch(part_name) is not None:
            return frozenset(_qualified(namespace, local_name) for namespace in namespaces)
    raise ReleaseArtifactInspectionError(
        f"presentation XML part is outside the generated deck contract: {part_name}"
    )


def _require_passive_presentation_xml(
    root: ElementTree.Element,
    *,
    part_name: str,
) -> None:
    for element in root.iter():
        local_name = element.tag.rsplit("}", maxsplit=1)[-1].casefold()
        if local_name in _ACTIVE_PRESENTATION_XML_ELEMENTS:
            raise ReleaseArtifactInspectionError(
                f"presentation must not contain active XML element {local_name!r} in {part_name}"
            )
        for attribute_name, attribute_value in element.attrib.items():
            attribute_local_name = attribute_name.rsplit("}", maxsplit=1)[-1].casefold()
            normalized_value = attribute_value.strip().casefold()
            if attribute_local_name == "action" or normalized_value.startswith(
                _ACTIVE_PRESENTATION_ATTRIBUTE_SCHEMES
            ):
                raise ReleaseArtifactInspectionError(
                    "presentation must not contain an active XML action or external URI in "
                    f"{part_name}: {attribute_local_name!r}"
                )


def _validate_presentation_xml_payloads(
    archive: zipfile.ZipFile,
    names: set[str],
) -> None:
    for part_name in sorted(names):
        if not (
            part_name == "[Content_Types].xml"
            or part_name.endswith(".xml")
            or part_name.endswith(".rels")
        ):
            continue
        data = archive.read(part_name)
        if len(data) > _MAX_PRESENTATION_XML_BYTES:
            raise ReleaseArtifactInspectionError(
                f"presentation XML part exceeds the generated deck limit: {part_name}"
            )
        without_declaration = re.sub(
            rb"^\s*<\?xml[^?]*\?>",
            b"",
            data,
            count=1,
            flags=re.IGNORECASE,
        )
        upper_payload = without_declaration.upper()
        if any(
            marker in upper_payload
            for marker in (b"<!--", b"<![CDATA[", b"<!DOCTYPE", b"<!ENTITY", b"<?")
        ):
            raise ReleaseArtifactInspectionError(
                f"presentation XML part contains unsupported hidden markup: {part_name}"
            )
        try:
            root = ElementTree.fromstring(data)
        except ElementTree.ParseError as error:
            raise ReleaseArtifactInspectionError(
                f"presentation XML payload is invalid XML: {part_name}: {error}"
            ) from error
        if root.tag not in _expected_presentation_xml_roots(part_name):
            raise ReleaseArtifactInspectionError(
                f"presentation XML payload has an unexpected root for {part_name}: {root.tag!r}"
            )
        if part_name != "[Content_Types].xml" and not part_name.endswith(".rels"):
            _require_passive_presentation_xml(root, part_name=part_name)


def _validate_png_image(data: bytes, *, part_name: str) -> None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ReleaseArtifactInspectionError(
            f"presentation PNG image has an invalid encoding signature: {part_name}"
        )
    offset = 8
    chunk_types: list[bytes] = []
    idat_payloads: list[bytes] = []
    width = height = channels = 0
    while offset < len(data):
        if offset + 12 > len(data):
            raise ReleaseArtifactInspectionError(
                f"presentation PNG image has a truncated chunk: {part_name}"
            )
        chunk_length = struct.unpack_from(">I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        payload_start = offset + 8
        payload_end = payload_start + chunk_length
        crc_end = payload_end + 4
        if crc_end > len(data):
            raise ReleaseArtifactInspectionError(
                f"presentation PNG image has an invalid chunk boundary: {part_name}"
            )
        payload = data[payload_start:payload_end]
        expected_crc = struct.unpack_from(">I", data, payload_end)[0]
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise ReleaseArtifactInspectionError(
                f"presentation PNG image has an invalid chunk checksum: {part_name}"
            )
        if chunk_type not in {b"IHDR", b"pHYs", b"IDAT", b"IEND"}:
            raise ReleaseArtifactInspectionError(
                f"presentation PNG image has an unsupported payload chunk: {part_name}"
            )
        chunk_types.append(chunk_type)
        if chunk_type == b"IHDR":
            if len(chunk_types) != 1 or chunk_length != 13:
                raise ReleaseArtifactInspectionError(
                    f"presentation PNG image has an invalid IHDR chunk: {part_name}"
                )
            (
                width,
                height,
                bit_depth,
                color_type,
                compression_method,
                filter_method,
                interlace_method,
            ) = struct.unpack(">IIBBBBB", payload)
            channels = {2: 3, 6: 4}.get(color_type, 0)
            if (
                width <= 0
                or height <= 0
                or width * height > _MAX_PRESENTATION_IMAGE_PIXELS
                or bit_depth != 8
                or channels == 0
                or compression_method != 0
                or filter_method != 0
                or interlace_method != 0
            ):
                raise ReleaseArtifactInspectionError(
                    f"presentation PNG image has unsupported dimensions or pixel mode: {part_name}"
                )
        elif chunk_type == b"pHYs":
            if chunk_length != 9 or chunk_types.count(b"pHYs") != 1 or b"IDAT" in chunk_types:
                raise ReleaseArtifactInspectionError(
                    f"presentation PNG image has an invalid physical-size chunk: {part_name}"
                )
        elif chunk_type == b"IDAT":
            if not payload or b"IEND" in chunk_types:
                raise ReleaseArtifactInspectionError(
                    f"presentation PNG image has an invalid image-data chunk: {part_name}"
                )
            idat_payloads.append(payload)
        else:
            if chunk_length != 0 or crc_end != len(data):
                raise ReleaseArtifactInspectionError(
                    f"presentation PNG image has bytes after its terminal chunk: {part_name}"
                )
        offset = crc_end
    if (
        not chunk_types
        or chunk_types[0] != b"IHDR"
        or chunk_types[-1] != b"IEND"
        or chunk_types.count(b"IHDR") != 1
        or chunk_types.count(b"IEND") != 1
        or not idat_payloads
    ):
        raise ReleaseArtifactInspectionError(
            f"presentation PNG image is structurally incomplete: {part_name}"
        )
    expected_decoded_size = height * (1 + width * channels)
    try:
        decompressor = zlib.decompressobj()
        decoded = decompressor.decompress(
            b"".join(idat_payloads),
            expected_decoded_size + 1,
        )
        remaining = expected_decoded_size + 1 - len(decoded)
        if remaining > 0:
            decoded += decompressor.flush(remaining)
    except zlib.error as error:
        raise ReleaseArtifactInspectionError(
            f"presentation PNG image data cannot be decoded: {part_name}: {error}"
        ) from error
    row_size = 1 + width * channels
    if (
        len(decoded) != expected_decoded_size
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or any(decoded[row_offset] > 4 for row_offset in range(0, len(decoded), row_size))
    ):
        raise ReleaseArtifactInspectionError(
            f"presentation PNG image data does not match its declared encoding: {part_name}"
        )


def _validate_jpeg_thumbnail(data: bytes, *, part_name: str) -> None:
    if not data.startswith(b"\xff\xd8\xff") or not data.endswith(b"\xff\xd9"):
        raise ReleaseArtifactInspectionError(
            f"presentation JPEG thumbnail has an invalid encoding boundary: {part_name}"
        )
    offset = 2
    saw_jfif = saw_frame = saw_scan = saw_end = False
    while offset < len(data):
        marker_offset = offset
        if data[offset] != 0xFF:
            raise ReleaseArtifactInspectionError(
                f"presentation JPEG thumbnail has invalid marker framing: {part_name}"
            )
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            saw_end = True
            if offset != len(data):
                raise ReleaseArtifactInspectionError(
                    f"presentation JPEG thumbnail has trailing bytes: {part_name}"
                )
            break
        if marker not in {0xE0, 0xDB, 0xC0, 0xC4, 0xDD, 0xDA} or offset + 2 > len(data):
            raise ReleaseArtifactInspectionError(
                f"presentation JPEG thumbnail has an unsupported marker: {part_name}"
            )
        segment_length = struct.unpack_from(">H", data, offset)[0]
        segment_end = offset + segment_length
        if segment_length < 2 or segment_end > len(data):
            raise ReleaseArtifactInspectionError(
                f"presentation JPEG thumbnail has an invalid segment boundary: {part_name}"
            )
        payload = data[offset + 2 : segment_end]
        if marker == 0xE0:
            if saw_jfif or not payload.startswith(b"JFIF\x00"):
                raise ReleaseArtifactInspectionError(
                    f"presentation JPEG thumbnail must use one JFIF header: {part_name}"
                )
            saw_jfif = True
        elif marker == 0xC0:
            if saw_frame or len(payload) < 6:
                raise ReleaseArtifactInspectionError(
                    f"presentation JPEG thumbnail has an invalid frame: {part_name}"
                )
            precision = payload[0]
            height, width = struct.unpack_from(">HH", payload, 1)
            component_count = payload[5]
            if (
                precision != 8
                or width <= 0
                or height <= 0
                or width * height > _MAX_PRESENTATION_IMAGE_PIXELS
                or component_count not in {1, 3}
                or len(payload) != 6 + 3 * component_count
            ):
                raise ReleaseArtifactInspectionError(
                    f"presentation JPEG thumbnail has unsupported dimensions or mode: {part_name}"
                )
            saw_frame = True
        elif marker == 0xDA:
            if saw_scan or not saw_frame:
                raise ReleaseArtifactInspectionError(
                    f"presentation JPEG thumbnail has an invalid image scan: {part_name}"
                )
            saw_scan = True
            scan_offset = segment_end
            while scan_offset + 1 < len(data):
                if data[scan_offset] != 0xFF:
                    scan_offset += 1
                    continue
                following = data[scan_offset + 1]
                if following == 0x00 or 0xD0 <= following <= 0xD7:
                    scan_offset += 2
                    continue
                break
            offset = scan_offset
            continue
        offset = segment_end
        if offset <= marker_offset:
            raise ReleaseArtifactInspectionError(
                f"presentation JPEG thumbnail marker did not advance: {part_name}"
            )
    if not (saw_jfif and saw_frame and saw_scan and saw_end):
        raise ReleaseArtifactInspectionError(
            f"presentation JPEG thumbnail is structurally incomplete: {part_name}"
        )


def _validate_presentation_media_payloads(
    archive: zipfile.ZipFile,
    names: set[str],
) -> None:
    for part_name in sorted(name for name in names if _IMAGE_NAME.fullmatch(name) is not None):
        _validate_png_image(archive.read(part_name), part_name=part_name)
    thumbnail_name = "docProps/thumbnail.jpeg"
    if thumbnail_name in names:
        _validate_jpeg_thumbnail(archive.read(thumbnail_name), part_name=thumbnail_name)


def _expected_presentation_part_content_type(part_name: str) -> str | None:
    if content_type := _PRESENTATION_EXACT_PART_CONTENT_TYPES.get(part_name):
        return content_type
    for pattern, content_type in _PRESENTATION_NUMBERED_PART_CONTENT_TYPES:
        if pattern.fullmatch(part_name) is not None:
            return content_type
    return None


def _require_generated_presentation_part(part_name: str) -> str:
    expected = _expected_presentation_part_content_type(part_name)
    if expected is not None:
        return expected
    normalized = f"/{part_name.casefold().strip('/')}"
    if part_name.casefold().endswith(".bin") or any(
        marker in normalized for marker in _ACTIVE_PRESENTATION_PATH_MARKERS
    ):
        raise ReleaseArtifactInspectionError(
            f"presentation contains an embedded/OLE or active payload part: {part_name}"
        )
    raise ReleaseArtifactInspectionError(
        f"presentation contains a part outside the generated presentation contract: {part_name}"
    )


def _validate_presentation_content_types(
    archive: zipfile.ZipFile,
    names: set[str],
    *,
    expected_slide_count: int,
    expected_note_count: int,
) -> None:
    label = "presentation graph"
    root = _parse_xml_part(archive, names, "[Content_Types].xml", label=label)
    if root.tag != _qualified(_CONTENT_TYPES_NAMESPACE, "Types"):
        raise ReleaseArtifactInspectionError(
            "presentation graph has an invalid OPC content-types root"
        )
    if root.attrib:
        raise ReleaseArtifactInspectionError(
            "presentation graph content types must not contain unexpected root metadata"
        )
    defaults: dict[str, str] = {}
    overrides: dict[str, str] = {}
    default_tag = _qualified(_CONTENT_TYPES_NAMESPACE, "Default")
    override_tag = _qualified(_CONTENT_TYPES_NAMESPACE, "Override")
    for element in root:
        if element.tag == default_tag:
            extension = element.attrib.get("Extension", "")
            content_type = element.attrib.get("ContentType", "")
            if (
                not extension
                or set(element.attrib) != {"ContentType", "Extension"}
                or extension != extension.casefold()
                or extension in defaults
                or _PRESENTATION_DEFAULT_CONTENT_TYPES.get(extension) != content_type
            ):
                raise ReleaseArtifactInspectionError(
                    "presentation graph has an invalid or unsupported content-type default"
                )
            defaults[extension] = content_type
            continue
        if element.tag != override_tag:
            raise ReleaseArtifactInspectionError(
                "presentation graph has an unsupported content-types entry"
            )
        part_name = element.attrib.get("PartName", "")
        content_type = element.attrib.get("ContentType", "")
        if (
            not part_name.startswith("/")
            or set(element.attrib) != {"ContentType", "PartName"}
            or part_name.startswith("//")
            or part_name in overrides
            or part_name[1:] not in names
        ):
            raise ReleaseArtifactInspectionError(
                "presentation graph has invalid or duplicate content-type overrides"
            )
        overrides[part_name] = content_type
    expected = {"/ppt/presentation.xml": _PRESENTATION_CONTENT_TYPE}
    expected.update(
        {
            f"/ppt/slides/slide{index}.xml": _SLIDE_CONTENT_TYPE
            for index in range(1, expected_slide_count + 1)
        }
    )
    expected.update(
        {
            f"/ppt/notesSlides/notesSlide{index}.xml": _NOTES_CONTENT_TYPE
            for index in range(1, expected_note_count + 1)
        }
    )
    for part_name, content_type in expected.items():
        if overrides.get(part_name) != content_type:
            raise ReleaseArtifactInspectionError(
                f"presentation graph content types do not bind required part {part_name}"
            )
    for part_name in sorted(names):
        if part_name == "[Content_Types].xml":
            continue
        expected_content_type = (
            _RELATIONSHIPS_CONTENT_TYPE
            if part_name.endswith(".rels")
            else _require_generated_presentation_part(part_name)
        )
        extension = part_name.rsplit(".", maxsplit=1)[-1].casefold()
        actual_content_type = overrides.get(
            f"/{part_name}",
            defaults.get(extension, ""),
        )
        if actual_content_type != expected_content_type:
            raise ReleaseArtifactInspectionError(
                "presentation content type does not match the generated part contract for "
                f"{part_name}: expected {expected_content_type!r}, found {actual_content_type!r}"
            )


def _validate_root_presentation_relationship(
    archive: zipfile.ZipFile,
    names: set[str],
) -> None:
    root = _parse_xml_part(archive, names, "_rels/.rels", label="presentation graph")
    _require_relationship_root(root, label="presentation graph root relationships")
    relationship_tag = _qualified(_PACKAGE_RELATIONSHIPS_NAMESPACE, "Relationship")
    office_document_relationships = [
        element
        for element in root
        if element.tag == relationship_tag
        and element.attrib.get("Type") in _OFFICE_DOCUMENT_RELATIONSHIP_TYPES
    ]
    if len(office_document_relationships) != 1:
        raise ReleaseArtifactInspectionError(
            "presentation graph must have exactly one office-document relationship"
        )
    relationship = office_document_relationships[0]
    if relationship.attrib.get("TargetMode", "Internal") != "Internal":
        raise ReleaseArtifactInspectionError(
            "presentation graph office-document relationship must be internal"
        )
    target = _normalized_relationship_target("", relationship.attrib.get("Target", ""))
    if target != "ppt/presentation.xml":
        raise ReleaseArtifactInspectionError(
            "presentation graph office-document relationship must target ppt/presentation.xml"
        )


def _presentation_slide_relationship_ids(
    root: ElementTree.Element,
    *,
    expected_slide_count: int,
) -> list[str]:
    presentation_namespace = next(
        (
            namespace
            for namespace in _PRESENTATION_NAMESPACES
            if root.tag == _qualified(namespace, "presentation")
        ),
        None,
    )
    if presentation_namespace is None:
        raise ReleaseArtifactInspectionError(
            "presentation graph has an invalid PresentationML root"
        )
    slide_id_lists = list(root.iter(_qualified(presentation_namespace, "sldIdLst")))
    if len(slide_id_lists) != 1:
        raise ReleaseArtifactInspectionError(
            "presentation graph must contain exactly one slide-id list"
        )
    slide_ids = [
        child
        for child in slide_id_lists[0]
        if child.tag == _qualified(presentation_namespace, "sldId")
    ]
    if len(slide_ids) != expected_slide_count:
        raise ReleaseArtifactInspectionError(
            f"presentation graph must reference exactly {expected_slide_count} slides"
        )
    relationship_ids: list[str] = []
    for slide_index, slide_id in enumerate(slide_ids, start=1):
        candidates = [
            slide_id.attrib[_qualified(namespace, "id")]
            for namespace in _OFFICE_RELATIONSHIPS_NAMESPACES
            if _qualified(namespace, "id") in slide_id.attrib
        ]
        if len(candidates) != 1 or not candidates[0]:
            raise ReleaseArtifactInspectionError(
                f"presentation graph slide {slide_index} has no unique relationship id"
            )
        relationship_ids.append(candidates[0])
    if len(set(relationship_ids)) != expected_slide_count:
        raise ReleaseArtifactInspectionError(
            "presentation graph slide relationship ids must be unique"
        )
    return relationship_ids


def _presentation_relationship_targets(
    root: ElementTree.Element,
    *,
    expected_slide_count: int,
) -> dict[str, str]:
    _require_relationship_root(root, label="presentation graph presentation relationships")
    relationship_tag = _qualified(_PACKAGE_RELATIONSHIPS_NAMESPACE, "Relationship")
    targets: dict[str, str] = {}
    for relationship in root:
        if (
            relationship.tag != relationship_tag
            or relationship.attrib.get("Type") not in _SLIDE_RELATIONSHIP_TYPES
        ):
            continue
        relationship_id = relationship.attrib.get("Id", "")
        if not relationship_id or relationship_id in targets:
            raise ReleaseArtifactInspectionError(
                "presentation graph has invalid or duplicate slide relationships"
            )
        if relationship.attrib.get("TargetMode", "Internal") != "Internal":
            raise ReleaseArtifactInspectionError(
                "presentation graph slide relationships must be internal"
            )
        targets[relationship_id] = _normalized_relationship_target(
            "ppt/presentation.xml",
            relationship.attrib.get("Target", ""),
        )
    if len(targets) != expected_slide_count:
        raise ReleaseArtifactInspectionError(
            f"presentation graph must define exactly {expected_slide_count} slide relationships"
        )
    return targets


def _validate_presentation_graph(
    archive: zipfile.ZipFile,
    names: set[str],
    *,
    expected_slide_count: int,
    expected_note_count: int,
) -> list[str]:
    _validate_presentation_content_types(
        archive,
        names,
        expected_slide_count=expected_slide_count,
        expected_note_count=expected_note_count,
    )
    _validate_root_presentation_relationship(archive, names)
    presentation = _parse_xml_part(
        archive,
        names,
        "ppt/presentation.xml",
        label="presentation graph",
    )
    relationship_ids = _presentation_slide_relationship_ids(
        presentation, expected_slide_count=expected_slide_count
    )
    relationships = _parse_xml_part(
        archive,
        names,
        "ppt/_rels/presentation.xml.rels",
        label="presentation graph",
    )
    targets_by_id = _presentation_relationship_targets(
        relationships, expected_slide_count=expected_slide_count
    )
    try:
        ordered_targets = [targets_by_id[relationship_id] for relationship_id in relationship_ids]
    except KeyError as error:
        raise ReleaseArtifactInspectionError(
            f"presentation graph slide id has no relationship: {error.args[0]}"
        ) from error
    expected_targets = [
        f"ppt/slides/slide{index}.xml" for index in range(1, expected_slide_count + 1)
    ]
    if ordered_targets != expected_targets:
        raise ReleaseArtifactInspectionError(
            "presentation graph must reference contiguous slide parts in deck order"
        )
    if not set(expected_targets).issubset(names):
        raise ReleaseArtifactInspectionError("presentation graph references a missing slide part")
    return ordered_targets


def _note_target_for_slide(
    archive: zipfile.ZipFile,
    names: set[str],
    slide_part: str,
    *,
    slide_index: int,
) -> str:
    relationship_part = _relationship_part_for_source(slide_part)
    if relationship_part not in names:
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} is missing its notes relationship part"
        )
    try:
        root = ElementTree.fromstring(archive.read(relationship_part))
    except (ElementTree.ParseError, KeyError, OSError) as error:
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} notes relationship XML is invalid: {error}"
        ) from error
    if root.tag != _qualified(_PACKAGE_RELATIONSHIPS_NAMESPACE, "Relationships"):
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} must use a supported notes relationship schema"
        )
    relationship_tag = _qualified(_PACKAGE_RELATIONSHIPS_NAMESPACE, "Relationship")
    relationships = [
        element
        for element in root
        if element.tag == relationship_tag
        and element.attrib.get("Type") in _NOTES_RELATIONSHIP_TYPES
    ]
    if len(relationships) != 1:
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} must have exactly one supported notes relationship"
        )
    relationship = relationships[0]
    if relationship.attrib.get("TargetMode", "Internal") != "Internal":
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} notes relationship must be internal"
        )
    target = _normalized_relationship_target(
        slide_part,
        relationship.attrib.get("Target", ""),
    )
    if target not in names:
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} notes relationship target is missing: {target}"
        )
    return target


def _require_note_slide_back_reference(
    archive: zipfile.ZipFile,
    names: set[str],
    note_part: str,
    *,
    slide_part: str,
    slide_index: int,
) -> None:
    relationship_part = _relationship_part_for_source(note_part)
    root = _parse_xml_part(
        archive,
        names,
        relationship_part,
        label=f"slide {slide_index} note back-reference",
    )
    _require_relationship_root(
        root,
        label=f"slide {slide_index} note back-reference",
    )
    relationship_tag = _qualified(_PACKAGE_RELATIONSHIPS_NAMESPACE, "Relationship")
    slide_relationships = [
        relationship
        for relationship in root
        if relationship.tag == relationship_tag
        and relationship.attrib.get("Type") in _SLIDE_RELATIONSHIP_TYPES
    ]
    if len(slide_relationships) != 1:
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} note must have exactly one slide back-reference"
        )
    relationship = slide_relationships[0]
    if relationship.attrib.get("TargetMode", "Internal") != "Internal":
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} note back-reference must be internal"
        )
    actual_target = _normalized_relationship_target(
        note_part,
        relationship.attrib.get("Target", ""),
    )
    expected_target = slide_part
    if actual_target != expected_target:
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} note back-reference must target {expected_target}"
        )


def _require_nonempty_note_text(
    archive: zipfile.ZipFile,
    note_part: str,
    *,
    slide_index: int,
) -> None:
    try:
        root = ElementTree.fromstring(archive.read(note_part))
    except (ElementTree.ParseError, KeyError, OSError) as error:
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} speaker-note XML is invalid: {error}"
        ) from error
    presentation_namespace = next(
        (
            namespace
            for namespace in _PRESENTATION_NAMESPACES
            if root.tag == _qualified(namespace, "notes")
        ),
        None,
    )
    if presentation_namespace is None:
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} must have a PresentationML notes root"
        )
    body_text: list[str] = []
    shape_tag = _qualified(presentation_namespace, "sp")
    placeholder_tag = _qualified(presentation_namespace, "ph")
    drawing_text_tags = {_qualified(namespace, "t") for namespace in _DRAWING_NAMESPACES}
    for shape in root.iter(shape_tag):
        has_body_placeholder = any(
            placeholder.attrib.get("type") == "body" for placeholder in shape.iter(placeholder_tag)
        )
        if not has_body_placeholder:
            continue
        body_text.extend(
            element.text.strip()
            for element in shape.iter()
            if element.tag in drawing_text_tags
            and element.text is not None
            and element.text.strip()
        )
    if not body_text:
        raise ReleaseArtifactInspectionError(
            f"slide {slide_index} notes body must contain nonempty speaker-note text in DrawingML"
        )


def inspect_presentation_artifact(
    path: Path,
    relative_path: str,
    *,
    expected_slide_count: int,
    expected_note_count: int,
) -> dict[str, object]:
    """Return identity, strict OPC counts, and slide-ordered resolved note parts."""
    identity, raw = _artifact_snapshot(path, relative_path)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            member_names = _validated_presentation_member_names(archive)
            _require_canonical_presentation_container(raw, archive)
            names = set(member_names)
            slide_indices = _indexed_parts(member_names, _SLIDE_NAME)
            expected_slide_indices = list(range(1, expected_slide_count + 1))
            if slide_indices != expected_slide_indices:
                raise ReleaseArtifactInspectionError(
                    f"release requires exactly {expected_slide_count} "
                    "presentation slides with contiguous indices; "
                    f"detected {slide_indices}"
                )
            note_indices = _indexed_parts(member_names, _NOTE_NAME)
            expected_note_indices = list(range(1, expected_note_count + 1))
            if note_indices != expected_note_indices:
                raise ReleaseArtifactInspectionError(
                    f"release requires exactly {expected_note_count} "
                    "speaker-note slides with contiguous indices; "
                    f"detected {note_indices}"
                )
            referenced_slide_parts = _validate_presentation_graph(
                archive,
                names,
                expected_slide_count=expected_slide_count,
                expected_note_count=expected_note_count,
            )
            resolved_note_parts: list[str] = []
            for slide_index, slide_part in enumerate(referenced_slide_parts, start=1):
                note_part = _note_target_for_slide(
                    archive,
                    names,
                    slide_part,
                    slide_index=slide_index,
                )
                resolved_note_parts.append(note_part)
                _require_nonempty_note_text(archive, note_part, slide_index=slide_index)
                _require_note_slide_back_reference(
                    archive,
                    names,
                    note_part,
                    slide_part=slide_part,
                    slide_index=slide_index,
                )
            if len(set(resolved_note_parts)) != expected_note_count:
                raise ReleaseArtifactInspectionError(
                    "each slide must resolve to a distinct speaker-note slide"
                )
            expected_note_parts = {
                f"ppt/notesSlides/notesSlide{index}.xml"
                for index in range(1, expected_note_count + 1)
            }
            if set(resolved_note_parts) != expected_note_parts:
                raise ReleaseArtifactInspectionError(
                    "speaker-note relationship targets do not cover all note slides"
                )
            _validate_presentation_relationship_graph(archive, names)
            _validate_presentation_xml_payloads(archive, names)
            _validate_presentation_media_payloads(archive, names)
    except ReleaseArtifactInspectionError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ReleaseArtifactInspectionError(
            f"cannot parse final presentation package: {error}"
        ) from error
    return {
        **identity,
        "detected_slide_count": len(referenced_slide_parts),
        "detected_speaker_note_slide_count": len(resolved_note_parts),
        "resolved_speaker_note_parts": resolved_note_parts,
    }
