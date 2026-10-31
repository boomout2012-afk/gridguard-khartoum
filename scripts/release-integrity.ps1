# Shared release-integrity primitives for the build gate and presentation launcher.
#
# We deliberately bind the unpacked PyInstaller onedir instead of producing a ZIP here.
# Compress-Archive records filesystem timestamps and does not promise byte-for-byte stable
# archives across PowerShell/.NET versions. A sorted per-file inventory plus a canonical tree
# digest provides deterministic, actionable verification without claiming archive reproducibility.

$script:DistributionTreeHashAlgorithm = "sha256(canonical-lines-v1)"
$script:AuditedDistributionSizeBytes = [int64]7021231590
$script:RequiredDistributionReductionBytes = [int64]2000000000
$script:DefaultMaximumDistributionSizeBytes = [int64](
    $script:AuditedDistributionSizeBytes - $script:RequiredDistributionReductionBytes
)
$script:ReleaseId = "gridguard-khartoum-v1.2.0"
$script:ReleaseDate = "2026-08-27"
$script:ExpectedBuildKind = "PyInstaller one-directory"
$script:UnsignedTrustBoundary = `
    "UNSIGNED_INTERNAL_CONSISTENCY_REQUIRES_INDEPENDENTLY_TRUSTED_GIT_COMMIT"
$script:RequiredPackagedRuntimeDataPaths = @(
    "_internal/gridguard/_bundled/data/evidence/frozen-benchmark.json",
    "_internal/gridguard/_bundled/data/evidence/paired-benchmark.json",
    "_internal/gridguard/_bundled/data/evidence/robustness-supplement.json",
    "_internal/gridguard/_bundled/data/evidence/proposal-supplement.json",
    "_internal/gridguard/_bundled/data/evidence/focused-ppo-study.json",
    "_internal/gridguard/_bundled/data/evidence/focused-ppo-training.json",
    "_internal/gridguard/_bundled/docs/focused-ppo-validation.json",
    "_internal/gridguard/_bundled/data/evidence/source-manifest.json",
    "_internal/gridguard/_bundled/data/model/manifest.json",
    "_internal/gridguard/_bundled/data/model/ppo-policy.npz",
    "_internal/gridguard/_bundled/data/model/focused-ppo-study/seed-8111-selected.npz",
    "_internal/gridguard/_bundled/data/model/training-provenance.yaml",
    "_internal/gridguard/_bundled/docs/source/graduation-project-proposal.pdf",
    "_internal/gridguard/_bundled/docs/PROPOSAL_EXTENSION.md",
    "_internal/gridguard/_bundled/tools/evaluate_proposal_extension.py",
    "_internal/gridguard/assets/fonts/AtkinsonHyperlegible-Bold.ttf",
    "_internal/gridguard/assets/fonts/AtkinsonHyperlegible-Regular.ttf",
    "_internal/gridguard/assets/fonts/IBMPlexMono-Regular.ttf",
    "_internal/gridguard/assets/fonts/IBMPlexMono-SemiBold.ttf",
    "_internal/gridguard/assets/fonts/SourceSerif4-Regular.ttf",
    "_internal/gridguard/assets/fonts/SourceSerif4-Semibold.ttf",
    "_internal/gridguard/assets/fonts/THIRD_PARTY_FONTS.md",
    "_internal/gridguard/assets/fonts/licenses/AtkinsonHyperlegible-OFL.txt",
    "_internal/gridguard/assets/fonts/licenses/IBMPlexMono-license.txt",
    "_internal/gridguard/assets/fonts/licenses/SourceSerif4-LICENSE.md",
    "_internal/gridguard/assets/map/THIRD_PARTY_DATA.md",
    "_internal/gridguard/assets/map/khartoum-localities.geojson",
    "_internal/gridguard/assets/map/khartoum-nile.geojson",
    "_internal/gridguard/assets/map/map-provenance.json"
)
$script:RequiredTrackedArtifactPaths = @(
    "data/evidence/paired-benchmark.json",
    "data/evidence/robustness-supplement.json",
    "data/evidence/source-manifest.json",
    "data/model/manifest.json",
    "data/model/ppo-policy.npz",
    "data/model/training-provenance.yaml",
    "docs/DEVILS_ADVOCATE_AUDIT.md",
    "docs/REVIEW_REMEDIATION.md",
    "docs/decisions/ADR-002-geographic-boundary-map.md",
    "docs/release-machine-validation.json",
    "docs/release-visual-review.json",
    "docs/source/graduation-project-proposal.pdf",
    "output/pdf/gridguard-khartoum-final-report.pdf",
    "output/presentation/gridguard-khartoum-defense.pptx",
    "src/gridguard/assets/map/THIRD_PARTY_DATA.md",
    "src/gridguard/assets/map/khartoum-localities.geojson",
    "src/gridguard/assets/map/khartoum-nile.geojson",
    "src/gridguard/assets/map/map-provenance.json"
)
$script:AllowedPostSourceAttestationPaths = @(
    "RELEASE_MANIFEST.json",
    "docs/release-machine-validation.json"
)

if ($null -eq ("GridGuard.ReleaseFileSystemInspector" -as [type])) {
    $releaseFileSystemInspector = @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace GridGuard
{
    public static class ReleaseFileSystemInspector
    {
        private const int FindStreamInfoStandard = 0;
        private const int ErrorHandleEof = 38;
        private const uint FileShareRead = 0x00000001;
        private const uint FileShareWrite = 0x00000002;
        private const uint FileShareDelete = 0x00000004;
        private const uint OpenExisting = 3;
        private const uint FileFlagBackupSemantics = 0x02000000;

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct Win32FindStreamData
        {
            public long StreamSize;

            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 296)]
            public string StreamName;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ByHandleFileInformation
        {
            public uint FileAttributes;
            public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime;
            public System.Runtime.InteropServices.ComTypes.FILETIME LastAccessTime;
            public System.Runtime.InteropServices.ComTypes.FILETIME LastWriteTime;
            public uint VolumeSerialNumber;
            public uint FileSizeHigh;
            public uint FileSizeLow;
            public uint NumberOfLinks;
            public uint FileIndexHigh;
            public uint FileIndexLow;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr FindFirstStreamW(
            string fileName,
            int infoLevel,
            out Win32FindStreamData findStreamData,
            uint flags
        );

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool FindNextStreamW(
            IntPtr findStream,
            out Win32FindStreamData findStreamData
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool FindClose(IntPtr findHandle);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern SafeFileHandle CreateFileW(
            string fileName,
            uint desiredAccess,
            uint shareMode,
            IntPtr securityAttributes,
            uint creationDisposition,
            uint flagsAndAttributes,
            IntPtr templateFile
        );

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetFileInformationByHandle(
            SafeFileHandle file,
            out ByHandleFileInformation fileInformation
        );

        public static string[] GetStreamNames(string path)
        {
            var names = new List<string>();
            Win32FindStreamData data;
            IntPtr handle = FindFirstStreamW(path, FindStreamInfoStandard, out data, 0);
            if (handle == new IntPtr(-1))
            {
                int error = Marshal.GetLastWin32Error();
                if (error == ErrorHandleEof)
                {
                    return names.ToArray();
                }
                throw new Win32Exception(error, "Cannot enumerate NTFS streams for " + path);
            }

            try
            {
                do
                {
                    names.Add(data.StreamName);
                }
                while (FindNextStreamW(handle, out data));

                int error = Marshal.GetLastWin32Error();
                if (error != ErrorHandleEof)
                {
                    throw new Win32Exception(error, "Cannot exhaust NTFS streams for " + path);
                }
            }
            finally
            {
                FindClose(handle);
            }
            return names.ToArray();
        }

        public static uint GetHardLinkCount(string path)
        {
            using (SafeFileHandle handle = CreateFileW(
                path,
                0,
                FileShareRead | FileShareWrite | FileShareDelete,
                IntPtr.Zero,
                OpenExisting,
                FileFlagBackupSemantics,
                IntPtr.Zero))
            {
                if (handle.IsInvalid)
                {
                    throw new Win32Exception(
                        Marshal.GetLastWin32Error(),
                        "Cannot open release file for link inspection: " + path
                    );
                }
                ByHandleFileInformation information;
                if (!GetFileInformationByHandle(handle, out information))
                {
                    throw new Win32Exception(
                        Marshal.GetLastWin32Error(),
                        "Cannot inspect release file links: " + path
                    );
                }
                return information.NumberOfLinks;
            }
        }
    }
}
'@
    Add-Type -TypeDefinition $releaseFileSystemInspector -ErrorAction Stop
}

function Assert-ReleaseFileSystemEntrySafety {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath
    )

    if (-not (Test-Path -LiteralPath $LiteralPath)) {
        throw "Release filesystem entry is missing: $LiteralPath"
    }
    $item = Get-Item -LiteralPath $LiteralPath -Force
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Release filesystem entry is a forbidden reparse point: $($item.FullName)"
    }
    $streamNames = @(
        [GridGuard.ReleaseFileSystemInspector]::GetStreamNames($item.FullName)
    )
    $namedStreams = @($streamNames | Where-Object { $_ -cne "::`$DATA" })
    if ($namedStreams.Count -ne 0) {
        throw (
            "Release filesystem entry has a forbidden NTFS alternate data stream: " +
            "$($item.FullName) [$($namedStreams -join ', ')]."
        )
    }
    if (-not $item.PSIsContainer) {
        $linkCount = [GridGuard.ReleaseFileSystemInspector]::GetHardLinkCount($item.FullName)
        if ($linkCount -ne 1) {
            throw (
                "Release file has a forbidden hard link count: " +
                "$($item.FullName) (links=$linkCount)."
            )
        }
    }
    return $item
}

function Assert-ReleaseTreeFileSystemSafety {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootPath
    )

    $root = Assert-ReleaseFileSystemEntrySafety -LiteralPath $RootPath
    if (-not $root.PSIsContainer) {
        throw "Release tree root must be a directory: $RootPath"
    }
    $pending = [System.Collections.Generic.Queue[object]]::new()
    $pending.Enqueue($root)
    while ($pending.Count -ne 0) {
        $directory = $pending.Dequeue()
        $children = @(Get-ChildItem -LiteralPath $directory.FullName -Force -ErrorAction Stop)
        if ($children.Count -eq 0) {
            throw "Release tree contains an unbound empty directory: $($directory.FullName)"
        }
        foreach ($child in $children) {
            $validated = Assert-ReleaseFileSystemEntrySafety -LiteralPath $child.FullName
            if ($validated.PSIsContainer) {
                $pending.Enqueue($validated)
            }
        }
    }
}

function Get-ReleaseSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath
    )

    [void](Assert-ReleaseFileSystemEntrySafety -LiteralPath $LiteralPath)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $LiteralPath).Hash.ToLowerInvariant()
}

function Get-RequiredPackagedProvenanceSourcePaths {
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
    $contractPath = Join-Path $projectRoot "hatch_build.py"
    if (-not (Test-Path -LiteralPath $contractPath -PathType Leaf)) {
        throw "Robustness provenance source contract is unavailable: $contractPath"
    }
    $contractReader = @'
import json
import runpy
import sys
from pathlib import Path

contract = runpy.run_path(sys.argv[1])
paths = [
    relative.as_posix()
    for relative, _source in contract['_validated_provenance_sources'](Path(sys.argv[1]).parent)
]
print(json.dumps(list(dict.fromkeys(paths))))
'@
    $encodedPaths = @(& uv run --no-sync python -c $contractReader $contractPath)
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot derive the packaged provenance-source inventory from $contractPath"
    }
    try {
        $decodedPaths = ConvertFrom-Json -InputObject ($encodedPaths -join "`n")
        $repositoryPaths = @($decodedPaths | ForEach-Object { $_ })
    }
    catch {
        throw "Packaged provenance-source inventory derivation returned invalid JSON."
    }
    if ($repositoryPaths.Count -eq 0) {
        throw "Packaged provenance-source inventory must not be empty."
    }
    $packagedPaths = [System.Collections.Generic.List[string]]::new()
    $seenPaths = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::Ordinal
    )
    foreach ($repositoryPathValue in $repositoryPaths) {
        $repositoryPath = [string]$repositoryPathValue
        if (
            $repositoryPath -notmatch "^(data/evidence/accepted-sources/)?src/gridguard/[A-Za-z0-9_]+\.py$" -or
            -not $seenPaths.Add($repositoryPath)
        ) {
            throw "Invalid or duplicate packaged provenance source path: $repositoryPath"
        }
        $packagedPaths.Add("_internal/gridguard/_bundled/$repositoryPath")
    }
    return @($packagedPaths | Sort-Object -CaseSensitive)
}

function Assert-ReleaseArchiveStructure {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WheelPath,
        [Parameter(Mandatory = $true)]
        [string]$SdistPath,
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    [void](Assert-ReleaseFileSystemEntrySafety -LiteralPath $WheelPath)
    [void](Assert-ReleaseFileSystemEntrySafety -LiteralPath $SdistPath)

    $expectedWheelDataPaths = @(
        @($script:RequiredPackagedRuntimeDataPaths) +
        @(Get-RequiredPackagedProvenanceSourcePaths) |
            ForEach-Object {
                if (-not $_.StartsWith(
                    "_internal/",
                    [System.StringComparison]::Ordinal
                )) {
                    throw "Packaged wheel path cannot be derived from: $_"
                }
                $_.Substring("_internal/".Length)
            } |
            Sort-Object -CaseSensitive -Unique
    )
    $expectedWheelDataJson = ConvertTo-Json `
        -InputObject @($expectedWheelDataPaths) `
        -Compress
    $expectedWheelDataBase64 = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes($expectedWheelDataJson)
    )
    $frozenMetadataPath = Join-Path `
        $ProjectRoot `
        "dist\GridGuard Khartoum\_internal\gridguard_khartoum-1.2.0.dist-info\METADATA"
    [void](Assert-ReleaseFileSystemEntrySafety -LiteralPath $frozenMetadataPath)
    $archiveValidator = @'
import base64
import configparser
import csv
from email import policy
from email.parser import BytesParser
import hashlib
import io
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import sys
import tarfile
import tomllib
import zipfile
import zlib

EXPECTED_NAME = 'gridguard-khartoum'
EXPECTED_VERSION = '1.2.0'
EXPECTED_ROOT = 'gridguard_khartoum-1.2.0'
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_METADATA_HEADERS = 4096
MAX_MEMBERS = 100_000
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
REPRODUCIBLE_ARCHIVE_EPOCH = 1580601600
TAR_BLOCK_SIZE = 512
TAR_RECORD_SIZE = 10240
EXPECTED_CONSOLE_SCRIPTS = {
    'gridguard-desktop': 'gridguard.ui.app:main',
    'gridguard-doctor': 'gridguard.cli:doctor_main',
    'gridguard-evaluate': 'gridguard.cli:main',
}
EXPECTED_DIST_INFO_FILES = {
    f'{EXPECTED_ROOT}.dist-info/METADATA',
    f'{EXPECTED_ROOT}.dist-info/WHEEL',
    f'{EXPECTED_ROOT}.dist-info/entry_points.txt',
    f'{EXPECTED_ROOT}.dist-info/RECORD',
}
METADATA_SINGLETON_FIELDS = (
    'Metadata-Version',
    'Name',
    'Version',
    'Summary',
    'Requires-Python',
)
EXCLUDED_SOURCE_ROOTS = {
    '.codebase-memory',
    '.git',
    '.pytest_cache',
    '.ruff_cache',
    '.uv-cache',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'tmp',
}
WINDOWS_RESERVED_NAMES = {'con', 'prn', 'aux', 'nul'}

def safe_name(name, label):
    if not isinstance(name, str) or not name:
        raise ValueError(f'{label} member path is empty')
    if any(ord(character) < 32 for character in name):
        raise ValueError(f'{label} member path contains a control character: {name!r}')
    raw_parts = name.split('/')
    path = PurePosixPath(name)
    if (
        '\\' in name
        or path.is_absolute()
        or any(part in {'', '.', '..'} for part in raw_parts)
        or path.as_posix() != name
    ):
        raise ValueError(f'unsafe {label} member path: {name!r}')
    for part in raw_parts:
        stem = part.split('.', 1)[0].casefold()
        if (
            ':' in part
            or part.endswith((' ', '.'))
            or stem in WINDOWS_RESERVED_NAMES
            or re.fullmatch(r'(?:com|lpt)[1-9]', stem) is not None
        ):
            raise ValueError(f'non-portable {label} member path: {name!r}')

def secret_like(name):
    for raw_part in name.split('/'):
        part = raw_part.casefold()
        if part == '.env' or (part.startswith('.env.') and part != '.env.example'):
            return True
        if part.startswith(('secret.', 'secrets.')):
            return True
        if part in {
            'credentials.json',
            'id_dsa',
            'id_ecdsa',
            'id_ed25519',
            'id_rsa',
        }:
            return True
        if any(part.endswith(suffix) for suffix in ('.key', '.p12', '.pem', '.pfx')):
            return True
    return False

def exact_unique_names(names, label):
    normalized = []
    seen = set()
    for name in names:
        safe_name(name, label)
        if secret_like(name):
            raise ValueError(f'forbidden secret-like archive member: {name!r}')
        folded = name.casefold()
        if folded in seen:
            raise ValueError(f'duplicate {label} member path: {name!r}')
        seen.add(folded)
        normalized.append(name)
    if not normalized or len(normalized) > MAX_MEMBERS:
        raise ValueError(f'{label} member count is invalid: {len(normalized)}')
    return normalized

def bounded_unpack(format_string, payload, offset, label):
    size = struct.calcsize(format_string)
    if offset < 0 or offset + size > len(payload):
        raise ValueError(f'{label} is truncated at byte offset {offset}')
    return struct.unpack_from(format_string, payload, offset)

def decoded_zip_name(raw_name, flags, label):
    encoding = 'utf-8' if flags & 0x800 else 'cp437'
    try:
        name = raw_name.decode(encoding)
    except UnicodeDecodeError as error:
        raise ValueError(f'{label} name is not valid {encoding}: {error}') from error
    safe_name(name, 'wheel')
    return name

def inflate_zip_member(compressed, compression, expected_size, expected_crc, label):
    if compression == zipfile.ZIP_STORED:
        if len(compressed) != expected_size:
            raise ValueError(f'{label} stored-size contract mismatch')
        payload = compressed
    elif compression == zipfile.ZIP_DEFLATED:
        inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        try:
            payload = inflater.decompress(compressed, MAX_MEMBER_BYTES + 1)
            if inflater.unconsumed_tail:
                raise ValueError(f'{label} exceeds the decompression budget')
            payload += inflater.flush()
        except zlib.error as error:
            raise ValueError(f'{label} has an invalid deflate stream: {error}') from error
        if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
            raise ValueError(f'{label} deflate payload is not exactly exhausted')
    else:
        raise ValueError(f'{label} uses unsupported compression method {compression}')
    if len(payload) != expected_size:
        raise ValueError(
            f'{label} uncompressed-size mismatch: '
            f'expected={expected_size}, actual={len(payload)}'
        )
    if zlib.crc32(payload) & 0xFFFFFFFF != expected_crc:
        raise ValueError(f'{label} CRC-32 mismatch')
    return payload

def validate_wheel_container(path):
    payload = path.read_bytes()
    if len(payload) < 22 or payload[-22:-18] != b'PK\x05\x06':
        raise ValueError('wheel EOCD must be the exact final 22-byte record')
    eocd_offset = len(payload) - 22
    (
        signature,
        disk_number,
        central_disk,
        disk_entries,
        total_entries,
        central_size,
        central_offset,
        archive_comment_size,
    ) = bounded_unpack('<4s4H2IH', payload, eocd_offset, 'wheel EOCD')
    if signature != b'PK\x05\x06':
        raise ValueError('wheel EOCD signature is invalid')
    if archive_comment_size != 0:
        raise ValueError('wheel archive comment must be empty')
    if (
        disk_number != 0
        or central_disk != 0
        or disk_entries != total_entries
        or total_entries in (0, 0xFFFF)
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
    ):
        raise ValueError('wheel must use one non-ZIP64 disk with a nonempty inventory')
    if central_offset + central_size != eocd_offset:
        raise ValueError('wheel central-directory/EOCD ranges are not exactly covered')

    central_records = []
    cursor = central_offset
    for index in range(total_entries):
        values = bounded_unpack(
            '<4s6H3I5H2I',
            payload,
            cursor,
            f'wheel central entry {index}',
        )
        if values[0] != b'PK\x01\x02':
            raise ValueError(f'wheel central entry {index} signature is invalid')
        (
            _signature,
            _created_version,
            required_version,
            flags,
            compression,
            modified_time,
            modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
            member_comment_size,
            disk_start,
            _internal_attributes,
            _external_attributes,
            local_offset,
        ) = values
        if flags not in (0, 0x800):
            raise ValueError(f'wheel central entry {index} has unsupported flags: {flags:#x}')
        if compression not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise ValueError(
                f'wheel central entry {index} has unsupported compression: {compression}'
            )
        if (
            compressed_size == 0xFFFFFFFF
            or uncompressed_size == 0xFFFFFFFF
            or local_offset == 0xFFFFFFFF
            or disk_start != 0
        ):
            raise ValueError('wheel ZIP64 or multi-disk member metadata is not permitted')
        if extra_size != 0 or member_comment_size != 0:
            raise ValueError('wheel member extra fields and comments must be empty')
        record_end = cursor + 46 + name_size
        if record_end > central_offset + central_size:
            raise ValueError(f'wheel central entry {index} exceeds the central directory')
        raw_name = payload[cursor + 46:record_end]
        name = decoded_zip_name(raw_name, flags, f'wheel central entry {index}')
        central_records.append(
            {
                'name': name,
                'raw_name': raw_name,
                'required_version': required_version,
                'flags': flags,
                'compression': compression,
                'modified_time': modified_time,
                'modified_date': modified_date,
                'crc32': crc32,
                'compressed_size': compressed_size,
                'uncompressed_size': uncompressed_size,
                'local_offset': local_offset,
            }
        )
        cursor = record_end
    if cursor != central_offset + central_size:
        raise ValueError('wheel central directory has an uncovered or overlapping range')

    local_cursor = 0
    seen_local_offsets = set()
    for index, record in enumerate(sorted(central_records, key=lambda item: item['local_offset'])):
        member_name = record.get('name')
        if not isinstance(member_name, str):
            raise ValueError(f'wheel central entry {index} has no decoded name')
        local_offset = record['local_offset']
        if local_offset in seen_local_offsets:
            raise ValueError('wheel local-header offsets must be unique')
        seen_local_offsets.add(local_offset)
        if local_offset != local_cursor:
            raise ValueError(
                'wheel local member ranges are not exactly contiguous: '
                f'expected={local_cursor}, actual={local_offset}'
            )
        values = bounded_unpack(
            '<4s5H3I2H',
            payload,
            local_offset,
            f'wheel local entry {index}',
        )
        (
            signature,
            required_version,
            flags,
            compression,
            modified_time,
            modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            name_size,
            extra_size,
        ) = values
        if signature != b'PK\x03\x04':
            raise ValueError(f'wheel local entry {index} signature is invalid')
        if extra_size != 0:
            raise ValueError('wheel local extra fields must be empty')
        name_start = local_offset + 30
        data_start = name_start + name_size
        data_end = data_start + compressed_size
        if data_end > central_offset:
            raise ValueError('wheel local member overlaps the central directory')
        raw_name = payload[name_start:data_start]
        decoded_zip_name(raw_name, flags, f'wheel local entry {index}')
        compared_fields = {
            'raw_name': raw_name,
            'required_version': required_version,
            'flags': flags,
            'compression': compression,
            'modified_time': modified_time,
            'modified_date': modified_date,
            'crc32': crc32,
            'compressed_size': compressed_size,
            'uncompressed_size': uncompressed_size,
        }
        for field, actual in compared_fields.items():
            if actual != record[field]:
                raise ValueError(
                    f'wheel local/central {field} mismatch for {member_name!r}'
                )
        inflate_zip_member(
            payload[data_start:data_end],
            compression,
            uncompressed_size,
            crc32,
            f'wheel member {member_name!r}',
        )
        local_cursor = data_end
    if local_cursor != central_offset:
        raise ValueError('wheel local/central ranges have an uncovered structural gap')
    return central_records

def canonical_tar_octal(value, width):
    encoded = f'{value:0{width - 1}o}\0'.encode('ascii')
    if len(encoded) != width:
        raise ValueError(f'tar numeric value does not fit canonical width {width}: {value}')
    return encoded

def tar_octal(field, width, label):
    if len(field) != width or not field.endswith(b'\0'):
        raise ValueError(f'{label} is not a canonical tar octal field')
    digits = field[:-1]
    if not digits or any(value not in b'01234567' for value in digits):
        raise ValueError(f'{label} is not a canonical tar octal field')
    value = int(digits, 8)
    if field != canonical_tar_octal(value, width):
        raise ValueError(f'{label} is not canonically encoded')
    return value

def tar_name(field, label):
    nul_offset = field.find(b'\0')
    if nul_offset <= 0 or any(field[nul_offset:]):
        raise ValueError(f'{label} is not a canonical null-padded tar name')
    try:
        name = field[:nul_offset].decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError(f'{label} is not UTF-8: {error}') from error
    safe_name(name, 'sdist')
    return name

def validate_tar_container(payload):
    if not payload or len(payload) % TAR_RECORD_SIZE != 0:
        raise ValueError('sdist tar payload must fill exact 10240-byte records')
    names = []
    cursor = 0
    while cursor + TAR_BLOCK_SIZE <= len(payload):
        header = payload[cursor:cursor + TAR_BLOCK_SIZE]
        if header == b'\0' * TAR_BLOCK_SIZE:
            if payload[cursor:cursor + 2 * TAR_BLOCK_SIZE] != b'\0' * (2 * TAR_BLOCK_SIZE):
                raise ValueError('sdist tar EOF must contain at least two zero blocks')
            expected_size = (
                (cursor + 2 * TAR_BLOCK_SIZE + TAR_RECORD_SIZE - 1)
                // TAR_RECORD_SIZE
                * TAR_RECORD_SIZE
            )
            if len(payload) != expected_size or any(payload[cursor:]):
                raise ValueError('sdist tar EOF/padding does not exactly cover its final record')
            return names
        if len(names) >= MAX_MEMBERS:
            raise ValueError('sdist tar member count exceeds the release budget')
        if header[257:263] != b'ustar\0' or header[263:265] != b'00':
            raise ValueError('sdist tar member must use the canonical USTAR header')
        if header[156:157] != b'0':
            raise ValueError('sdist tar must contain canonical regular-file entries only')
        if (
            any(header[157:257])
            or any(header[265:345])
            or any(header[345:500])
            or any(header[500:512])
        ):
            raise ValueError('sdist tar link/owner/device/prefix metadata must be empty')
        stored_checksum = tar_octal(header[148:154] + b'\0', 7, 'sdist tar checksum')
        if header[154:156] != b'\0 ':
            raise ValueError('sdist tar checksum terminator is not canonical')
        checksum_header = header[:148] + b'        ' + header[156:]
        if sum(checksum_header) != stored_checksum:
            raise ValueError('sdist tar header checksum mismatch')
        name = tar_name(header[:100], 'sdist tar member name')
        mode = tar_octal(header[100:108], 8, f'{name} mode')
        uid = tar_octal(header[108:116], 8, f'{name} uid')
        gid = tar_octal(header[116:124], 8, f'{name} gid')
        size = tar_octal(header[124:136], 12, f'{name} size')
        mtime = tar_octal(header[136:148], 12, f'{name} mtime')
        if (
            mode != 0o644
            or uid != 0
            or gid != 0
            or mtime != REPRODUCIBLE_ARCHIVE_EPOCH
        ):
            raise ValueError(
                f'sdist tar metadata is not deterministic for {name!r}: '
                f'mode={mode:o}, uid={uid}, gid={gid}, mtime={mtime}'
            )
        if size > MAX_MEMBER_BYTES:
            raise ValueError(f'sdist tar member exceeds the release budget: {name!r}')
        data_start = cursor + TAR_BLOCK_SIZE
        data_end = data_start + size
        padded_end = (
            (data_end + TAR_BLOCK_SIZE - 1) // TAR_BLOCK_SIZE * TAR_BLOCK_SIZE
        )
        if padded_end > len(payload):
            raise ValueError(f'sdist tar member is truncated: {name!r}')
        if any(payload[data_end:padded_end]):
            raise ValueError(f'sdist tar member padding is nonzero: {name!r}')
        names.append(name)
        cursor = padded_end
    raise ValueError('sdist tar payload has no canonical two-block EOF')

def validate_sdist_container(path):
    payload = path.read_bytes()
    if len(payload) < 18:
        raise ValueError('sdist gzip stream is truncated')
    if payload[:3] != b'\x1f\x8b\x08':
        raise ValueError('sdist must contain a gzip DEFLATE stream')
    flags = payload[3]
    mtime = int.from_bytes(payload[4:8], 'little')
    xfl = payload[8]
    operating_system = payload[9]
    if (
        flags != 0
        or mtime != REPRODUCIBLE_ARCHIVE_EPOCH
        or xfl != 2
        or operating_system != 255
    ):
        raise ValueError(
            'sdist gzip metadata is not deterministic: '
            f'flags={flags:#x}, mtime={mtime}, xfl={xfl}, os={operating_system}'
        )
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        tar_payload = inflater.decompress(payload, MAX_UNCOMPRESSED_BYTES + 1)
        if inflater.unconsumed_tail:
            raise ValueError('sdist gzip payload exceeds the release budget')
        tar_payload += inflater.flush()
    except zlib.error as error:
        raise ValueError(f'sdist gzip stream is invalid: {error}') from error
    if not inflater.eof or inflater.unused_data or inflater.unconsumed_tail:
        raise ValueError('sdist must contain exactly one exhausted gzip stream')
    if len(tar_payload) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError('sdist tar payload exceeds the release budget')
    return tar_payload, validate_tar_container(tar_payload)

def single_header(message, field, label):
    values = message.get_all(field, [])
    if len(values) != 1 or not values[0]:
        raise ValueError(f'{label} must contain exactly one {field} header')
    return str(values[0])

def strict_metadata_message(payload, label):
    if not payload or len(payload) > MAX_METADATA_BYTES or b'\0' in payload:
        raise ValueError(f'{label} metadata size or byte contract is invalid')
    try:
        message = BytesParser(
            policy=policy.default.clone(raise_on_defect=True),
        ).parsebytes(payload)
    except Exception as error:
        raise ValueError(f'{label} metadata parser defect: {error}') from error
    if message.defects:
        raise ValueError(f'{label} metadata parser defects: {message.defects!r}')
    if message.get_unixfrom() is not None:
        raise ValueError(f'{label} metadata contains a forbidden Unix envelope')
    if message.is_multipart():
        raise ValueError(f'{label} metadata must not be multipart')
    if len(list(message.raw_items())) > MAX_METADATA_HEADERS:
        raise ValueError(f'{label} metadata header count exceeds the release budget')
    return message

def read_bounded_regular_file(path, label):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'{label} must be one plain regular file')
    try:
        expected_size = path.stat().st_size
    except OSError as error:
        raise ValueError(f'{label} cannot be inspected: {error}') from error
    if expected_size <= 0 or expected_size > MAX_METADATA_BYTES:
        raise ValueError(f'{label} size is invalid')
    try:
        with path.open('rb') as stream:
            payload = stream.read(MAX_METADATA_BYTES + 1)
    except OSError as error:
        raise ValueError(f'{label} cannot be read: {error}') from error
    if len(payload) > MAX_METADATA_BYTES or len(payload) != expected_size:
        raise ValueError(f'{label} changed or exceeded its size bound while reading')
    return payload

def canonical_requirement(value, label):
    try:
        requirement = Requirement(value)
    except InvalidRequirement as error:
        raise ValueError(f'{label} contains an invalid requirement: {value!r}') from error
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
        str(requirement.specifier),
        requirement.url or '',
        str(requirement.marker) if requirement.marker is not None else '',
    )

def canonical_requires_python(value, label):
    try:
        return str(SpecifierSet(value))
    except InvalidSpecifier as error:
        raise ValueError(f'{label} is not a valid Requires-Python specifier') from error

def validate_metadata(payload, label):
    metadata = strict_metadata_message(payload, label)
    singleton = {
        field: single_header(metadata, field, f'{label} metadata')
        for field in METADATA_SINGLETON_FIELDS
    }
    raw_requirements = [str(value) for value in metadata.get_all('Requires-Dist', [])]
    requirements = [
        canonical_requirement(value, f'{label} metadata Requires-Dist')
        for value in raw_requirements
    ]
    if len(requirements) != len(set(requirements)):
        raise ValueError(f'{label} metadata contains duplicate Requires-Dist semantics')
    raw_extras = [str(value) for value in metadata.get_all('Provides-Extra', [])]
    extras = []
    for value in raw_extras:
        canonical = canonicalize_name(value)
        if value != canonical:
            raise ValueError(
                f'{label} metadata Provides-Extra must be canonical: {value!r}'
            )
        extras.append(canonical)
    if len(extras) != len(set(extras)):
        raise ValueError(f'{label} metadata contains duplicate Provides-Extra semantics')
    return {
        'metadata_version': singleton['Metadata-Version'],
        'name': singleton['Name'],
        'version': singleton['Version'],
        'summary': singleton['Summary'],
        'requires_python': canonical_requires_python(
            singleton['Requires-Python'],
            f'{label} metadata Requires-Python',
        ),
        'requires_dist': tuple(sorted(requirements)),
        'provides_extra': tuple(sorted(extras)),
    }

def project_metadata_contract(project_root):
    pyproject_path = project_root / 'pyproject.toml'
    raw = read_bounded_regular_file(pyproject_path, 'pyproject metadata contract')
    if b'\0' in raw:
        raise ValueError('pyproject metadata contract bytes are invalid')
    try:
        document = tomllib.loads(raw.decode('utf-8'))
        project = document['project']
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise ValueError(f'pyproject metadata contract is invalid: {error}') from error
    required_strings = ('name', 'version', 'description', 'requires-python')
    if any(type(project.get(field)) is not str or not project[field] for field in required_strings):
        raise ValueError('pyproject metadata identity fields must be nonempty strings')
    dependencies = project.get('dependencies')
    optional = project.get('optional-dependencies')
    if type(dependencies) is not list or type(optional) is not dict:
        raise ValueError('pyproject dependency fields have invalid types')
    requirement_values = []
    for index, value in enumerate(dependencies):
        if type(value) is not str or not value:
            raise ValueError(f'pyproject dependency {index} must be a nonempty string')
        requirement_values.append(
            canonical_requirement(value, f'pyproject dependency {index}')
        )
    extras = []
    for raw_extra, values in optional.items():
        if type(raw_extra) is not str or not raw_extra or type(values) is not list:
            raise ValueError('pyproject optional dependency group is invalid')
        extra = canonicalize_name(raw_extra)
        if raw_extra != extra:
            raise ValueError(
                f'pyproject optional dependency group must be canonical: {raw_extra!r}'
            )
        extras.append(extra)
        for index, value in enumerate(values):
            if type(value) is not str or not value:
                raise ValueError(
                    f'pyproject optional dependency {extra}[{index}] must be a string'
                )
            requirement_part, separator, marker_part = value.partition(';')
            marker = f"extra == '{extra}'"
            if separator:
                marker = f'({marker_part.strip()}) and {marker}'
            requirement_values.append(
                canonical_requirement(
                    f'{requirement_part.strip()}; {marker}',
                    f'pyproject optional dependency {extra}[{index}]',
                )
            )
    if len(extras) != len(set(extras)):
        raise ValueError('pyproject contains duplicate optional dependency groups')
    if len(requirement_values) != len(set(requirement_values)):
        raise ValueError('pyproject contains duplicate dependency semantics')
    contract = {
        'metadata_version': '2.5',
        'name': project['name'],
        'version': project['version'],
        'summary': project['description'],
        'requires_python': canonical_requires_python(
            project['requires-python'],
            'pyproject requires-python',
        ),
        'requires_dist': tuple(sorted(requirement_values)),
        'provides_extra': tuple(sorted(extras)),
    }
    if contract['name'] != EXPECTED_NAME or contract['version'] != EXPECTED_VERSION:
        raise ValueError(
            'pyproject metadata identity does not match the exact release name/version'
        )
    return contract

def validate_wheel(payload, label):
    if not payload or len(payload) > MAX_METADATA_BYTES:
        raise ValueError(f'{label} WHEEL size is invalid')
    wheel = BytesParser().parsebytes(payload)
    actual = {
        'Wheel-Version': single_header(wheel, 'Wheel-Version', label),
        'Root-Is-Purelib': single_header(wheel, 'Root-Is-Purelib', label),
        'Tag': wheel.get_all('Tag', []),
    }
    expected = {
        'Wheel-Version': '1.0',
        'Root-Is-Purelib': 'true',
        'Tag': ['py3-none-any'],
    }
    if actual != expected:
        raise ValueError(f'{label} wheel contract mismatch: {actual!r}')

def validate_entry_points(payload, label):
    if not payload or len(payload) > MAX_METADATA_BYTES:
        raise ValueError(f'{label} entry_points.txt size is invalid')
    try:
        text = payload.decode('utf-8')
        parser = configparser.ConfigParser(interpolation=None, strict=True)
        parser.optionxform = str
        parser.read_string(text)
    except (UnicodeDecodeError, configparser.Error) as error:
        raise ValueError(f'{label} entry_points.txt is invalid: {error}') from error
    if parser.sections() != ['console_scripts']:
        raise ValueError(f'{label} console-script contract mismatch: sections={parser.sections()}')
    scripts = dict(parser.items('console_scripts', raw=True))
    if scripts != EXPECTED_CONSOLE_SCRIPTS:
        raise ValueError(f'{label} console-script contract mismatch: {scripts!r}')

def inventory_mismatch(label, expected, actual):
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f'{label} member inventory mismatch: '
            f'missing={missing[:20]!r}, unexpected={unexpected[:20]!r}'
        )

def project_source_path(project_root, wheel_path):
    if wheel_path.startswith('gridguard/_bundled/'):
        return project_root / wheel_path.removeprefix('gridguard/_bundled/')
    return project_root / 'src' / wheel_path

def source_inventory(project_root):
    git_marker = project_root / '.git'
    if git_marker.exists():
        result = subprocess.run(
            [
                'git',
                '-C',
                str(project_root),
                'ls-files',
                '-z',
                '--cached',
                '--others',
                '--exclude-standard',
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise ValueError(
                'cannot derive the sdist inventory from Git: '
                + result.stderr.decode('utf-8', errors='replace')
            )
        try:
            relative_paths = [
                Path(value)
                for value in result.stdout.decode('utf-8').split('\0')
                if value
            ]
        except UnicodeDecodeError as error:
            raise ValueError('Git returned a non-UTF-8 source path') from error
    else:
        relative_paths = [
            path.relative_to(project_root)
            for path in project_root.rglob('*')
            if path.is_file()
            and path.relative_to(project_root).parts[0] not in EXCLUDED_SOURCE_ROOTS
        ]
    inventory = []
    for relative_path in relative_paths:
        relative_name = relative_path.as_posix()
        if relative_name == 'RELEASE_MANIFEST.json':
            continue
        safe_name(relative_name, 'source')
        source_path = project_root / relative_path
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f'sdist source is not a plain regular file: {relative_name!r}')
        inventory.append(relative_name)
    if len(set(inventory)) != len(inventory):
        raise ValueError('source inventory contains duplicate paths')
    return sorted(inventory)

def validate_record(wheel, names):
    record_name = f'{EXPECTED_ROOT}.dist-info/RECORD'
    try:
        record_text = wheel.read(record_name).decode('utf-8')
        rows = list(csv.reader(io.StringIO(record_text, newline='')))
    except (KeyError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f'wheel RECORD is invalid: {error}') from error
    bindings = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError(f'wheel RECORD row must have three fields: {row!r}')
        member_name, digest, size = row
        safe_name(member_name, 'wheel RECORD')
        if member_name in bindings:
            raise ValueError(f'wheel RECORD contains duplicate member: {member_name!r}')
        bindings[member_name] = (digest, size)
    inventory_mismatch('wheel RECORD', set(names), set(bindings))
    for member_name in names:
        digest, size = bindings[member_name]
        if member_name == record_name:
            if digest or size:
                raise ValueError('wheel RECORD must leave its own digest and size empty')
            continue
        payload = wheel.read(member_name)
        encoded_digest = base64.urlsafe_b64encode(
            hashlib.sha256(payload).digest()
        ).rstrip(b'=').decode('ascii')
        if digest != f'sha256={encoded_digest}' or size != str(len(payload)):
            raise ValueError(f'wheel RECORD binding mismatch: {member_name!r}')

wheel_path = Path(sys.argv[1])
sdist_path = Path(sys.argv[2])
project_root = Path(sys.argv[3]).resolve(strict=True)
expected_wheel_data = set(
    __import__('json').loads(base64.b64decode(sys.argv[4]).decode('utf-8'))
)
embedded_metadata_path = Path(sys.argv[5])
for archive_path in (wheel_path, sdist_path):
    size = archive_path.stat().st_size
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        raise ValueError(f'archive size is invalid: {archive_path} ({size} bytes)')

wheel_records = validate_wheel_container(wheel_path)
with zipfile.ZipFile(wheel_path, 'r') as wheel:
    infos = wheel.infolist()
    if [entry.filename for entry in infos] != [record['name'] for record in wheel_records]:
        raise ValueError('wheel logical inventory differs from its raw central directory')
    for entry, record in zip(infos, wheel_records, strict=True):
        if (
            entry.header_offset != record['local_offset']
            or entry.compress_size != record['compressed_size']
            or entry.file_size != record['uncompressed_size']
            or entry.CRC != record['crc32']
        ):
            raise ValueError(f'wheel logical/raw member metadata mismatch: {entry.filename!r}')
    if any(entry.is_dir() for entry in infos):
        raise ValueError('wheel must contain files only, without directory entries')
    for entry in infos:
        file_type = (entry.external_attr >> 16) & 0o170000
        if file_type not in (0, stat.S_IFREG):
            raise ValueError(f'wheel contains a non-regular member: {entry.filename!r}')
    names = exact_unique_names([entry.filename for entry in infos], 'wheel')
    uncompressed_size = sum(entry.file_size for entry in infos)
    if (
        uncompressed_size > MAX_UNCOMPRESSED_BYTES
        or any(entry.file_size > MAX_MEMBER_BYTES for entry in infos)
    ):
        raise ValueError('wheel uncompressed size budget is invalid')
    source_code = {
        'gridguard/' + path.relative_to(project_root / 'src' / 'gridguard').as_posix()
        for path in (project_root / 'src' / 'gridguard').rglob('*.py')
        if path.is_file() and not path.is_symlink()
    }
    expected_names = source_code | expected_wheel_data | EXPECTED_DIST_INFO_FILES
    inventory_mismatch('wheel', expected_names, set(names))
    for member_name in source_code | expected_wheel_data:
        source_path = project_source_path(project_root, member_name)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError(f'wheel source is unavailable: {source_path}')
        if wheel.read(member_name) != source_path.read_bytes():
            raise ValueError(f'wheel source-byte mismatch: {member_name!r}')
    metadata_name = f'{EXPECTED_ROOT}.dist-info/METADATA'
    wheel_name = f'{EXPECTED_ROOT}.dist-info/WHEEL'
    entry_points_name = f'{EXPECTED_ROOT}.dist-info/entry_points.txt'
    metadata_info = wheel.getinfo(metadata_name)
    if metadata_info.file_size > MAX_METADATA_BYTES:
        raise ValueError('wheel METADATA is too large')
    wheel_metadata_contract = validate_metadata(wheel.read(metadata_info), 'wheel')
    validate_wheel(wheel.read(wheel_name), 'wheel')
    validate_entry_points(wheel.read(entry_points_name), 'wheel')
    validate_record(wheel, names)

tar_payload, raw_sdist_names = validate_sdist_container(sdist_path)
with tarfile.open(fileobj=io.BytesIO(tar_payload), mode='r:') as sdist:
    members = sdist.getmembers()
    if any(not member.isfile() for member in members):
        raise ValueError('sdist must contain regular files only')
    names = exact_unique_names([member.name for member in members], 'sdist')
    if names != raw_sdist_names:
        raise ValueError('sdist logical inventory differs from its raw tar container')
    uncompressed_size = sum(member.size for member in members)
    if (
        uncompressed_size > MAX_UNCOMPRESSED_BYTES
        or any(member.size > MAX_MEMBER_BYTES for member in members)
    ):
        raise ValueError('sdist uncompressed size budget is invalid')
    if any(
        name != EXPECTED_ROOT and not name.startswith(f'{EXPECTED_ROOT}/')
        for name in names
    ):
        raise ValueError('sdist contains a member outside the exact 1.2.0 root')
    metadata_name = f'{EXPECTED_ROOT}/PKG-INFO'
    expected_source_paths = source_inventory(project_root)
    expected_names = {
        *(f'{EXPECTED_ROOT}/{relative_path}' for relative_path in expected_source_paths),
        metadata_name,
    }
    inventory_mismatch('sdist', expected_names, set(names))
    # Hatchling writes selected source files in its deterministic traversal order,
    # then appends generated core metadata. The source inventory/content checks above
    # bind every preceding member without incorrectly assuming a global lexical sort.
    if names[-1] != metadata_name:
        raise ValueError('sdist PKG-INFO is not the canonical final member')
    for relative_path in expected_source_paths:
        member_name = f'{EXPECTED_ROOT}/{relative_path}'
        stream = sdist.extractfile(member_name)
        if stream is None or stream.read() != (project_root / relative_path).read_bytes():
            raise ValueError(f'sdist source-byte mismatch: {relative_path!r}')
    try:
        metadata_info = sdist.getmember(metadata_name)
    except KeyError as error:
        raise ValueError('sdist is missing exact 1.2.0 PKG-INFO') from error
    if not metadata_info.isfile() or metadata_info.size > MAX_METADATA_BYTES:
        raise ValueError('sdist PKG-INFO type or size is invalid')
    stream = sdist.extractfile(metadata_info)
    if stream is None:
        raise ValueError('sdist PKG-INFO cannot be read')
    sdist_metadata_contract = validate_metadata(
        stream.read(MAX_METADATA_BYTES + 1),
        'sdist',
    )

embedded_metadata_contract = validate_metadata(
    read_bounded_regular_file(embedded_metadata_path, 'embedded metadata'),
    'embedded',
)
metadata_contracts = {
    'wheel': wheel_metadata_contract,
    'sdist': sdist_metadata_contract,
    'embedded': embedded_metadata_contract,
}
baseline = metadata_contracts['wheel']
for label, contract in metadata_contracts.items():
    if contract != baseline:
        differing_fields = sorted(
            field for field in baseline if contract[field] != baseline[field]
        )
        raise ValueError(
            f'cross-artifact metadata mismatch for {label}: {differing_fields!r}'
        )
expected_metadata_contract = project_metadata_contract(project_root)
if baseline != expected_metadata_contract:
    differing_fields = sorted(
        field
        for field in expected_metadata_contract
        if baseline[field] != expected_metadata_contract[field]
    )
    raise ValueError(
        'metadata does not match the independently derived pyproject contract: '
        f'{differing_fields!r}'
    )
'@
    $validatorOutput = @()
    $validatorSucceeded = $false
    $validatorExitCode = $null
    $validatorOutput = @(
        $archiveValidator | & uv run --no-sync python - `
            $WheelPath `
            $SdistPath `
            $ProjectRoot `
            $expectedWheelDataBase64 `
            $frozenMetadataPath `
            2>&1
        $validatorSucceeded = $?
        $validatorExitCode = $LASTEXITCODE
    )
    if (-not $validatorSucceeded -or $validatorExitCode -ne 0) {
        throw (
            "Release archive structure is invalid for '$WheelPath' or '$SdistPath': " +
            "$($validatorOutput -join ' ')"
        )
    }
}

function Assert-GridGuardDistributionMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$MetadataRoot
    )

    $metadataValidator = @'
import configparser
from email.parser import BytesParser
from pathlib import Path
import sys

EXPECTED_CONSOLE_SCRIPTS = {
    'gridguard-desktop': 'gridguard.ui.app:main',
    'gridguard-doctor': 'gridguard.cli:doctor_main',
    'gridguard-evaluate': 'gridguard.cli:main',
}
MAX_METADATA_BYTES = 1024 * 1024

def read_limited(path, label):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'{label} must be one regular, unredirected file')
    payload = path.read_bytes()
    if not payload or len(payload) > MAX_METADATA_BYTES:
        raise ValueError(f'{label} size is invalid')
    return payload

def single_header(message, field, label):
    values = message.get_all(field, [])
    if len(values) != 1 or not values[0]:
        raise ValueError(f'{label} must contain exactly one {field} header')
    return values[0]

metadata_root = Path(sys.argv[1])
metadata = BytesParser().parsebytes(read_limited(metadata_root / 'METADATA', 'METADATA'))
actual_identity = {
    'Name': single_header(metadata, 'Name', 'METADATA'),
    'Version': single_header(metadata, 'Version', 'METADATA'),
}
expected_identity = {'Name': 'gridguard-khartoum', 'Version': '1.2.0'}
if actual_identity != expected_identity:
    raise ValueError(f'metadata identity mismatch: {actual_identity!r}')

wheel = BytesParser().parsebytes(read_limited(metadata_root / 'WHEEL', 'WHEEL'))
actual_wheel = {
    'Wheel-Version': single_header(wheel, 'Wheel-Version', 'WHEEL'),
    'Root-Is-Purelib': single_header(wheel, 'Root-Is-Purelib', 'WHEEL'),
    'Tag': wheel.get_all('Tag', []),
}
expected_wheel = {
    'Wheel-Version': '1.0',
    'Root-Is-Purelib': 'true',
    'Tag': ['py3-none-any'],
}
if actual_wheel != expected_wheel:
    raise ValueError(f'wheel contract mismatch: {actual_wheel!r}')

try:
    entry_points = read_limited(
        metadata_root / 'entry_points.txt',
        'entry_points.txt',
    ).decode('utf-8')
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    parser.read_string(entry_points)
except (UnicodeDecodeError, configparser.Error) as error:
    raise ValueError(f'entry_points.txt is invalid: {error}') from error
if parser.sections() != ['console_scripts']:
    raise ValueError(
        f'console-script contract mismatch: sections={parser.sections()!r}'
    )
scripts = dict(parser.items('console_scripts', raw=True))
if scripts != EXPECTED_CONSOLE_SCRIPTS:
    raise ValueError(f'console-script contract mismatch: {scripts!r}')
'@
    $validatorOutput = @()
    $validatorSucceeded = $false
    $validatorExitCode = $null
    $validatorOutput = @(
        & uv run --no-sync python -c $metadataValidator $MetadataRoot 2>&1
        $validatorSucceeded = $?
        $validatorExitCode = $LASTEXITCODE
    )
    if (-not $validatorSucceeded -or $validatorExitCode -ne 0) {
        throw (
            "GridGuard distribution metadata is invalid at '$MetadataRoot': " +
            "$($validatorOutput -join ' ')"
        )
    }
}

function Assert-ReleaseDistributionPolicy {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DistributionPath,
        [int64]$MaxSizeBytes = $script:DefaultMaximumDistributionSizeBytes
    )

    if (-not (Test-Path -LiteralPath $DistributionPath -PathType Container)) {
        throw "Release distribution is unavailable: $DistributionPath"
    }
    if ($MaxSizeBytes -le 0) {
        throw "Release distribution size budget must be positive."
    }

    $unresolvedDistRoot = Split-Path -Parent ([System.IO.Path]::GetFullPath($DistributionPath))
    Assert-ReleaseTreeFileSystemSafety -RootPath $unresolvedDistRoot
    $resolvedDistribution = (Resolve-Path -LiteralPath $DistributionPath).Path
    $distRoot = Split-Path -Parent $resolvedDistribution
    $expectedDistInventory = @(
        ".gitignore|file",
        "GridGuard Khartoum|directory",
        "gridguard_khartoum-1.2.0-py3-none-any.whl|file",
        "gridguard_khartoum-1.2.0.tar.gz|file"
    ) | Sort-Object -CaseSensitive
    $actualDistInventory = @(
        Get-ChildItem -LiteralPath $distRoot -Force |
            ForEach-Object {
                $entryType = if ($_.PSIsContainer) { "directory" } else { "file" }
                "$($_.Name)|$entryType"
            }
    ) | Sort-Object -CaseSensitive
    $distInventoryDifferences = @(
        Compare-Object `
            -ReferenceObject $expectedDistInventory `
            -DifferenceObject $actualDistInventory `
            -CaseSensitive
    )
    if ($distInventoryDifferences.Count -ne 0) {
        $missingDistEntries = @(
            $distInventoryDifferences |
                Where-Object SideIndicator -eq "<=" |
                ForEach-Object InputObject
        )
        $unexpectedDistEntries = @(
            $distInventoryDifferences |
                Where-Object SideIndicator -eq "=>" |
                ForEach-Object InputObject
        )
        throw (
            "Exact release dist inventory mismatch: " +
            "missing=[$($missingDistEntries -join ', ')]; " +
            "unexpected=[$($unexpectedDistEntries -join ', ')]."
        )
    }

    $wheelArchivePath = Join-Path `
        $distRoot `
        "gridguard_khartoum-1.2.0-py3-none-any.whl"
    $sourceArchivePath = Join-Path $distRoot "gridguard_khartoum-1.2.0.tar.gz"
    $projectRoot = Split-Path -Parent $distRoot
    Assert-ReleaseArchiveStructure `
        -WheelPath $wheelArchivePath `
        -SdistPath $sourceArchivePath `
        -ProjectRoot $projectRoot

    $files = @(Get-ChildItem -LiteralPath $resolvedDistribution -Recurse -File -Force)
    [int64]$sizeBytes = ($files | Measure-Object -Property Length -Sum).Sum
    if ($sizeBytes -gt $MaxSizeBytes) {
        $budgetLabel = $MaxSizeBytes.ToString(
            "N0",
            [System.Globalization.CultureInfo]::InvariantCulture
        )
        throw (
            "Release distribution exceeds the $budgetLabel-byte budget: " +
            "actual=$sizeBytes."
        )
    }

    $unexpectedVersionArtifacts = @(
        Get-ChildItem -LiteralPath $distRoot -Recurse -File -Force |
            Where-Object {
                $versionMatch = [regex]::Match(
                    $_.FullName,
                    "(?i)gridguard[_ -]khartoum[-_](?<version>\d+\.\d+\.\d+)"
                )
                return (
                    $versionMatch.Success -and
                    $versionMatch.Groups["version"].Value -ne "1.2.0"
                )
            }
    )
    if ($unexpectedVersionArtifacts.Count -ne 0) {
        $unexpectedVersionPaths = @(
            $unexpectedVersionArtifacts | ForEach-Object {
                $_.FullName.Substring($distRoot.Length).TrimStart("\", "/").Replace("\", "/")
            }
        )
        throw (
            "Unexpected GridGuard release-version artifacts remain: " +
            "$($unexpectedVersionPaths -join ', ')."
        )
    }

    $gridguardMetadataPrefix = "_internal/gridguard_khartoum-1.2.0.dist-info/"
    $expectedMetadataInventory = @(
        "${gridguardMetadataPrefix}METADATA|file",
        "${gridguardMetadataPrefix}WHEEL|file",
        "${gridguardMetadataPrefix}entry_points.txt|file"
    ) | Sort-Object -CaseSensitive
    $metadataRoot = Join-Path `
        $resolvedDistribution `
        "_internal\gridguard_khartoum-1.2.0.dist-info"
    $actualMetadataInventory = @()
    if (Test-Path -LiteralPath $metadataRoot -PathType Container) {
        $actualMetadataInventory = @(
            Get-ChildItem -LiteralPath $metadataRoot -Force |
                ForEach-Object {
                    $entryType = if ($_.PSIsContainer) { "directory" } else { "file" }
                    "${gridguardMetadataPrefix}$($_.Name)|$entryType"
                }
        ) | Sort-Object -CaseSensitive
    }
    $missingMetadata = @(
        $expectedMetadataInventory |
            Where-Object { $actualMetadataInventory -cnotcontains $_ }
    )
    $unexpectedMetadata = @(
        $actualMetadataInventory |
            Where-Object { $expectedMetadataInventory -cnotcontains $_ }
    )
    if ($missingMetadata.Count -ne 0 -or $unexpectedMetadata.Count -ne 0) {
        throw (
            "Exact GridGuard distribution metadata inventory mismatch: " +
            "missing=[$($missingMetadata -join ', ')]; " +
            "unexpected=[$($unexpectedMetadata -join ', ')]."
        )
    }
    Assert-GridGuardDistributionMetadata -MetadataRoot $metadataRoot

    $torchDllFingerprints = @{}
    $nvidiaDlls = [System.Collections.Generic.List[object]]::new()
    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring($resolvedDistribution.Length).
            TrimStart("\", "/").Replace("\", "/")
        if ($relativePath -match "(?i)^_internal/torch/lib/.+\.dll$") {
            $hash = Get-ReleaseSha256 -LiteralPath $file.FullName
            $basename = [System.IO.Path]::GetFileName($relativePath).ToLowerInvariant()
            $fingerprint = "$basename|$hash"
            if (-not $torchDllFingerprints.ContainsKey($fingerprint)) {
                $torchDllFingerprints[$fingerprint] = `
                    [System.Collections.Generic.List[string]]::new()
            }
            $torchDllFingerprints[$fingerprint].Add($relativePath)
        }
        elseif ($relativePath -match "(?i)^_internal/nvidia/[^/]+/bin/.+\.dll$") {
            $nvidiaDlls.Add([pscustomobject]@{
                path = $relativePath
                basename = [System.IO.Path]::GetFileName($relativePath).ToLowerInvariant()
                sha256 = Get-ReleaseSha256 -LiteralPath $file.FullName
            })
        }
    }

    $duplicatePairs = [System.Collections.Generic.List[string]]::new()
    foreach ($nvidiaDll in $nvidiaDlls) {
        $fingerprint = "$($nvidiaDll.basename)|$($nvidiaDll.sha256)"
        if (-not $torchDllFingerprints.ContainsKey($fingerprint)) {
            continue
        }
        foreach ($torchPath in $torchDllFingerprints[$fingerprint]) {
            $duplicatePairs.Add("$torchPath <-> $($nvidiaDll.path)")
        }
    }
    if ($duplicatePairs.Count -ne 0) {
        throw (
            "Byte-identical Torch/NVIDIA CUDA DLL pairs remain: " +
            "$($duplicatePairs -join '; ')."
        )
    }

    $expectedProvenanceSourcePaths = @(Get-RequiredPackagedProvenanceSourcePaths)
    $actualProvenanceSourcePaths = @(
        foreach ($file in $files) {
            $relativePath = $file.FullName.Substring($resolvedDistribution.Length).
                TrimStart("\", "/").Replace("\", "/")
            if (
                $relativePath.StartsWith(
                    "_internal/gridguard/_bundled/src/",
                    [System.StringComparison]::Ordinal
                ) -or
                $relativePath.StartsWith(
                    "_internal/gridguard/_bundled/data/evidence/accepted-sources/",
                    [System.StringComparison]::Ordinal
                )
            ) {
                $relativePath
            }
        }
    ) | Sort-Object -CaseSensitive
    $missingProvenanceSources = @(
        $expectedProvenanceSourcePaths |
            Where-Object { $actualProvenanceSourcePaths -cnotcontains $_ }
    )
    $unexpectedProvenanceSources = @(
        $actualProvenanceSourcePaths |
            Where-Object { $expectedProvenanceSourcePaths -cnotcontains $_ }
    )
    if (
        $missingProvenanceSources.Count -ne 0 -or
        $unexpectedProvenanceSources.Count -ne 0
    ) {
        throw (
            "Exact packaged provenance-source inventory mismatch: " +
            "missing=[$($missingProvenanceSources -join ', ')]; " +
            "unexpected=[$($unexpectedProvenanceSources -join ', ')]."
        )
    }

    $expectedGridGuardPayloadPaths = @(
        @($script:RequiredPackagedRuntimeDataPaths) +
        @($expectedProvenanceSourcePaths) |
            Sort-Object -CaseSensitive -Unique
    )
    $actualGridGuardPayloadPaths = @(
        foreach ($file in $files) {
            $relativePath = $file.FullName.Substring($resolvedDistribution.Length).
                TrimStart("\", "/").Replace("\", "/")
            if (
                $relativePath.StartsWith(
                    "_internal/gridguard/_bundled/",
                    [System.StringComparison]::Ordinal
                ) -or
                $relativePath.StartsWith(
                    "_internal/gridguard/assets/",
                    [System.StringComparison]::Ordinal
                )
            ) {
                $relativePath
            }
        }
    ) | Sort-Object -CaseSensitive
    $missingGridGuardPayload = @(
        $expectedGridGuardPayloadPaths |
            Where-Object { $actualGridGuardPayloadPaths -cnotcontains $_ }
    )
    $unexpectedGridGuardPayload = @(
        $actualGridGuardPayloadPaths |
            Where-Object { $expectedGridGuardPayloadPaths -cnotcontains $_ }
    )
    if (
        $missingGridGuardPayload.Count -ne 0 -or
        $unexpectedGridGuardPayload.Count -ne 0
    ) {
        throw (
            "Exact packaged GridGuard runtime/provenance inventory mismatch: " +
            "missing=[$($missingGridGuardPayload -join ', ')]; " +
            "unexpected=[$($unexpectedGridGuardPayload -join ', ')]."
        )
    }

    return [pscustomobject][ordered]@{
        duplicate_cuda_dll_pair_count = 0
        file_count = $files.Count
        forbidden_editable_metadata_count = 0
        packaged_runtime_data_count = $script:RequiredPackagedRuntimeDataPaths.Count
        size_bytes = $sizeBytes
        status = "PASS"
        unexpected_gridguard_version_artifact_count = 0
    }
}

function Get-ReleaseSourceTreeIdentity {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    $identityReader = @'
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

EXCLUDED_ROOTS = {
    '.codebase-memory',
    '.git',
    '.pytest_cache',
    '.ruff_cache',
    '.uv-cache',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'tmp',
}

def safe_relative_path(value):
    if not value or '\\' in value or any(ord(character) < 32 for character in value):
        raise ValueError(f'invalid tested source path: {value!r}')
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {'', '.', '..'} or ':' in part for part in path.parts)
    ):
        raise ValueError(f'invalid tested source path: {value!r}')
    return path

root = Path(sys.argv[1]).resolve(strict=True)
if not (root / '.git').exists():
    raise ValueError(f'tested source root is not a Git worktree: {root}')
result = subprocess.run(
    [
        'git',
        '-C',
        str(root),
        'ls-files',
        '-z',
        '--cached',
        '--others',
        '--exclude-standard',
    ],
    check=False,
    capture_output=True,
)
if result.returncode != 0:
    raise ValueError(
        'cannot enumerate tested source files: '
        + result.stderr.decode('utf-8', errors='replace')
    )
try:
    raw_paths = [value for value in result.stdout.decode('utf-8').split('\0') if value]
except UnicodeDecodeError as error:
    raise ValueError('Git returned a non-UTF-8 tested source path') from error

entries = []
seen = set()
for raw_path in raw_paths:
    relative = safe_relative_path(raw_path)
    if relative.parts[0] in EXCLUDED_ROOTS:
        continue
    folded = raw_path.casefold()
    if folded in seen:
        raise ValueError(f'case-insensitive duplicate tested source path: {raw_path!r}')
    seen.add(folded)
    source = root / Path(*relative.parts)
    metadata = source.lstat()
    attributes = getattr(metadata, 'st_file_attributes', 0)
    reparse_flag = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    if source.is_symlink() or attributes & reparse_flag or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f'tested source is not a plain regular file: {raw_path!r}')
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    entries.append(
        {
            'path': raw_path,
            'sha256': digest.hexdigest(),
            'size_bytes': metadata.st_size,
        }
    )
entries.sort(key=lambda entry: entry['path'])
canonical = ''.join(
    '{sha256}\t{size_bytes}\t{path}\n'.format(**entry)
    for entry in entries
).encode('utf-8')
payload = {
    'algorithm': 'sha256(canonical-lines-v1)',
    'file_count': len(entries),
    'files': entries,
    'tree_sha256': hashlib.sha256(canonical).hexdigest(),
}
print(json.dumps(payload, ensure_ascii=False, separators=(',', ':'), sort_keys=True))
'@
    $readerOutput = @(
        & uv run --no-sync python -c $identityReader $ProjectRoot 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot derive the tested source identity: $($readerOutput -join ' ')"
    }
    try {
        $identity = ConvertFrom-Json -InputObject ($readerOutput -join "")
    }
    catch {
        throw "Tested source identity reader returned invalid JSON: $($_.Exception.Message)"
    }
    if (
        [string]$identity.algorithm -cne $script:DistributionTreeHashAlgorithm -or
        [int]$identity.file_count -ne @($identity.files).Count -or
        [string]$identity.tree_sha256 -notmatch "^[0-9a-f]{64}$"
    ) {
        throw "Tested source identity has an invalid closed schema."
    }
    foreach ($entry in @($identity.files)) {
        $absolutePath = Resolve-ReleaseRelativePath `
            -RootPath $ProjectRoot `
            -RelativePath ([string]$entry.path) `
            -FieldName "tested_source.files.path"
        [void](Assert-ReleaseFileSystemEntrySafety -LiteralPath $absolutePath)
    }
    return $identity
}

function Assert-ReleaseSourceTreeIdentity {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Identity
    )

    $current = Get-ReleaseSourceTreeIdentity -ProjectRoot $ProjectRoot
    $expectedFiles = @($Identity.files)
    $currentFiles = @($current.files)
    if (
        [string]$Identity.algorithm -cne [string]$current.algorithm -or
        [int]$Identity.file_count -ne [int]$current.file_count -or
        [string]$Identity.tree_sha256 -cne [string]$current.tree_sha256
    ) {
        $expectedByPath = @{}
        foreach ($entry in $expectedFiles) {
            $expectedByPath[[string]$entry.path] = $entry
        }
        $currentByPath = @{}
        foreach ($entry in $currentFiles) {
            $currentByPath[[string]$entry.path] = $entry
        }
        $allPaths = @($expectedByPath.Keys + $currentByPath.Keys | Sort-Object -Unique)
        $firstDifference = "inventory or tree digest"
        foreach ($path in $allPaths) {
            if (
                -not $expectedByPath.ContainsKey($path) -or
                -not $currentByPath.ContainsKey($path) -or
                [string]$expectedByPath[$path].sha256 -cne
                    [string]$currentByPath[$path].sha256 -or
                [int64]$expectedByPath[$path].size_bytes -ne
                    [int64]$currentByPath[$path].size_bytes
            ) {
                $firstDifference = $path
                break
            }
        }
        throw "The live tree no longer matches the exact tested source identity: $firstDifference"
    }
}

function Assert-ReleaseCandidateSourceIdentity {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CandidateRoot,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Identity
    )

    if (-not (Test-Path -LiteralPath $CandidateRoot -PathType Container)) {
        throw "Verified release candidate source is unavailable: $CandidateRoot"
    }
    $resolvedRoot = (Resolve-Path -LiteralPath $CandidateRoot).Path.TrimEnd("\", "/")
    $ignoredGeneratedRoots = @("build", "dist", "tmp")
    $actualFiles = [System.Collections.Generic.List[string]]::new()
    $pending = [System.Collections.Generic.Queue[object]]::new()
    $pending.Enqueue((Assert-ReleaseFileSystemEntrySafety -LiteralPath $resolvedRoot))
    while ($pending.Count -ne 0) {
        $directory = $pending.Dequeue()
        $children = @(Get-ChildItem -LiteralPath $directory.FullName -Force -ErrorAction Stop)
        if ($children.Count -eq 0) {
            throw "Verified candidate source contains an empty directory: $($directory.FullName)"
        }
        foreach ($child in $children) {
            $relativePath = $child.FullName.Substring($resolvedRoot.Length + 1).
                Replace("\", "/")
            $topLevel = $relativePath.Split("/", 2)[0]
            if ($topLevel -in $ignoredGeneratedRoots) {
                continue
            }
            $validated = Assert-ReleaseFileSystemEntrySafety -LiteralPath $child.FullName
            if ($validated.PSIsContainer) {
                $pending.Enqueue($validated)
            }
            else {
                $actualFiles.Add($relativePath)
            }
        }
    }
    $actualFiles.Sort([System.StringComparer]::Ordinal)
    $expectedFiles = @($Identity.files)
    if ($actualFiles.Count -ne $expectedFiles.Count) {
        throw (
            "Verified candidate source file-count mismatch: " +
            "expected=$($expectedFiles.Count), actual=$($actualFiles.Count)."
        )
    }
    for ($index = 0; $index -lt $expectedFiles.Count; $index++) {
        $expected = $expectedFiles[$index]
        if ([string]$expected.path -cne $actualFiles[$index]) {
            throw (
                "Verified candidate source inventory mismatch at index $index`: " +
                "expected=$($expected.path), actual=$($actualFiles[$index])."
            )
        }
        $absolutePath = Join-Path $resolvedRoot $actualFiles[$index].Replace("/", "\")
        $item = Assert-ReleaseFileSystemEntrySafety -LiteralPath $absolutePath
        if (
            [int64]$expected.size_bytes -ne [int64]$item.Length -or
            [string]$expected.sha256 -cne (Get-ReleaseSha256 -LiteralPath $absolutePath)
        ) {
            throw "Verified candidate source-byte mismatch: $($expected.path)"
        }
    }
}

function New-VerifiedReleaseCandidateSource {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Identity
    )

    $resolvedProject = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd("\", "/")
    $resolvedDestination = [System.IO.Path]::GetFullPath($DestinationPath).TrimEnd("\", "/")
    $expectedPrefix = Join-Path $resolvedProject "tmp\release-candidate"
    if (-not $resolvedDestination.StartsWith(
        $expectedPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Verified release candidate must use the project tmp/release-candidate prefix."
    }
    if (Test-Path -LiteralPath $resolvedDestination) {
        throw "Verified release candidate destination must not already exist: $resolvedDestination"
    }
    New-Item -ItemType Directory -Path $resolvedDestination | Out-Null
    try {
        foreach ($entry in @($Identity.files)) {
            $sourcePath = Resolve-ReleaseRelativePath `
                -RootPath $resolvedProject `
                -RelativePath ([string]$entry.path) `
                -FieldName "tested_source.files.path"
            $destination = Join-Path $resolvedDestination ([string]$entry.path).Replace("/", "\")
            $parent = Split-Path -Parent $destination
            if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
                New-Item -ItemType Directory -Path $parent -Force | Out-Null
            }
            Copy-Item -LiteralPath $sourcePath -Destination $destination
        }
        Assert-ReleaseCandidateSourceIdentity `
            -CandidateRoot $resolvedDestination `
            -Identity $Identity
    }
    catch {
        if (
            (Test-Path -LiteralPath $resolvedDestination) -and
            $resolvedDestination.StartsWith(
                $expectedPrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        ) {
            Remove-Item -LiteralPath $resolvedDestination -Recurse -Force
        }
        throw
    }
    return $resolvedDestination
}

function Resolve-ReleaseRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootPath,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [Parameter(Mandatory = $true)]
        [string]$FieldName
    )

    if (
        [string]::IsNullOrWhiteSpace($RelativePath) -or
        [System.IO.Path]::IsPathRooted($RelativePath) -or
        $RelativePath.Contains("\") -or
        $RelativePath.Contains(":") -or
        $RelativePath.Contains("`0")
    ) {
        throw "$FieldName must be a non-empty, forward-slash relative path."
    }

    $resolvedRoot = [System.IO.Path]::GetFullPath($RootPath).TrimEnd("\", "/")
    $candidate = [System.IO.Path]::GetFullPath(
        (Join-Path $resolvedRoot $RelativePath.Replace("/", "\"))
    )
    if (-not $candidate.StartsWith(
        "$resolvedRoot\",
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "$FieldName escapes the release root: $RelativePath"
    }
    return $candidate
}

function Get-FileBinding {
    param(
        [Parameter(Mandatory = $true)]
        [string]$AbsolutePath,
        [Parameter(Mandatory = $true)]
        [string]$ManifestPathValue
    )

    if (-not (Test-Path -LiteralPath $AbsolutePath -PathType Leaf)) {
        throw "Release artifact is missing: $AbsolutePath"
    }
    $item = Assert-ReleaseFileSystemEntrySafety -LiteralPath $AbsolutePath
    return [pscustomobject][ordered]@{
        path = $ManifestPathValue
        sha256 = Get-ReleaseSha256 -LiteralPath $AbsolutePath
        size_bytes = [int64]$item.Length
    }
}

function Assert-FileBinding {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootPath,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Binding,
        [Parameter(Mandatory = $true)]
        [string]$FieldName,
        [string]$ExpectedPath
    )

    if (
        -not [string]::IsNullOrEmpty($ExpectedPath) -and
        $Binding.path -ne $ExpectedPath
    ) {
        throw "$FieldName.path must be $ExpectedPath."
    }
    if ([string]$Binding.sha256 -notmatch "^[0-9a-f]{64}$") {
        throw "$FieldName.sha256 must be a lowercase SHA-256 digest."
    }
    if ($null -eq $Binding.size_bytes -or [int64]$Binding.size_bytes -lt 0) {
        throw "$FieldName.size_bytes must be a non-negative integer."
    }

    $absolutePath = Resolve-ReleaseRelativePath `
        -RootPath $RootPath `
        -RelativePath ([string]$Binding.path) `
        -FieldName "$FieldName.path"
    if (-not (Test-Path -LiteralPath $absolutePath -PathType Leaf)) {
        throw "$FieldName is missing: $($Binding.path)"
    }
    $item = Assert-ReleaseFileSystemEntrySafety -LiteralPath $absolutePath
    if ([int64]$Binding.size_bytes -ne [int64]$item.Length) {
        throw (
            "$FieldName size mismatch for '$($Binding.path)': " +
            "expected=$($Binding.size_bytes), actual=$($item.Length)."
        )
    }
    $actualSha256 = Get-ReleaseSha256 -LiteralPath $absolutePath
    if (-not [string]::Equals(
        [string]$Binding.sha256,
        $actualSha256,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw (
            "$FieldName SHA-256 mismatch for '$($Binding.path)': " +
            "expected=$($Binding.sha256), actual=$actualSha256."
        )
    }
    return $absolutePath
}

function Assert-MachineValidationSourceBinding {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$MachineValidationPayload,
        [Parameter(Mandatory = $true)]
        [string]$ExpectedGitCommit,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ExpectedUvLockBinding
    )

    if ([int]$MachineValidationPayload.schema_version -ne 3) {
        throw "Machine-validation schema_version must be 3."
    }
    if (
        [string]$MachineValidationPayload.source.git_commit -cne
        $ExpectedGitCommit
    ) {
        throw (
            "The machine-validation source Git commit does not match the release " +
            "manifest source Git commit."
        )
    }
    [void](Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $MachineValidationPayload.source.uv_lock `
        -FieldName "machine_validation.source.uv_lock" `
        -ExpectedPath "uv.lock")
    foreach ($field in @("path", "sha256", "size_bytes")) {
        if (
            [string]$MachineValidationPayload.source.uv_lock.$field -cne
            [string]$ExpectedUvLockBinding.$field
        ) {
            throw (
                "The machine-validation source uv.lock binding does not match " +
                "the release manifest source uv.lock binding."
            )
        }
    }
}

function Get-DistributionRelativePaths {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DistributionPath
    )

    if (-not (Test-Path -LiteralPath $DistributionPath -PathType Container)) {
        throw "Presentation distribution is missing: $DistributionPath"
    }

    Assert-ReleaseTreeFileSystemSafety -RootPath $DistributionPath
    $resolvedRoot = (Resolve-Path -LiteralPath $DistributionPath).Path.TrimEnd("\", "/")
    $prefixLength = $resolvedRoot.Length + 1
    $relativePaths = [System.Collections.Generic.List[string]]::new()
    foreach (
        $file in Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Force
    ) {
        if (-not $file.FullName.StartsWith(
            "$resolvedRoot\",
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Distribution enumeration escaped its root: $($file.FullName)"
        }
        $relativePath = $file.FullName.Substring($prefixLength).Replace("\", "/")
        if ($relativePath.Contains("`t") -or $relativePath.Contains("`n") -or
            $relativePath.Contains("`r")) {
            throw "Distribution path cannot be represented canonically: $relativePath"
        }
        $relativePaths.Add($relativePath)
    }
    $relativePaths.Sort([System.StringComparer]::Ordinal)
    return $relativePaths.ToArray()
}

function Get-DistributionTreeSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Files
    )

    $canonical = [System.Text.StringBuilder]::new()
    foreach ($file in $Files) {
        [void]$canonical.Append(
            "$([string]$file.sha256)`t$([int64]$file.size_bytes)`t$([string]$file.path)`n"
        )
    }
    $encoding = [System.Text.UTF8Encoding]::new($false)
    $bytes = $encoding.GetBytes($canonical.ToString())
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha256.Dispose()
    }
}

function Get-DistributionBinding {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DistributionPath,
        [Parameter(Mandatory = $true)]
        [string]$ManifestPathValue
    )

    $files = [System.Collections.Generic.List[object]]::new()
    [int64]$sizeBytes = 0
    foreach ($relativePath in Get-DistributionRelativePaths -DistributionPath $DistributionPath) {
        $absolutePath = Join-Path $DistributionPath $relativePath.Replace("/", "\")
        $item = Get-Item -LiteralPath $absolutePath
        $entry = [ordered]@{
            path = $relativePath
            sha256 = Get-ReleaseSha256 -LiteralPath $absolutePath
            size_bytes = [int64]$item.Length
        }
        $files.Add([pscustomobject]$entry)
        $sizeBytes += [int64]$item.Length
    }
    $fileArray = $files.ToArray()

    return [pscustomobject][ordered]@{
        path = $ManifestPathValue
        file_count = $fileArray.Count
        size_bytes = $sizeBytes
        tree_hash_algorithm = $script:DistributionTreeHashAlgorithm
        tree_sha256 = Get-DistributionTreeSha256 -Files $fileArray
        files = $fileArray
    }
}

function Assert-DistributionBinding {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DistributionPath,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Binding
    )

    if ($Binding.tree_hash_algorithm -ne $script:DistributionTreeHashAlgorithm) {
        throw (
            "Unsupported distribution tree hash algorithm: " +
            "$($Binding.tree_hash_algorithm)"
        )
    }

    $actualPaths = @(Get-DistributionRelativePaths -DistributionPath $DistributionPath)
    $expectedFiles = @($Binding.files)
    if ($Binding.file_count -ne $expectedFiles.Count) {
        throw (
            "Release manifest distribution file_count does not match its inventory: " +
            "declared=$($Binding.file_count), inventory=$($expectedFiles.Count)."
        )
    }
    if ($actualPaths.Count -ne $expectedFiles.Count) {
        throw (
            "Presentation distribution file-count mismatch: " +
            "expected=$($expectedFiles.Count), actual=$($actualPaths.Count)."
        )
    }

    $actualEntries = [System.Collections.Generic.List[object]]::new()
    [int64]$actualSizeBytes = 0
    for ($index = 0; $index -lt $expectedFiles.Count; $index++) {
        $expected = $expectedFiles[$index]
        $actualRelativePath = $actualPaths[$index]
        if (-not [string]::Equals(
            [string]$expected.path,
            $actualRelativePath,
            [System.StringComparison]::Ordinal
        )) {
            throw (
                "Presentation distribution inventory mismatch at index $index`: " +
                "expected=$($expected.path), actual=$actualRelativePath."
            )
        }

        $absolutePath = Join-Path $DistributionPath $actualRelativePath.Replace("/", "\")
        $item = Get-Item -LiteralPath $absolutePath
        if ([int64]$expected.size_bytes -ne [int64]$item.Length) {
            throw (
                "Presentation distribution size mismatch for '$actualRelativePath': " +
                "expected=$($expected.size_bytes), actual=$($item.Length)."
            )
        }
        $actualSha256 = Get-ReleaseSha256 -LiteralPath $absolutePath
        if (-not [string]::Equals(
            [string]$expected.sha256,
            $actualSha256,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw (
                "Presentation distribution SHA-256 mismatch for '$actualRelativePath': " +
                "expected=$($expected.sha256), actual=$actualSha256."
            )
        }

        $actualEntries.Add([pscustomobject][ordered]@{
            path = $actualRelativePath
            sha256 = $actualSha256
            size_bytes = [int64]$item.Length
        })
        $actualSizeBytes += [int64]$item.Length
    }

    if ([int64]$Binding.size_bytes -ne $actualSizeBytes) {
        throw (
            "Presentation distribution byte-count mismatch: " +
            "expected=$($Binding.size_bytes), actual=$actualSizeBytes."
        )
    }
    $actualTreeSha256 = Get-DistributionTreeSha256 -Files $actualEntries.ToArray()
    if (-not [string]::Equals(
        [string]$Binding.tree_sha256,
        $actualTreeSha256,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw (
            "Presentation distribution tree SHA-256 mismatch: " +
            "expected=$($Binding.tree_sha256), actual=$actualTreeSha256."
        )
    }
}

function Get-ValidatedFrozenSmokePayload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportPath
    )

    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        throw "Frozen presentation smoke report is missing: $ReportPath"
    }
    [void](Assert-ReleaseFileSystemEntrySafety -LiteralPath $ReportPath)
    $smokeValidator = @'
import base64
import json
import math
from pathlib import Path
import re
import sys

MAX_SMOKE_BYTES = 1024 * 1024
ROOT_FIELDS = {'gpu', 'live_scenario', 'release_id', 'status', 'window_title'}
GPU_FIELDS = {'cupy_version', 'device_name', 'torch_cuda_version', 'torch_version'}
LIVE_FIELDS = {
    'controllers',
    'device',
    'expected_results_table_rows',
    'finite_core_values',
    'frame_counts',
    'hour_zero',
    'installed_in_live_page',
    'results_table_rows',
    'seed',
    'selected_controller',
    'selected_hour',
    'shortage_level',
    'validated_core_value_count',
}
HOUR_ZERO_FIELDS = {
    'demand_mw',
    'fixed_rotation_served_mw',
    'ppo_served_mw',
    'supply_mw',
}

def duplicate_safe_object(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f'duplicate frozen-smoke key: {key!r}')
        payload[key] = value
    return payload

def reject_constant(token):
    raise ValueError(f'nonfinite frozen-smoke number is not permitted: {token}')

def canonical_bytes(payload):
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + '\n'
    ).encode('utf-8')

def exact_object(value, fields, label):
    if type(value) is not dict:
        raise ValueError(f'{label} must be an object')
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f'{label} field contract mismatch: '
            f'missing={sorted(fields - actual)!r}, unknown={sorted(actual - fields)!r}'
        )
    return value

def exact(value, expected, label):
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f'{label} must be exactly {expected!r}')

def string(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f'{label} must be a nonempty string')
    return value

def positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{label} must be a positive integer')
    return value

def finite_nonnegative(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be a finite non-negative number')
    return float(value)

path = Path(sys.argv[1])
expected_release_id = sys.argv[2]
raw = path.read_bytes()
if not raw or len(raw) > MAX_SMOKE_BYTES:
    raise ValueError(f'frozen-smoke byte size is invalid: {len(raw)}')
payload = json.loads(
    raw.decode('utf-8'),
    object_pairs_hook=duplicate_safe_object,
    parse_constant=reject_constant,
)
if canonical_bytes(payload) != raw:
    raise ValueError('frozen-smoke report is not canonical JSON')

root = exact_object(payload, ROOT_FIELDS, 'frozen-smoke report')
exact(root['release_id'], expected_release_id, 'frozen-smoke release_id')
exact(root['status'], 'PASS', 'frozen-smoke status')
exact(
    root['window_title'],
    'GridGuard Khartoum',
    'frozen-smoke window_title',
)

gpu = exact_object(root['gpu'], GPU_FIELDS, 'frozen-smoke gpu')
device_name = string(gpu['device_name'], 'frozen-smoke gpu.device_name')
if 'NVIDIA' not in device_name.upper():
    raise ValueError('frozen-smoke gpu.device_name must identify an NVIDIA GPU')
exact(gpu['torch_version'], '2.8.0+cu128', 'frozen-smoke gpu.torch_version')
exact(gpu['torch_cuda_version'], '12.8', 'frozen-smoke gpu.torch_cuda_version')
cupy_version = string(gpu['cupy_version'], 'frozen-smoke gpu.cupy_version')
if re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,2}', cupy_version) is None:
    raise ValueError('frozen-smoke gpu.cupy_version is invalid')

live = exact_object(root['live_scenario'], LIVE_FIELDS, 'frozen-smoke live_scenario')
exact(live['shortage_level'], 'moderate', 'frozen-smoke live_scenario.shortage_level')
exact(live['seed'], 50002, 'frozen-smoke live_scenario.seed')
exact(live['device'], 'cuda:0', 'frozen-smoke live_scenario.device')
# Retain validation of the original frozen release report; new launcher requires all three.
legacy_controllers = ['PPO', 'Fixed Rotation']
current_controllers = [*legacy_controllers, 'Optimization']
if live['controllers'] not in (legacy_controllers, current_controllers):
    raise ValueError('frozen-smoke live_scenario.controllers must match a complete release contract')
has_optimization = live['controllers'] == current_controllers
frame_counts = exact_object(
    live['frame_counts'],
    set(live['controllers']),
    'frozen-smoke live_scenario.frame_counts',
)
exact(frame_counts['PPO'], 24, 'frozen-smoke PPO frame count')
exact(frame_counts['Fixed Rotation'], 24, 'frozen-smoke Fixed Rotation frame count')
if has_optimization:
    exact(frame_counts['Optimization'], 24, 'frozen-smoke Optimization frame count')
exact(live['finite_core_values'], True, 'frozen-smoke finite_core_values')
positive_integer(
    live['validated_core_value_count'],
    'frozen-smoke validated_core_value_count',
)
exact(live['installed_in_live_page'], True, 'frozen-smoke installed_in_live_page')
exact(
    live['selected_controller'],
    'Optimization' if has_optimization else 'Fixed Rotation',
    'frozen-smoke selected_controller',
)
exact(live['selected_hour'], 23, 'frozen-smoke selected_hour')
exact(live['expected_results_table_rows'], 3, 'frozen-smoke expected_results_table_rows')
exact(live['results_table_rows'], 3, 'frozen-smoke results_table_rows')

hour_zero = exact_object(
    live['hour_zero'],
    HOUR_ZERO_FIELDS | ({'optimization_served_mw'} if has_optimization else set()),
    'frozen-smoke live_scenario.hour_zero',
)
demand = finite_nonnegative(hour_zero['demand_mw'], 'frozen-smoke hour-zero demand')
supply = finite_nonnegative(hour_zero['supply_mw'], 'frozen-smoke hour-zero supply')
ppo_served = finite_nonnegative(
    hour_zero['ppo_served_mw'],
    'frozen-smoke hour-zero PPO served power',
)
fixed_served = finite_nonnegative(
    hour_zero['fixed_rotation_served_mw'],
    'frozen-smoke hour-zero Fixed Rotation served power',
)
if demand <= 0 or supply <= 0:
    raise ValueError('frozen-smoke hour-zero demand and supply must be positive')
tolerance = 1e-6
if ppo_served > min(demand, supply) + tolerance or fixed_served > min(demand, supply) + tolerance:
    raise ValueError('frozen-smoke hour-zero served power exceeds demand or supply')
if has_optimization:
    optimization_served = finite_nonnegative(
        hour_zero['optimization_served_mw'],
        'frozen-smoke hour-zero Optimization served power',
    )
    if optimization_served > min(demand, supply) + tolerance:
        raise ValueError('frozen-smoke hour-zero Optimization power exceeds demand or supply')

transport = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=False,
    separators=(',', ':'),
    sort_keys=True,
).encode('utf-8')
print(base64.b64encode(transport).decode('ascii'))
'@
    $validatorOutput = @(
        & uv run --no-sync python -c $smokeValidator $ReportPath $script:ReleaseId 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Frozen presentation smoke report is invalid: " +
            "$($validatorOutput -join ' ')"
        )
    }
    try {
        $jsonBytes = [Convert]::FromBase64String(($validatorOutput -join ""))
        $smokePayload = ConvertFrom-Json `
            -InputObject ([Text.Encoding]::UTF8.GetString($jsonBytes))
    }
    catch {
        throw (
            "Frozen presentation smoke validator returned invalid data: " +
            "$($_.Exception.Message)"
        )
    }
    return $smokePayload
}

function Get-StrictReleaseManifestPayload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath
    )

    $manifestValidator = @'
import base64
import json
from pathlib import Path
import re
import sys

MAX_MANIFEST_BYTES = 64 * 1024 * 1024
ROOT_FIELDS = {
    'release_date',
    'release_id',
    'schema_version',
    'source',
    'tracked_artifacts',
    'validation',
    'windows_build',
}
BINDING_FIELDS = {'path', 'sha256', 'size_bytes'}
WINDOWS_BUILD_FIELDS = {
    'build_kind',
    'distribution',
    'distribution_file_count',
    'distribution_size_bytes',
    'distribution_tree_sha256',
    'executable_path',
    'executable_sha256',
    'executable_size_bytes',
    'frozen_smoke',
    'frozen_smoke_report',
    'no_gpu_behavior',
    'sdist',
    'tracked',
    'validated_runtime',
    'wheel',
}

def duplicate_safe_object(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f'duplicate release manifest key: {key!r}')
        payload[key] = value
    return payload

def reject_number(token):
    raise ValueError(f'floating or nonfinite JSON number is not permitted: {token}')

def canonical_bytes(payload):
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + '\n'
    ).encode('utf-8')

def exact_object(value, fields, label):
    if type(value) is not dict:
        raise ValueError(f'{label} must be an object')
    actual = set(value)
    missing = sorted(fields - actual)
    unknown = sorted(actual - fields)
    if missing or unknown:
        raise ValueError(
            f'{label} field contract mismatch: missing={missing!r}, unknown={unknown!r}'
        )
    return value

def array(value, label, nonempty=False):
    if type(value) is not list:
        raise ValueError(f'{label} must be an array')
    if nonempty and not value:
        raise ValueError(f'{label} must be a nonempty array')
    return value

def string(value, label):
    if type(value) is not str or not value:
        raise ValueError(f'{label} must be a nonempty string')
    return value

def boolean(value, label):
    if type(value) is not bool:
        raise ValueError(f'{label} must be a boolean')
    return value

def integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer greater than or equal to {minimum}')
    return value

def sha256(value, label):
    string(value, label)
    if re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{label} must be a lowercase SHA-256 digest')

def binding(value, label):
    value = exact_object(value, BINDING_FIELDS, label)
    string(value['path'], f'{label}.path')
    sha256(value['sha256'], f'{label}.sha256')
    integer(value['size_bytes'], f'{label}.size_bytes')

def validate(payload):
    payload = exact_object(payload, ROOT_FIELDS, 'release manifest')
    if type(payload['schema_version']) is not int or payload['schema_version'] != 3:
        raise ValueError('release manifest schema_version must be the integer 3')
    string(payload['release_id'], 'release manifest release_id')
    string(payload['release_date'], 'release manifest release_date')

    source = exact_object(payload['source'], {'git_commit', 'uv_lock'}, 'source')
    string(source['git_commit'], 'source.git_commit')
    binding(source['uv_lock'], 'source.uv_lock')

    tracked = array(payload['tracked_artifacts'], 'tracked_artifacts', nonempty=True)
    for index, value in enumerate(tracked):
        binding(value, f'tracked_artifacts[{index}]')

    validation = exact_object(
        payload['validation'],
        {'authenticity_boundary', 'machine_validation', 'visual_review'},
        'validation',
    )
    string(validation['authenticity_boundary'], 'validation.authenticity_boundary')
    binding(validation['machine_validation'], 'validation.machine_validation')
    binding(validation['visual_review'], 'validation.visual_review')

    windows = exact_object(payload['windows_build'], WINDOWS_BUILD_FIELDS, 'windows_build')
    if string(windows['build_kind'], 'windows_build.build_kind') != 'PyInstaller one-directory':
        raise ValueError(
            'windows_build.build_kind must be exactly PyInstaller one-directory'
        )
    if boolean(windows['tracked'], 'windows_build.tracked') is not False:
        raise ValueError('windows_build.tracked must be exactly false')
    string(windows['no_gpu_behavior'], 'windows_build.no_gpu_behavior')
    integer(windows['distribution_file_count'], 'windows_build.distribution_file_count')
    integer(windows['distribution_size_bytes'], 'windows_build.distribution_size_bytes')
    sha256(windows['distribution_tree_sha256'], 'windows_build.distribution_tree_sha256')
    string(windows['executable_path'], 'windows_build.executable_path')
    sha256(windows['executable_sha256'], 'windows_build.executable_sha256')
    integer(windows['executable_size_bytes'], 'windows_build.executable_size_bytes')
    string(windows['frozen_smoke'], 'windows_build.frozen_smoke')
    binding(windows['frozen_smoke_report'], 'windows_build.frozen_smoke_report')
    binding(windows['wheel'], 'windows_build.wheel')
    binding(windows['sdist'], 'windows_build.sdist')

    runtime = exact_object(
        windows['validated_runtime'],
        {'cupy', 'cuda', 'gpu', 'pytorch'},
        'windows_build.validated_runtime',
    )
    for field in ('cupy', 'cuda', 'gpu', 'pytorch'):
        string(runtime[field], f'windows_build.validated_runtime.{field}')

    distribution = exact_object(
        windows['distribution'],
        {
            'file_count',
            'files',
            'path',
            'size_bytes',
            'tree_hash_algorithm',
            'tree_sha256',
        },
        'windows_build.distribution',
    )
    string(distribution['path'], 'windows_build.distribution.path')
    integer(distribution['file_count'], 'windows_build.distribution.file_count', minimum=1)
    integer(distribution['size_bytes'], 'windows_build.distribution.size_bytes')
    string(
        distribution['tree_hash_algorithm'],
        'windows_build.distribution.tree_hash_algorithm',
    )
    sha256(distribution['tree_sha256'], 'windows_build.distribution.tree_sha256')
    files = array(distribution['files'], 'windows_build.distribution.files', nonempty=True)
    for index, value in enumerate(files):
        binding(value, f'windows_build.distribution.files[{index}]')

path = Path(sys.argv[1])
raw = path.read_bytes()
if not raw or len(raw) > MAX_MANIFEST_BYTES:
    raise ValueError(f'release manifest byte size is invalid: {len(raw)}')
try:
    text = raw.decode('utf-8')
    payload = json.loads(
        text,
        object_pairs_hook=duplicate_safe_object,
        parse_constant=reject_number,
        parse_float=reject_number,
    )
except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
    raise ValueError(f'release manifest JSON is not canonical: {error}') from error
if raw != canonical_bytes(payload):
    raise ValueError('release manifest JSON is not in canonical UTF-8 form')
validate(payload)
transport = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=False,
    separators=(',', ':'),
    sort_keys=True,
).encode('utf-8')
print(base64.b64encode(transport).decode('ascii'))
'@
    $validatorOutput = @(
        & uv run --no-sync python -c $manifestValidator $ManifestPath 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Release manifest validation failed for '$ManifestPath': " +
            "$($validatorOutput -join ' ')"
        )
    }
    try {
        $jsonBytes = [Convert]::FromBase64String(($validatorOutput -join ""))
        $json = [Text.Encoding]::UTF8.GetString($jsonBytes)
        return ConvertFrom-Json -InputObject $json
    }
    catch {
        throw "Release manifest validator returned invalid data: $($_.Exception.Message)"
    }
}

function Get-ReleaseManifestPayload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath
    )

    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Release manifest is unavailable: $ManifestPath"
    }
    return Get-StrictReleaseManifestPayload -ManifestPath $ManifestPath
}

function Get-ReleaseManifestForUpdate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath
    )

    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Release manifest is unavailable: $ManifestPath"
    }
    $migrationReader = @'
import base64
import json
from pathlib import Path
import sys

def duplicate_safe_object(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f'duplicate release manifest key: {key!r}')
        payload[key] = value
    return payload

def reject_number(token):
    raise ValueError(f'floating or nonfinite JSON number is not permitted: {token}')

raw = Path(sys.argv[1]).read_bytes()
if not raw or len(raw) > 64 * 1024 * 1024:
    raise ValueError(f'release manifest byte size is invalid: {len(raw)}')
payload = json.loads(
    raw.decode('utf-8'),
    object_pairs_hook=duplicate_safe_object,
    parse_constant=reject_number,
    parse_float=reject_number,
)
if type(payload) is not dict:
    raise ValueError('release manifest must be an object')
schema_version = payload.get('schema_version')
if type(schema_version) is not int or schema_version not in (1, 2, 3):
    raise ValueError('release manifest migration schema_version must be integer 1, 2, or 3')
transport = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=False,
    separators=(',', ':'),
    sort_keys=True,
).encode('utf-8')
print(base64.b64encode(transport).decode('ascii'))
'@
    $readerOutput = @(
        & uv run --no-sync python -c $migrationReader $ManifestPath 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Release manifest migration input is invalid: " +
            "$($readerOutput -join ' ')"
        )
    }
    try {
        $jsonBytes = [Convert]::FromBase64String(($readerOutput -join ""))
        $payload = ConvertFrom-Json `
            -InputObject ([Text.Encoding]::UTF8.GetString($jsonBytes))
    }
    catch {
        throw "Release manifest migration reader returned invalid data: $($_.Exception.Message)"
    }
    if ($payload.schema_version -eq 3 -or $payload.schema_version -eq 2) {
        return $payload
    }
    if ($payload.schema_version -ne 1) {
        throw "Only release manifest schema_version 1, 2, or 3 can be refreshed."
    }

    return [pscustomobject][ordered]@{
        schema_version = 3
        release_id = [string]$payload.release_id
        release_date = [string]$payload.release_date
        source = [pscustomobject][ordered]@{
            git_commit = ""
            uv_lock = [pscustomobject]@{}
        }
        tracked_artifacts = @()
        windows_build = $payload.windows_build
        validation = $payload.validation
    }
}

function Assert-ReleaseManifestCore {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath,
        [Parameter(Mandatory = $true)]
        [string]$DistributionPath,
        [Parameter(Mandatory = $true)]
        [string]$ExecutablePath,
        [Parameter(Mandatory = $true)]
        [string]$SmokeReportPath,
        [pscustomobject]$SmokePayload
    )

    $releaseManifest = Get-ReleaseManifestPayload -ManifestPath $ManifestPath
    if (
        [int]$releaseManifest.schema_version -ne 3 -or
        [string]$releaseManifest.release_id -cne $script:ReleaseId -or
        [string]$releaseManifest.release_date -cne $script:ReleaseDate
    ) {
        throw (
            "Release manifest identity must be schema 3, $($script:ReleaseId), " +
            "$($script:ReleaseDate)."
        )
    }
    if ([string]$releaseManifest.source.git_commit -notmatch "^[0-9a-f]{40}$") {
        throw "Release manifest source.git_commit must be a lowercase Git commit ID."
    }
    if (
        [string]$releaseManifest.windows_build.build_kind -cne
        $script:ExpectedBuildKind -or
        [bool]$releaseManifest.windows_build.tracked
    ) {
        throw (
            "Release manifest Windows build semantics must be " +
            "$($script:ExpectedBuildKind) with tracked=false."
        )
    }
    [void](Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $releaseManifest.source.uv_lock `
        -FieldName "source.uv_lock" `
        -ExpectedPath "uv.lock")
    $machineValidationPath = Join-Path `
        $ProjectRoot `
        "docs/release-machine-validation.json"
    try {
        $machineValidationPayload = Get-Content `
            -Raw `
            -LiteralPath $machineValidationPath | ConvertFrom-Json
    }
    catch {
        throw "Machine-validation record is invalid JSON: $($_.Exception.Message)"
    }
    Assert-MachineValidationSourceBinding `
        -ProjectRoot $ProjectRoot `
        -MachineValidationPayload $machineValidationPayload `
        -ExpectedGitCommit ([string]$releaseManifest.source.git_commit) `
        -ExpectedUvLockBinding $releaseManifest.source.uv_lock

    $trackedArtifacts = @($releaseManifest.tracked_artifacts)
    $actualTrackedPaths = @($trackedArtifacts | ForEach-Object { [string]$_.path })
    if ($actualTrackedPaths.Count -ne $script:RequiredTrackedArtifactPaths.Count) {
        throw (
            "Release manifest tracked artifact count mismatch: " +
            "expected=$($script:RequiredTrackedArtifactPaths.Count), " +
            "actual=$($actualTrackedPaths.Count)."
        )
    }
    for ($index = 0; $index -lt $script:RequiredTrackedArtifactPaths.Count; $index++) {
        $expectedPath = $script:RequiredTrackedArtifactPaths[$index]
        if ($actualTrackedPaths[$index] -ne $expectedPath) {
            throw (
                "Release manifest tracked artifact mismatch at index $index`: " +
                "expected=$expectedPath, actual=$($actualTrackedPaths[$index])."
            )
        }
        [void](Assert-FileBinding `
            -RootPath $ProjectRoot `
            -Binding $trackedArtifacts[$index] `
            -FieldName "tracked_artifacts[$index]" `
            -ExpectedPath $expectedPath)
    }

    $validationFields = @(
        $releaseManifest.validation.PSObject.Properties.Name | Sort-Object
    )
    $expectedValidationFields = @(
        "authenticity_boundary",
        "machine_validation",
        "visual_review"
    )
    if (
        $validationFields.Count -ne $expectedValidationFields.Count -or
        (Compare-Object $validationFields $expectedValidationFields).Count -ne 0
    ) {
        throw "Release manifest validation must contain only strict record bindings."
    }
    if (
        [string]$releaseManifest.validation.authenticity_boundary -cne
        $script:UnsignedTrustBoundary
    ) {
        throw "Release manifest must disclose the unsigned Git-anchor trust boundary."
    }
    [void](Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $releaseManifest.validation.machine_validation `
        -FieldName "validation.machine_validation" `
        -ExpectedPath "docs/release-machine-validation.json")
    [void](Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $releaseManifest.validation.visual_review `
        -FieldName "validation.visual_review" `
        -ExpectedPath "docs/release-visual-review.json")

    $expectedDistributionPath = "dist/GridGuard Khartoum"
    if ($releaseManifest.windows_build.distribution.path -ne $expectedDistributionPath) {
        throw "Release manifest distribution.path must be $expectedDistributionPath."
    }
    Assert-DistributionBinding `
        -DistributionPath $DistributionPath `
        -Binding $releaseManifest.windows_build.distribution
    if (
        $releaseManifest.windows_build.distribution_file_count -ne
        $releaseManifest.windows_build.distribution.file_count -or
        $releaseManifest.windows_build.distribution_size_bytes -ne
        $releaseManifest.windows_build.distribution.size_bytes -or
        $releaseManifest.windows_build.distribution_tree_sha256 -ne
        $releaseManifest.windows_build.distribution.tree_sha256
    ) {
        throw "Release manifest distribution summary does not match its file inventory."
    }

    $expectedWheelPath = "dist/gridguard_khartoum-1.2.0-py3-none-any.whl"
    $expectedSdistPath = "dist/gridguard_khartoum-1.2.0.tar.gz"
    $wheelPath = Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $releaseManifest.windows_build.wheel `
        -FieldName "windows_build.wheel" `
        -ExpectedPath $expectedWheelPath
    $sdistPath = Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $releaseManifest.windows_build.sdist `
        -FieldName "windows_build.sdist" `
        -ExpectedPath $expectedSdistPath
    Assert-ReleaseArchiveStructure `
        -WheelPath $wheelPath `
        -SdistPath $sdistPath `
        -ProjectRoot $ProjectRoot

    $expectedExecutablePath = "dist/GridGuard Khartoum/GridGuard Khartoum.exe"
    $executableBinding = [pscustomobject]@{
        path = $releaseManifest.windows_build.executable_path
        sha256 = $releaseManifest.windows_build.executable_sha256
        size_bytes = $releaseManifest.windows_build.executable_size_bytes
    }
    [void](Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $executableBinding `
        -FieldName "windows_build.executable" `
        -ExpectedPath $expectedExecutablePath)
    if (-not [string]::Equals(
        (Resolve-Path -LiteralPath $ExecutablePath).Path,
        (Resolve-ReleaseRelativePath `
            -RootPath $ProjectRoot `
            -RelativePath $expectedExecutablePath `
            -FieldName "windows_build.executable_path"),
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Executable verification target does not match the release manifest."
    }

    [void](Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $releaseManifest.windows_build.frozen_smoke_report `
        -FieldName "windows_build.frozen_smoke_report" `
        -ExpectedPath "build/gridguard/frozen-smoke.json")
    if ($null -eq $SmokePayload) {
        $SmokePayload = Get-ValidatedFrozenSmokePayload -ReportPath $SmokeReportPath
    }
    if (
        $releaseManifest.windows_build.frozen_smoke -ne $SmokePayload.status -or
        $releaseManifest.windows_build.no_gpu_behavior -ne "FAIL_CLOSED" -or
        $releaseManifest.windows_build.validated_runtime.gpu -ne
        $SmokePayload.gpu.device_name -or
        $releaseManifest.windows_build.validated_runtime.pytorch -ne
        $SmokePayload.gpu.torch_version -or
        $releaseManifest.windows_build.validated_runtime.cuda -ne
        $SmokePayload.gpu.torch_cuda_version -or
        $releaseManifest.windows_build.validated_runtime.cupy -ne
        $SmokePayload.gpu.cupy_version
    ) {
        throw "Release manifest runtime or frozen-smoke facts do not match the smoke report."
    }
    return $releaseManifest
}

function Assert-ReleaseSourceSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath,
        [switch]$RequireCleanWorktree
    )

    $releaseManifest = Get-ReleaseManifestPayload -ManifestPath $ManifestPath
    $gitCommit = [string]$releaseManifest.source.git_commit
    if ($gitCommit -notmatch "^[0-9a-f]{40}$") {
        throw "Release manifest source.git_commit must be a lowercase Git commit ID."
    }

    & git -C $ProjectRoot cat-file -e "$gitCommit^{commit}"
    if ($LASTEXITCODE -ne 0) {
        throw "Release source snapshot commit does not exist in this repository: $gitCommit"
    }
    & git -C $ProjectRoot merge-base --is-ancestor $gitCommit HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Release source snapshot is not an ancestor of the current HEAD: $gitCommit"
    }

    $changeReader = @'
import json
import subprocess
import sys

root, git_commit = sys.argv[1:3]

def run(arguments, label):
    result = subprocess.run(arguments, check=False, capture_output=True)
    if result.returncode != 0:
        raise ValueError(
            f'{label} failed: ' + result.stderr.decode('utf-8', errors='replace')
        )
    return result.stdout

def decode_path(value, label):
    try:
        path = value.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ValueError(f'{label} returned a non-UTF-8 path') from error
    if not path or '\0' in path:
        raise ValueError(f'{label} returned an invalid path')
    return path

diff_raw = run(
    [
        'git',
        '-C',
        root,
        'diff',
        '--no-ext-diff',
        '--no-renames',
        '--name-status',
        '-z',
        f'{git_commit}..HEAD',
        '--',
    ],
    'release source commit comparison',
)
diff_tokens = diff_raw.split(b'\0')
if diff_tokens[-1:] == [b'']:
    diff_tokens.pop()
if len(diff_tokens) % 2:
    raise ValueError('release source commit comparison returned truncated name-status data')
committed = []
for index in range(0, len(diff_tokens), 2):
    try:
        status = diff_tokens[index].decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError('release source commit comparison returned an invalid status') from error
    if status not in {'A', 'B', 'D', 'M', 'T', 'U', 'X'}:
        raise ValueError(f'release source commit comparison returned status {status!r}')
    committed.append(
        {
            'path': decode_path(diff_tokens[index + 1], 'release source commit comparison'),
            'status': status,
        }
    )

status_raw = run(
    [
        'git',
        '-C',
        root,
        'status',
        '--porcelain=v1',
        '-z',
        '--untracked-files=all',
        '--no-renames',
    ],
    'release source worktree inspection',
)
worktree = []
for record in status_raw.split(b'\0'):
    if not record:
        continue
    if len(record) < 4 or record[2:3] != b' ':
        raise ValueError('release source worktree inspection returned malformed status data')
    try:
        status = record[:2].decode('ascii')
    except UnicodeDecodeError as error:
        raise ValueError('release source worktree inspection returned an invalid status') from error
    worktree.append(
        {
            'path': decode_path(record[3:], 'release source worktree inspection'),
            'status': status,
        }
    )

print(
    json.dumps(
        {'committed': committed, 'worktree': worktree},
        ensure_ascii=False,
        separators=(',', ':'),
        sort_keys=True,
    )
)
'@
    $changeOutput = @(
        & uv run --no-sync python -c $changeReader $ProjectRoot $gitCommit 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Cannot inspect release source changes: $($changeOutput -join ' ')"
    }
    try {
        $changes = ConvertFrom-Json -InputObject ($changeOutput -join "")
    }
    catch {
        throw "Release source change reader returned invalid JSON: $($_.Exception.Message)"
    }
    $committedChanges = @($changes.committed)
    $worktreeChanges = @($changes.worktree)
    if ($RequireCleanWorktree -and $worktreeChanges.Count -ne 0) {
        throw (
            "Final release handoff requires a clean worktree; commit or remove: " +
            "$(($worktreeChanges | ForEach-Object { "$($_.status) $($_.path)" }) -join '; ')"
        )
    }

    # The visual record and technical ledger must already be frozen in the tested source
    # commit. Publication is proven later from remote Git refs and the final handoff; no
    # tracked or untracked publication-attestation file is permitted here.
    $unexpectedChanges = [System.Collections.Generic.List[string]]::new()
    foreach ($committedChange in $committedChanges) {
        $committedPath = [string]$committedChange.path
        if ($committedPath -notin $script:AllowedPostSourceAttestationPaths) {
            $unexpectedChanges.Add("$($committedChange.status) $committedPath")
        }
    }
    foreach ($worktreeChange in $worktreeChanges) {
        $statusPath = [string]$worktreeChange.path
        if ($statusPath -notin $script:AllowedPostSourceAttestationPaths) {
            $unexpectedChanges.Add("$($worktreeChange.status) $statusPath")
        }
    }
    if ($unexpectedChanges.Count -ne 0) {
        throw (
            "Release source snapshot is stale; only the machine-validation and " +
            "release-manifest attestation files may differ: " +
            "$($unexpectedChanges -join '; ')"
        )
    }

    return $gitCommit
}

function Assert-ReleaseManifestBinding {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath,
        [Parameter(Mandatory = $true)]
        [string]$DistributionPath,
        [Parameter(Mandatory = $true)]
        [string]$ExecutablePath,
        [Parameter(Mandatory = $true)]
        [string]$SmokeReportPath,
        [pscustomobject]$SmokePayload
    )

    return Assert-ReleaseManifestCore @PSBoundParameters
}

function Assert-FinalReleaseReadiness {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ReleaseManifest
    )

    if (
        [int]$ReleaseManifest.schema_version -ne 3 -or
        [string]$ReleaseManifest.release_id -cne $script:ReleaseId -or
        [string]$ReleaseManifest.release_date -cne $script:ReleaseDate -or
        [string]$ReleaseManifest.windows_build.build_kind -cne
            $script:ExpectedBuildKind -or
        [bool]$ReleaseManifest.windows_build.tracked
    ) {
        throw "Final readiness requires the exact schema-3 GridGuard Khartoum v1.2.0 contract."
    }

    $machinePath = Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $ReleaseManifest.validation.machine_validation `
        -FieldName "validation.machine_validation" `
        -ExpectedPath "docs/release-machine-validation.json"
    try {
        $machine = Get-Content -Raw -LiteralPath $machinePath | ConvertFrom-Json
    }
    catch {
        throw "Final-readiness machine-validation record is invalid JSON: $($_.Exception.Message)"
    }
    if (
        [string]$machine.schema_name -cne "gridguard.release_machine_validation" -or
        [int]$machine.schema_version -ne 3 -or
        [string]$machine.release.version -cne "1.2.0" -or
        [string]$machine.release.release_id -cne $script:ReleaseId -or
        [string]$machine.source.git_commit -cne
            [string]$ReleaseManifest.source.git_commit
    ) {
        throw (
            "Final-readiness machine-validation identity does not match the " +
            "schema-3 v1.2.0 release manifest."
        )
    }

    $ledgerPath = Join-Path $ProjectRoot "docs\DEVILS_ADVOCATE_AUDIT.md"
    [void](Assert-ReleaseFileSystemEntrySafety -LiteralPath $ledgerPath)
    $statuses = @{}
    foreach ($line in @(Get-Content -LiteralPath $ledgerPath)) {
        if (
            $line -match (
                '^\|\s*(?<finding>A(?:0[1-9]|1[0-9]|2[0-4]))\s*\|' +
                '(?:[^|]*\|){4}\s*(?<status>[^|]+?)\s*\|'
            )
        ) {
            $finding = [string]$Matches["finding"]
            if ($statuses.ContainsKey($finding)) {
                throw "Final-readiness audit ledger contains duplicate finding $finding."
            }
            $statuses[$finding] = ([string]$Matches["status"]).Trim()
        }
    }
    if ($statuses.Count -ne 24) {
        throw (
            "Final-readiness audit ledger must contain exactly A01-A24; " +
            "found $($statuses.Count) rows."
        )
    }
    for ($number = 1; $number -le 24; $number++) {
        $finding = "A$($number.ToString('00'))"
        $expectedStatus = if ($finding -in @("A01", "A21")) {
            "Post-freeze handoff"
        }
        elseif ($finding -eq "A02") {
            "Post-freeze attestation"
        }
        else {
            "Closed"
        }
        if (
            -not $statuses.ContainsKey($finding) -or
            [string]$statuses[$finding] -cne $expectedStatus
        ) {
            throw (
                "Final-readiness audit ledger status mismatch for $finding`: " +
                "expected=$expectedStatus, actual=$($statuses[$finding])."
            )
        }
    }
    return $ReleaseManifest
}

function Set-ReleaseObjectProperty {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$InputObject,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [AllowNull()]
        [object]$Value
    )

    $InputObject | Add-Member `
        -MemberType NoteProperty `
        -Name $Name `
        -Value $Value `
        -Force
}

function Update-ReleaseManifestBindings {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath,
        [Parameter(Mandatory = $true)]
        [string]$DistributionPath,
        [Parameter(Mandatory = $true)]
        [string]$ExecutablePath,
        [Parameter(Mandatory = $true)]
        [string]$SmokeReportPath,
        [string]$GitCommit
    )

    $releaseManifest = Get-ReleaseManifestForUpdate -ManifestPath $ManifestPath
    $machineValidationPath = Join-Path `
        $ProjectRoot `
        "docs/release-machine-validation.json"
    try {
        $machineValidationPayload = Get-Content `
            -Raw `
            -LiteralPath $machineValidationPath | ConvertFrom-Json
    }
    catch {
        throw "Machine-validation record is invalid JSON: $($_.Exception.Message)"
    }
    if ([string]::IsNullOrWhiteSpace($GitCommit)) {
        $statusLines = @(& git -C $ProjectRoot status --porcelain --untracked-files=all)
        if ($LASTEXITCODE -ne 0) {
            throw "Cannot inspect the Git worktree for release provenance."
        }
        $unexpectedChanges = [System.Collections.Generic.List[string]]::new()
        foreach ($statusLine in $statusLines) {
            $statusPath = if ($statusLine.Length -ge 4) {
                $statusLine.Substring(3)
            }
            else {
                $statusLine
            }
            if ($statusPath -ne "RELEASE_MANIFEST.json") {
                $unexpectedChanges.Add($statusLine)
            }
        }
        if ($unexpectedChanges.Count -ne 0) {
            throw (
                "Release provenance requires a clean source worktree before freezing. " +
                "Commit or revert: $($unexpectedChanges -join '; ')"
            )
        }

        # The machine-validation record names the exact source commit whose tests it
        # summarizes. The validation-record and manifest commits form a later
        # attestation envelope and therefore cannot self-reference their own hashes.
        $GitCommit = [string]$machineValidationPayload.source.git_commit
    }
    if ($GitCommit -notmatch "^[0-9a-f]{40}$") {
        throw "Release provenance Git commit must be a lowercase 40-character ID."
    }

    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest `
        -Name "schema_version" `
        -Value 3
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest `
        -Name "release_id" `
        -Value $script:ReleaseId
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest `
        -Name "release_date" `
        -Value $script:ReleaseDate
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.source `
        -Name "git_commit" `
        -Value $GitCommit
    $uvLockPath = Join-Path $ProjectRoot "uv.lock"
    $uvLockBinding = Get-FileBinding `
        -AbsolutePath $uvLockPath `
        -ManifestPathValue "uv.lock"
    Assert-MachineValidationSourceBinding `
        -ProjectRoot $ProjectRoot `
        -MachineValidationPayload $machineValidationPayload `
        -ExpectedGitCommit $GitCommit `
        -ExpectedUvLockBinding $uvLockBinding
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.source `
        -Name "uv_lock" `
        -Value $uvLockBinding

    $trackedArtifacts = [System.Collections.Generic.List[object]]::new()
    foreach ($relativePath in $script:RequiredTrackedArtifactPaths) {
        $absolutePath = Resolve-ReleaseRelativePath `
            -RootPath $ProjectRoot `
            -RelativePath $relativePath `
            -FieldName "tracked_artifact.path"
        $trackedArtifacts.Add(
            (Get-FileBinding -AbsolutePath $absolutePath -ManifestPathValue $relativePath)
        )
    }
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest `
        -Name "tracked_artifacts" `
        -Value $trackedArtifacts.ToArray()

    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest `
        -Name "validation" `
        -Value ([pscustomobject][ordered]@{
            authenticity_boundary = $script:UnsignedTrustBoundary
            machine_validation = Get-FileBinding `
                -AbsolutePath (Join-Path $ProjectRoot "docs/release-machine-validation.json") `
                -ManifestPathValue "docs/release-machine-validation.json"
            visual_review = Get-FileBinding `
                -AbsolutePath (Join-Path $ProjectRoot "docs/release-visual-review.json") `
                -ManifestPathValue "docs/release-visual-review.json"
        })

    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "build_kind" `
        -Value $script:ExpectedBuildKind
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "tracked" `
        -Value $false

    $distributionBinding = Get-DistributionBinding `
        -DistributionPath $DistributionPath `
        -ManifestPathValue "dist/GridGuard Khartoum"
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "distribution" `
        -Value $distributionBinding
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "distribution_file_count" `
        -Value ([int]$distributionBinding.file_count)
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "distribution_size_bytes" `
        -Value ([int64]$distributionBinding.size_bytes)
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "distribution_tree_sha256" `
        -Value ([string]$distributionBinding.tree_sha256)

    $wheelPath = Join-Path `
        $ProjectRoot `
        "dist\gridguard_khartoum-1.2.0-py3-none-any.whl"
    $sdistPath = Join-Path $ProjectRoot "dist\gridguard_khartoum-1.2.0.tar.gz"
    Assert-ReleaseArchiveStructure `
        -WheelPath $wheelPath `
        -SdistPath $sdistPath `
        -ProjectRoot $ProjectRoot
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "wheel" `
        -Value (Get-FileBinding `
            -AbsolutePath $wheelPath `
            -ManifestPathValue "dist/gridguard_khartoum-1.2.0-py3-none-any.whl")
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "sdist" `
        -Value (Get-FileBinding `
            -AbsolutePath $sdistPath `
            -ManifestPathValue "dist/gridguard_khartoum-1.2.0.tar.gz")

    $executableBinding = Get-FileBinding `
        -AbsolutePath $ExecutablePath `
        -ManifestPathValue "dist/GridGuard Khartoum/GridGuard Khartoum.exe"
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "executable_path" `
        -Value $executableBinding.path
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "executable_sha256" `
        -Value $executableBinding.sha256
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "executable_size_bytes" `
        -Value ([int64]$executableBinding.size_bytes)

    $smokePayload = Get-ValidatedFrozenSmokePayload -ReportPath $SmokeReportPath
    $smokeBinding = Get-FileBinding `
        -AbsolutePath $SmokeReportPath `
        -ManifestPathValue "build/gridguard/frozen-smoke.json"
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "frozen_smoke_report" `
        -Value $smokeBinding
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "frozen_smoke" `
        -Value ([string]$smokePayload.status)
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.windows_build `
        -Name "validated_runtime" `
        -Value ([pscustomobject][ordered]@{
            cupy = [string]$smokePayload.gpu.cupy_version
            cuda = [string]$smokePayload.gpu.torch_cuda_version
            gpu = [string]$smokePayload.gpu.device_name
            pytorch = [string]$smokePayload.gpu.torch_version
        })

    $serialized = $releaseManifest | ConvertTo-Json -Depth 12
    $encoding = [System.Text.UTF8Encoding]::new($false)
    $temporaryPath = "$ManifestPath.tmp"
    try {
        [System.IO.File]::WriteAllText($temporaryPath, "$serialized`n", $encoding)
        $canonicalizer = @'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding='utf-8'))
canonical = (
    json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    + '\n'
).encode('utf-8')
path.write_bytes(canonical)
'@
        $canonicalizerOutput = @(
            & uv run --no-sync python -c $canonicalizer $temporaryPath 2>&1
        )
        if ($LASTEXITCODE -ne 0) {
            throw (
                "Cannot serialize the canonical release manifest: " +
                "$($canonicalizerOutput -join ' ')"
            )
        }
        [void](Get-StrictReleaseManifestPayload -ManifestPath $temporaryPath)
        Move-Item -LiteralPath $temporaryPath -Destination $ManifestPath -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

function Write-LaunchDiagnostic {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DiagnosticPath,
        [Parameter(Mandatory = $true)]
        [string]$Stage,
        [Parameter(Mandatory = $true)]
        [string]$Status,
        [Parameter(Mandatory = $true)]
        [string]$ExecutablePath,
        [string]$Message = "",
        [Nullable[int]]$ExitCode = $null
    )

    $diagnosticDirectory = Split-Path -Parent $DiagnosticPath
    New-Item -ItemType Directory -Path $diagnosticDirectory -Force | Out-Null
    $payload = [ordered]@{
        schema_version = 1
        timestamp_utc = [DateTime]::UtcNow.ToString("o")
        stage = $Stage
        status = $Status
        message = $Message
        exit_code = if ($null -eq $ExitCode) { $null } else { [int]$ExitCode }
        executable_path = $ExecutablePath
        working_directory = if ([string]::IsNullOrEmpty($ExecutablePath)) {
            ""
        }
        else {
            Split-Path -Parent $ExecutablePath
        }
        powershell_version = $PSVersionTable.PSVersion.ToString()
        process_architecture = [System.Environment]::GetEnvironmentVariable(
            "PROCESSOR_ARCHITECTURE"
        )
        os_version = [System.Environment]::OSVersion.VersionString
        remediation = @(
            "Run scripts/build-presentation.ps1 -VerifyOnly for the first failing binding.",
            "Keep the complete dist/GridGuard Khartoum directory together.",
            "For an early non-zero exit, verify the NVIDIA driver and Microsoft Visual C++ runtime."
        )
    }
    $serialized = $payload | ConvertTo-Json -Depth 5
    [System.IO.File]::WriteAllText(
        $DiagnosticPath,
        "$serialized`n",
        [System.Text.UTF8Encoding]::new($false)
    )
    return $DiagnosticPath
}
