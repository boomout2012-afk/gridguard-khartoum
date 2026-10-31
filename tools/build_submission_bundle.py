"""Prepare and seal a portable submission without deleting existing project files.

Python standard library only. Run prepare first, perform the relocated checks, then
refresh any changed allowlisted files and run finalize. A sealed folder is immutable:
use a new destination to create another submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

SOURCE_DIRECTORIES = (
    "src",
    "tests",
    "data",
    "docs",
    "tools",
    "scripts",
    "packaging",
    "tasks",
    ".github",
)
SOURCE_ROOT_FILES = (
    "README.md",
    "SUBMISSION_README.md",
    "SPEC.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "pyproject.toml",
    "uv.lock",
    "hatch_build.py",
    ".python-version",
    ".gitignore",
    ".gitattributes",
    "RELEASE_MANIFEST.json",
    "requirements-document-build.txt",
)
DOCUMENTS = (
    "output/pdf/gridguard-khartoum-gui-pages.pdf",
    "output/pdf/updated/gridguard-khartoum-thesis.pdf",
    "output/pdf/updated/gridguard-khartoum-project-presentation.pdf",
    "output/presentation/updated/gridguard-khartoum-defense.pptx",
)
SOURCE_OUTPUTS = DOCUMENTS + (
    "output/pdf/gridguard-khartoum-gui-pages.build.json",
    "output/pdf/updated/gridguard-khartoum-thesis.build.json",
    "output/pdf/updated/gridguard-khartoum-thesis.source.md",
    "output/pdf/gridguard-khartoum-final-report.pdf",
    "output/presentation/gridguard-khartoum-defense.pptx",
)
MANIFEST = "SUBMISSION_MANIFEST.json"
CHECKSUM = "SUBMISSION_MANIFEST.sha256"
IGNORED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cache",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        ".previous",
        "logs",
        "log",
        ".idea",
        ".vscode",
    }
)
IGNORED_SUFFIXES = frozenset({".pyc", ".pyo", ".log", ".tmp", ".bak", ".lnk"})
SECRET_SUFFIXES = frozenset({".pem", ".key", ".pfx", ".p12", ".jks", ".keystore"})
SECRET_NAMES = frozenset(
    {
        "credentials.json",
        "credentials.toml",
        "credentials.yaml",
        "credentials.yml",
        "secrets.json",
        "secrets.toml",
        "secrets.yaml",
        "secrets.yml",
        "id_rsa",
        "id_ed25519",
        ".npmrc",
        ".pypirc",
        ".netrc",
        "thumbs.db",
        ".ds_store",
    }
)

START_HERE = """# GridGuard Khartoum — submission

University of Khartoum · Faculty of Mathematical Sciences and Informatics

Open the final documents:

- [Thesis — 35 pages](Documents/gridguard-khartoum-thesis.pdf)
- [Presentation — 24 slides](Documents/gridguard-khartoum-project-presentation.pdf)
- [Editable PowerPoint](Documents/gridguard-khartoum-defense.pptx)
- [GUI collection — 19 pages](Documents/gridguard-khartoum-gui-pages.pdf)

Double-click **Run GridGuard.cmd** to launch. Keep the entire App folder together.
The packaged app runs offline without Python. It requires 64-bit Windows and a
compatible NVIDIA GPU and driver for the bundled CUDA 12.8 runtime. CPU-only
computers can read the documents and inspect the source.

Run **Verify Submission.ps1** in Windows PowerShell to check every file's SHA-256,
size and inventory. No Python is required. The ZIP's adjacent SHA-256 file checks
the archive itself. Integrity checks detect changes; they are not a digital signature.

[Source/SUBMISSION_README.md](Source/SUBMISSION_README.md) explains setup, testing,
evidence reproduction, document editing and rebuilding. Verification contains the
final review reports. Source contains clean code, tests, locked dependencies, models,
evidence, document sources and assets. Historical test fixtures are identified in the
source guide and are not the final submission documents.

The live scenario uses the selected PPO actor (seed 8111, checkpoint 2500); Results
shows the five-run study, so their numbers differ intentionally. The documents retain
the optimizer efficiency tradeoff. All experiments use synthetic decision-support data.
"""

LAUNCHER = r"""@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0App\GridGuard Khartoum\GridGuard Khartoum.exe" (
  echo The application folder is missing. Extract the complete submission first.
  pause
  exit /b 1
)
set "gridguardApp=%~dp0App\GridGuard Khartoum"
start "GridGuard Khartoum" /D "%gridguardApp%" "%gridguardApp%\GridGuard Khartoum.exe"
"""

POWERSHELL_VERIFIER = r"""[CmdletBinding()]
param([switch]$Json)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
try {
    $bundleRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\', '/')
    $prefix = $bundleRoot + [IO.Path]::DirectorySeparatorChar
    function Assert-RegularPath([string]$Path, [bool]$Directory) {
        $item = Get-Item -LiteralPath $Path -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "Links/reparse points are forbidden: $Path"
        }
        if ($item.PSObject.Properties['LinkType'] -and $item.LinkType) {
            throw "Linked files are forbidden: $Path"
        }
        if ($Directory -ne [bool]$item.PSIsContainer) { throw "Wrong file type: $Path" }
        return $item
    }
    function Get-Sha256([string]$Path) {
        $stream = [IO.File]::OpenRead($Path)
        $hasher = [Security.Cryptography.SHA256]::Create()
        try {
            return [BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLower()
        } finally {
            $hasher.Dispose()
            $stream.Dispose()
        }
    }
    $ancestor = $bundleRoot
    while ($ancestor) {
        $null = Assert-RegularPath $ancestor $true
        $parent = [IO.Directory]::GetParent($ancestor)
        if ($null -eq $parent) { break }
        $ancestor = $parent.FullName
    }
    $manifestPath = Join-Path $bundleRoot 'SUBMISSION_MANIFEST.json'
    $checksumPath = Join-Path $bundleRoot 'SUBMISSION_MANIFEST.sha256'
    $null = Assert-RegularPath $manifestPath $false
    $null = Assert-RegularPath $checksumPath $false
    $checksumText = (Get-Content -LiteralPath $checksumPath -Raw).Trim()
    if ($checksumText -notmatch '^([0-9a-f]{64})  SUBMISSION_MANIFEST\.json$') {
        throw 'Malformed manifest checksum.'
    }
    $manifestDigest = $Matches[1]
    $actualDigest = Get-Sha256 $manifestPath
    if ($actualDigest -cne $manifestDigest) {
        throw 'Manifest SHA-256 mismatch.'
    }
    $manifestData = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifestData.schema -cne 'gridguard.submission_manifest.1') {
        throw 'Unknown manifest schema.'
    }
    $expected = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $previous = $null
    [long]$totalBytes = 0
    foreach ($record in $manifestData.files) {
        $relative = [string]$record.path
        if (-not $relative -or $relative.Contains('\') -or $relative.Contains(':') -or
            [IO.Path]::IsPathRooted($relative) -or $relative -match '(^|/)(\.|\.\.|)(/|$)' -or
            $relative.EndsWith('/') -or $relative -eq 'SUBMISSION_MANIFEST.json' -or
            $relative -eq 'SUBMISSION_MANIFEST.sha256') { throw "Unsafe manifest path: $relative" }
        if ($null -ne $previous -and
            [StringComparer]::Ordinal.Compare($previous, $relative) -ge 0) {
            throw 'Manifest entries must be uniquely sorted by ordinal relative path.'
        }
        $previous = $relative
        if (-not $expected.Add($relative)) { throw "Case-colliding manifest path: $relative" }
        if ([string]$record.sha256 -cnotmatch '^[0-9a-f]{64}$' -or [long]$record.size_bytes -lt 0) {
            throw "Invalid manifest hash or size: $relative"
        }
        $full = [IO.Path]::GetFullPath((Join-Path $bundleRoot $relative))
        if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Path escapes submission: $relative"
        }
        $cursor = [IO.Path]::GetDirectoryName($full)
        while ($cursor -ne $bundleRoot) {
            $null = Assert-RegularPath $cursor $true
            $cursor = [IO.Path]::GetDirectoryName($cursor)
        }
        $item = Assert-RegularPath $full $false
        if ($item.Length -ne [long]$record.size_bytes) { throw "Size mismatch: $relative" }
        $fileDigest = Get-Sha256 $full
        if ($fileDigest -cne [string]$record.sha256) {
            throw "SHA-256 mismatch: $relative"
        }
        $totalBytes += $item.Length
    }
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $pending = [Collections.Generic.Stack[string]]::new()
    $pending.Push($bundleRoot)
    while ($pending.Count -gt 0) {
        foreach ($entry in [IO.Directory]::EnumerateFileSystemEntries($pending.Pop())) {
            $attributes = [IO.File]::GetAttributes($entry)
            if (($attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Links/reparse points are forbidden: $entry"
            }
            if (($attributes -band [IO.FileAttributes]::Directory) -ne 0) {
                $pending.Push($entry)
            } else {
                $null = Assert-RegularPath $entry $false
                $relative = $entry.Substring($prefix.Length).Replace('\', '/')
                if ($relative -in @('SUBMISSION_MANIFEST.json', 'SUBMISSION_MANIFEST.sha256')) {
                    continue
                }
                if (-not $expected.Contains($relative)) { throw "Unexpected file: $relative" }
                if (-not $seen.Add($relative)) { throw "Duplicate file: $relative" }
            }
        }
    }
    if ($seen.Count -ne $expected.Count) { throw 'Missing files in submission.' }
    if ($Json) {
        [ordered]@{
            passed=$true; files=$seen.Count; size_bytes=$totalBytes
            manifest_sha256=$manifestDigest
        } | ConvertTo-Json -Compress
    } else {
        Write-Host "PASS: $($seen.Count) files; $totalBytes bytes. Exact inventory verified."
    }
    exit 0
} catch {
    [Console]::Error.WriteLine("Submission verification failed: " + $_.Exception.Message)
    exit 1
}
"""


def _metadata(path: Path) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"Links/reparse points are forbidden: {path}")
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        raise ValueError(f"Only regular files/directories are accepted: {path}")
    return info


def _safe_ancestors(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.exists() or candidate.is_symlink():
            _metadata(candidate)


def safe_relative(value: str) -> str:
    parts = value.split("/")
    if (
        not value
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or PurePosixPath(value).is_absolute()
        or any(p in {"", ".", ".."} for p in parts)
        or any(p.endswith((".", " ")) for p in parts)
    ):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return value


def confined(root: Path, relative: str) -> Path:
    safe_relative(relative)
    target = root.joinpath(*relative.split("/"))
    _safe_ancestors(target)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes destination: {relative}")
    return target


def digest(path: Path) -> str:
    _safe_ancestors(path)
    if not stat.S_ISREG(_metadata(path).st_mode):
        raise ValueError(f"Expected regular file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def excluded_source(path: Path) -> bool:
    name = path.name.casefold()
    return (
        name in IGNORED_DIRECTORIES
        or name in SECRET_NAMES
        or name.startswith(".env")
        or path.suffix.casefold() in IGNORED_SUFFIXES | SECRET_SUFFIXES
        or name.startswith((".chart-data-", ".presentation-export-"))
    )


def tree_files(root: Path, *, clean_source: bool = False) -> list[Path]:
    """Enumerate without following links; reject every encountered link before filtering."""
    _safe_ancestors(root)
    if not stat.S_ISDIR(_metadata(root).st_mode):
        raise ValueError(f"Expected directory: {root}")
    result = []
    pending = [root]
    while pending:
        for path in sorted(pending.pop().iterdir(), key=lambda item: item.name):
            info = _metadata(path)
            if clean_source and excluded_source(path):
                continue
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            else:
                result.append(path)
    return sorted(result)


def _add(plan: dict[str, Path], relative: str, source: Path) -> None:
    safe_relative(relative)
    _safe_ancestors(source)
    if not stat.S_ISREG(_metadata(source).st_mode):
        raise ValueError(f"Missing/nonregular required input: {source}")
    collisions = {key.casefold(): key for key in plan}
    if relative.casefold() in collisions:
        raise ValueError(f"Duplicate output path: {relative}")
    plan[relative] = source


def make_plan(
    project: Path, app: Path, verification_dirs: list[Path], verification_files: list[Path]
) -> dict[str, Path]:
    plan: dict[str, Path] = {}
    for name in SOURCE_DIRECTORIES:
        for path in tree_files(project / name, clean_source=True):
            _add(plan, "Source/" + path.relative_to(project).as_posix(), path)
    for name in SOURCE_ROOT_FILES + SOURCE_OUTPUTS:
        _add(plan, "Source/" + name, project / name)
    for name in DOCUMENTS:
        _add(plan, "Documents/" + Path(name).name, project / name)
    for path in tree_files(app):
        _add(plan, "App/GridGuard Khartoum/" + path.relative_to(app).as_posix(), path)
    if "App/GridGuard Khartoum/GridGuard Khartoum.exe" not in plan:
        raise ValueError("The required application executable is missing.")
    for directory in verification_dirs:
        for path in tree_files(directory, clean_source=True):
            # Verification keeps deliberate audit logs, but never runtime caches.
            _add(plan, "Verification/" + path.relative_to(directory).as_posix(), path)
        for path in tree_files(directory):
            if path.suffix.casefold() == ".log" and not any(
                part.casefold() in IGNORED_DIRECTORIES
                for part in path.relative_to(directory).parts[:-1]
            ):
                _add(plan, "Verification/" + path.relative_to(directory).as_posix(), path)
    for path in verification_files:
        _add(plan, "Verification/" + path.name, path)
    return plan


def _generated() -> dict[str, bytes]:
    return {
        "START HERE.md": START_HERE.encode("utf-8"),
        "Run GridGuard.cmd": LAUNCHER.replace("\n", "\r\n").encode("ascii"),
        "Verify Submission.ps1": POWERSHELL_VERIFIER.replace("\n", "\r\n").encode("utf-8-sig"),
    }


def _state_path(destination: Path) -> Path:
    return destination.parent / (destination.name + ".prepare.json")


def _write_atomic(path: Path, content: bytes) -> None:
    _safe_ancestors(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    with temporary.open("xb") as stream:
        stream.write(content)
    os.replace(temporary, path)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def records_for(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in sorted(tree_files(root), key=lambda p: p.relative_to(root).as_posix())
        if path.relative_to(root).as_posix() not in {MANIFEST, CHECKSUM}
    ]


def verify_records(root: Path, records: list[dict[str, object]], *, sealed: bool = False) -> None:
    _safe_ancestors(root)
    expected: set[str] = set()
    folded: set[str] = set()
    previous = ""
    for record in records:
        relative = safe_relative(str(record["path"]))
        if (
            relative in {MANIFEST, CHECKSUM}
            or relative.casefold() in folded
            or relative <= previous
        ):
            raise ValueError(f"Duplicate, unsorted or reserved manifest entry: {relative}")
        previous = relative
        folded.add(relative.casefold())
        expected.add(relative)
        path = confined(root, relative)
        info = _metadata(path)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"Expected an ordinary unlinked file: {relative}")
        if info.st_size != record["size_bytes"] or digest(path) != record["sha256"]:
            raise ValueError(f"Integrity mismatch: {relative}")
    actual = {p.relative_to(root).as_posix() for p in tree_files(root)}
    if sealed:
        expected |= {MANIFEST, CHECKSUM}
    if actual != expected:
        raise ValueError(
            f"File inventory mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def prepare(
    project: Path,
    app: Path,
    destination: Path,
    *,
    refresh: bool = False,
    verification_dirs: list[Path] | None = None,
    verification_files: list[Path] | None = None,
) -> dict[str, object]:
    project, app, destination = (p.absolute() for p in (project, app, destination))
    for path in (project, app, destination):
        _safe_ancestors(path)
    if (
        destination == project
        or project.is_relative_to(destination)
        or destination.is_relative_to(app)
    ):
        raise ValueError("Destination overlaps an input root.")
    if any(destination.is_relative_to(project / name) for name in SOURCE_DIRECTORIES):
        raise ValueError("Destination would recursively copy itself.")
    if (destination / MANIFEST).exists() or (destination / CHECKSUM).exists():
        raise ValueError("The destination is sealed; choose a new submission folder.")
    exists = destination.exists()
    if exists and not refresh:
        raise FileExistsError(
            "Destination already exists; --refresh is required for allowlisted updates."
        )
    state_path = _state_path(destination)
    if exists and not state_path.is_file():
        raise ValueError("Refusing to refresh a folder not created by this builder.")
    plan = make_plan(project, app, verification_dirs or [], verification_files or [])
    generated = _generated()
    allowed = set(plan) | set(generated)
    if exists:
        old_state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            old_state.get("destination") != str(destination)
            or old_state.get("schema") != "gridguard.submission_prepare.1"
        ):
            raise ValueError("Prepare state is not bound to this destination.")
        actual = {p.relative_to(destination).as_posix() for p in tree_files(destination)}
        unexpected = actual - allowed
        if unexpected:
            raise ValueError(
                f"Unexpected files block refresh; nothing will be deleted: {sorted(unexpected)}"
            )
    destination.mkdir(parents=True, exist_ok=refresh)
    copied = unchanged = 0
    for relative, source in sorted(plan.items()):
        target = confined(destination, relative)
        source_hash = digest(source)
        if (
            target.is_file()
            and target.stat().st_size == source.stat().st_size
            and digest(target) == source_hash
        ):
            unchanged += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".copying-" + uuid.uuid4().hex)
        shutil.copyfile(source, temporary)
        if digest(temporary) != source_hash:
            raise ValueError(f"Source changed while copying: {source}")
        os.replace(temporary, target)
        copied += 1
    for relative, content in generated.items():
        target = confined(destination, relative)
        if target.is_file() and target.read_bytes() == content:
            unchanged += 1
        else:
            _write_atomic(target, content)
            copied += 1
    records = records_for(destination)
    state = {
        "schema": "gridguard.submission_prepare.1",
        "destination": str(destination),
        "project": str(project),
        "app_input": str(app),
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "files": records,
    }
    _write_atomic(state_path, _json_bytes(state))
    return {
        "phase": "prepared",
        "destination": str(destination),
        "copied": copied,
        "unchanged": unchanged,
        "files": len(records),
        "size_bytes": sum(int(r["size_bytes"]) for r in records),
        "state": str(state_path),
    }


def finalize(destination: Path, archive: Path | None = None) -> dict[str, object]:
    destination = destination.absolute()
    archive = archive.absolute() if archive else destination.with_suffix(".zip")
    _safe_ancestors(destination)
    _safe_ancestors(archive)
    if archive.is_relative_to(destination):
        raise ValueError("Archive must be outside the submission folder.")
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    if (
        archive.exists()
        or checksum.exists()
        or (destination / MANIFEST).exists()
        or (destination / CHECKSUM).exists()
    ):
        raise FileExistsError("Final outputs already exist; use a new destination/archive.")
    state_path = _state_path(destination)
    _safe_ancestors(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema") != "gridguard.submission_prepare.1" or state.get("destination") != str(
        destination
    ):
        raise ValueError("Prepare state is not bound to the selected folder.")
    records = state["files"]
    if not any(str(r["path"]).startswith("Verification/") for r in records):
        raise ValueError("Final verification reports are required before sealing.")
    verify_records(destination, records)
    manifest = {
        "schema": "gridguard.submission_manifest.1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "folder_name": destination.name,
        "hash_algorithm": "SHA-256",
        "files": records,
    }
    _write_atomic(destination / MANIFEST, _json_bytes(manifest))
    manifest_hash = digest(destination / MANIFEST)
    _write_atomic(destination / CHECKSUM, f"{manifest_hash}  {MANIFEST}\n".encode("ascii"))
    verify_records(destination, records, sealed=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".writing-" + uuid.uuid4().hex)
    with zipfile.ZipFile(
        temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as bundle:
        for path in tree_files(destination):
            bundle.write(
                path, arcname=destination.name + "/" + path.relative_to(destination).as_posix()
            )
    os.replace(temporary, archive)
    archive_hash = digest(archive)
    _write_atomic(checksum, f"{archive_hash}  {archive.name}\n".encode("ascii"))
    return {
        "phase": "finalized",
        "destination": str(destination),
        "files": len(records),
        "size_bytes": sum(int(r["size_bytes"]) for r in records),
        "manifest_sha256": manifest_hash,
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "finalize", "verify"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--app-input", type=Path)
    parser.add_argument("--verification-dir", type=Path, action="append", default=[])
    parser.add_argument("--verification-file", type=Path, action="append", default=[])
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args(argv)
    project = args.project_root.absolute()
    destination = (
        args.destination or project / "output/submission/GridGuard-Khartoum-Submission-2026-10-23"
    )
    try:
        if args.phase == "prepare":
            result = prepare(
                project,
                args.app_input or project / "output/app/GridGuard Khartoum",
                destination,
                refresh=args.refresh,
                verification_dirs=args.verification_dir,
                verification_files=args.verification_file,
            )
        elif args.phase == "finalize":
            result = finalize(destination, args.archive)
        else:
            manifest_path = destination / MANIFEST
            checksum = (destination / CHECKSUM).read_text(encoding="ascii").strip()
            if checksum != f"{digest(manifest_path)}  {MANIFEST}":
                raise ValueError("Manifest checksum mismatch.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") != "gridguard.submission_manifest.1":
                raise ValueError("Unknown submission manifest schema.")
            verify_records(destination, manifest["files"], sealed=True)
            result = {"phase": "verified", "passed": True, "files": len(manifest["files"])}
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Submission build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
