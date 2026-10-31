<#
.SYNOPSIS
Export the editable submission deck to PDF using Microsoft PowerPoint on Windows.
.DESCRIPTION
Works from a copied project folder; no Python, Node.js, Codex or private tmp files
are required. Install the supplied Atkinson Hyperlegible fonts before exporting
to preserve the submitted typography. The supplied PDF is ready to view without
PowerPoint. The PPTX is the portable editable source, including charts and tables.

Full data-driven regeneration with tools/build_focused_presentation.mjs is an
optional authoring route requiring the Codex artifact runtime and presentation
skill. This script exports edits made directly in the supplied PPTX instead.

Relative paths are resolved against the project root, independent of the current
working directory. The input is opened read-only and is never saved or modified.
.EXAMPLE
powershell -ExecutionPolicy Bypass -File scripts/export-submission-presentation.ps1
.EXAMPLE
.\scripts\export-submission-presentation.ps1 -InputPath 'my-deck.pptx' -OutputPath 'my-deck.pdf'
#>
[CmdletBinding()]
param(
    [string]$InputPath = 'output/presentation/updated/gridguard-khartoum-defense.pptx',
    [string]$OutputPath = 'output/pdf/updated/gridguard-khartoum-project-presentation.pdf'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
function Resolve-ProjectPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot $Value))
}
$source = Resolve-ProjectPath $InputPath
$destination = Resolve-ProjectPath $OutputPath
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Input PPTX not found: $source" }
if ([IO.Path]::GetExtension($source) -ne '.pptx') { throw 'InputPath must point to a .pptx file.' }
if ([IO.Path]::GetExtension($destination) -ne '.pdf') { throw 'OutputPath must end in .pdf.' }
$sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash
$outputDirectory = [IO.Path]::GetDirectoryName($destination)
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$staging = Join-Path $outputDirectory ('.presentation-export-' + [guid]::NewGuid().ToString('N') + '.pdf')
$application = $null
$deck = $null
try {
    try { $application = New-Object -ComObject PowerPoint.Application }
    catch { throw 'Microsoft PowerPoint for Windows is required to export. The included PDF can be viewed without it.' }
    # msoTrue = -1; msoFalse = 0. Open read-only, without a document window.
    $deck = $application.Presentations.Open($source, -1, 0, 0)
    $slideCount = $deck.Slides.Count
    if ($slideCount -lt 1) { throw 'The input presentation has no slides.' }
    # ppSaveAsPDF = 32. Save to a unique temporary PDF, preserving any prior output on failure.
    $deck.SaveAs($staging, 32)
    if (-not (Test-Path -LiteralPath $staging -PathType Leaf)) { throw 'PowerPoint did not create a PDF.' }
    if ((Get-Item -LiteralPath $staging).Length -eq 0) { throw 'PowerPoint created an empty PDF.' }
    if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $sourceHash) {
        throw 'The input presentation changed during export.'
    }
    Move-Item -LiteralPath $staging -Destination $destination -Force
    [ordered]@{
        source = $source
        source_sha256 = $sourceHash.ToLowerInvariant()
        pdf = $destination
        pdf_sha256 = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
        slides = $slideCount
        application = 'Microsoft PowerPoint'
    } | ConvertTo-Json
} finally {
    if ($null -ne $deck) {
        $deck.Close()
        [Runtime.InteropServices.Marshal]::ReleaseComObject($deck) | Out-Null
    }
    if ($null -ne $application) {
        # Do not close unrelated presentations if COM reused an existing PowerPoint process.
        if ($application.Presentations.Count -eq 0) { $application.Quit() }
        [Runtime.InteropServices.Marshal]::ReleaseComObject($application) | Out-Null
    }
    if (Test-Path -LiteralPath $staging -PathType Leaf) { Remove-Item -LiteralPath $staging }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
