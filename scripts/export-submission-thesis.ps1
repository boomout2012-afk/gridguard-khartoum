param(
    [Parameter(Mandatory = $true)][string]$DocumentPath,
    [Parameter(Mandatory = $true)][string]$PdfPath
)

$ErrorActionPreference = 'Stop'
$documentFile = [IO.Path]::GetFullPath($DocumentPath)
$pdfFile = [IO.Path]::GetFullPath($PdfPath)
if (-not (Test-Path -LiteralPath $documentFile -PathType Leaf)) {
    throw "Word document does not exist: $documentFile"
}
$null = New-Item -ItemType Directory -Force -Path ([IO.Path]::GetDirectoryName($pdfFile))
$word = $null
$document = $null
try {
    # A dedicated hidden instance leaves the user's existing Word documents alone.
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($documentFile, $false, $false)
    $document.Repaginate()
    foreach ($contents in $document.TablesOfContents) { $contents.Update() }
    $fieldResult = $document.Fields.Update()
    if ($fieldResult -ne 0) { throw "Word could not update field $fieldResult" }
    $document.Repaginate()
    foreach ($contents in $document.TablesOfContents) { $contents.Update() }
    $fieldResult = $document.Fields.Update()
    if ($fieldResult -ne 0) { throw "Word could not update field $fieldResult" }
    foreach ($story in $document.StoryRanges) { $null = $story.Fields.Update() }
    $document.Save()
    $document.ExportAsFixedFormat($pdfFile, 17)
    $pageCount = $document.ComputeStatistics(2)
    $document.Close(0)
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    $document = $null
    $receiptPath = $documentFile + '.build.json'
    if (Test-Path -LiteralPath $receiptPath -PathType Leaf) {
        $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json -AsHashtable
        $receipt.docx_sha256 = (Get-FileHash -LiteralPath $documentFile -Algorithm SHA256).Hash.ToLowerInvariant()
        $receipt.fields_require_refresh = $false
        $receipt.word_export = [ordered]@{
            renderer = 'Microsoft Word'
            page_count = $pageCount
            pdf_sha256 = (Get-FileHash -LiteralPath $pdfFile -Algorithm SHA256).Hash.ToLowerInvariant()
            export_script_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 20) + "`n", [Text.UTF8Encoding]::new($false))
    }
    Write-Output "Word pages: $pageCount"
    Write-Output $pdfFile
}
finally {
    if ($null -ne $document) { $document.Close(0) }
    if ($null -ne $word -and $word.Documents.Count -eq 0) { $word.Quit() }
    if ($null -ne $document) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($document) }
    if ($null -ne $word) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($word) }
}
