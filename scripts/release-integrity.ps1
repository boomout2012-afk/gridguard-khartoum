# Shared release-integrity primitives for the build gate and presentation launcher.
#
# We deliberately bind the unpacked PyInstaller onedir instead of producing a ZIP here.
# Compress-Archive records filesystem timestamps and does not promise byte-for-byte stable
# archives across PowerShell/.NET versions. A sorted per-file inventory plus a canonical tree
# digest provides deterministic, actionable verification without claiming archive reproducibility.

$script:DistributionTreeHashAlgorithm = "sha256(canonical-lines-v1)"
$script:RequiredTrackedArtifactPaths = @(
    "data/evidence/paired-benchmark.json",
    "data/evidence/source-manifest.json",
    "data/model/manifest.json",
    "data/model/ppo-policy.npz",
    "docs/decisions/ADR-002-geographic-boundary-map.md",
    "docs/source/graduation-project-proposal.pdf",
    "output/pdf/gridguard-khartoum-final-report.pdf",
    "output/presentation/gridguard-khartoum-defense.pptx",
    "src/gridguard/assets/map/THIRD_PARTY_DATA.md",
    "src/gridguard/assets/map/khartoum-localities.geojson",
    "src/gridguard/assets/map/khartoum-nile.geojson",
    "src/gridguard/assets/map/map-provenance.json"
)

function Get-ReleaseSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath
    )

    return (Get-FileHash -Algorithm SHA256 -LiteralPath $LiteralPath).Hash.ToLowerInvariant()
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
    $item = Get-Item -LiteralPath $AbsolutePath
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
    $item = Get-Item -LiteralPath $absolutePath
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

function Get-DistributionRelativePaths {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DistributionPath
    )

    if (-not (Test-Path -LiteralPath $DistributionPath -PathType Container)) {
        throw "Presentation distribution is missing: $DistributionPath"
    }

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
    try {
        $smokePayload = Get-Content -Raw -LiteralPath $ReportPath | ConvertFrom-Json
    }
    catch {
        throw "Frozen presentation smoke report is invalid JSON: $($_.Exception.Message)"
    }
    if ($smokePayload.status -ne "PASS") {
        throw "Frozen presentation smoke did not pass: $ReportPath"
    }
    if (
        $smokePayload.live_scenario.shortage_level -ne "moderate" -or
        $smokePayload.live_scenario.seed -ne 50002 -or
        $smokePayload.live_scenario.device -ne "cuda:0" -or
        $smokePayload.live_scenario.controllers[0] -ne "PPO" -or
        $smokePayload.live_scenario.controllers[1] -ne "Fixed Rotation" -or
        $smokePayload.live_scenario.frame_counts.PPO -ne 24 -or
        $smokePayload.live_scenario.frame_counts."Fixed Rotation" -ne 24 -or
        $smokePayload.live_scenario.finite_core_values -ne $true -or
        $smokePayload.live_scenario.validated_core_value_count -le 0 -or
        $smokePayload.live_scenario.installed_in_live_page -ne $true -or
        $smokePayload.live_scenario.selected_controller -ne "Fixed Rotation" -or
        $smokePayload.live_scenario.selected_hour -ne 23 -or
        $smokePayload.live_scenario.expected_results_table_rows -ne 7 -or
        $smokePayload.live_scenario.results_table_rows -ne 7
    ) {
        throw "Frozen presentation live CUDA smoke is incomplete: $ReportPath"
    }
    return $smokePayload
}

function Get-ReleaseManifestPayload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath
    )

    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Release manifest is unavailable: $ManifestPath"
    }
    try {
        $payload = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
    }
    catch {
        throw "Release manifest is invalid JSON: $($_.Exception.Message)"
    }
    if ($payload.schema_version -ne 2) {
        throw "Release manifest schema_version must be 2."
    }
    return $payload
}

function Get-ReleaseManifestForUpdate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ManifestPath
    )

    if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
        throw "Release manifest is unavailable: $ManifestPath"
    }
    try {
        $payload = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
    }
    catch {
        throw "Release manifest is invalid JSON: $($_.Exception.Message)"
    }
    if ($payload.schema_version -eq 2) {
        return $payload
    }
    if ($payload.schema_version -ne 1) {
        throw "Only release manifest schema_version 1 or 2 can be refreshed."
    }

    return [pscustomobject][ordered]@{
        schema_version = 2
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
    if ([string]$releaseManifest.source.git_commit -notmatch "^[0-9a-f]{40}$") {
        throw "Release manifest source.git_commit must be a lowercase Git commit ID."
    }
    [void](Assert-FileBinding `
        -RootPath $ProjectRoot `
        -Binding $releaseManifest.source.uv_lock `
        -FieldName "source.uv_lock" `
        -ExpectedPath "uv.lock")

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

        # This is the source snapshot used to build the release. The manifest freeze commit is
        # normally created afterward, avoiding an impossible self-reference to its own commit.
        $GitCommit = (& git -C $ProjectRoot rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0) {
            throw "Cannot read the Git commit for release provenance."
        }
    }
    if ($GitCommit -notmatch "^[0-9a-f]{40}$") {
        throw "Release provenance Git commit must be a lowercase 40-character ID."
    }

    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.source `
        -Name "git_commit" `
        -Value $GitCommit
    $uvLockPath = Join-Path $ProjectRoot "uv.lock"
    Set-ReleaseObjectProperty `
        -InputObject $releaseManifest.source `
        -Name "uv_lock" `
        -Value (Get-FileBinding -AbsolutePath $uvLockPath -ManifestPathValue "uv.lock")

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
