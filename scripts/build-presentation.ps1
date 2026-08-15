param(
    [switch]$VerifyOnly,
    [switch]$RefreshManifest
)

$ErrorActionPreference = "Stop"

if ($VerifyOnly -and $RefreshManifest) {
    throw "VerifyOnly and RefreshManifest cannot be used together."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryParent = Split-Path -Parent $projectRoot
$distribution = Join-Path $projectRoot "dist\GridGuard Khartoum"
$executable = Join-Path $distribution "GridGuard Khartoum.exe"
$smokeRoot = Join-Path $projectRoot "build\gridguard"
$smokeReport = Join-Path $smokeRoot "frozen-smoke.json"
$releaseManifestPath = Join-Path $projectRoot "RELEASE_MANIFEST.json"
$releaseIntegrityHelper = Join-Path $PSScriptRoot "release-integrity.ps1"
if (-not (Test-Path -LiteralPath $releaseIntegrityHelper -PathType Leaf)) {
    throw "Release-integrity helper is unavailable: $releaseIntegrityHelper"
}
. $releaseIntegrityHelper

function Assert-NativeCommandSucceeded {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Operation,
        [Parameter(Mandatory = $true)]
        [int]$ExitCode
    )

    if ($ExitCode -ne 0) {
        throw "$Operation failed with exit code $ExitCode."
    }
}

function Get-ValidatedFrozenSmokePayload {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportPath
    )

    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        throw "Frozen presentation smoke report was not created: $ReportPath"
    }
    $smokePayload = Get-Content -Raw -LiteralPath $ReportPath | ConvertFrom-Json
    if ($smokePayload.status -ne "PASS") {
        $smokeDetails = Get-Content -Raw -LiteralPath $ReportPath
        throw "Frozen presentation smoke did not pass: $smokeDetails"
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
        $smokeDetails = Get-Content -Raw -LiteralPath $ReportPath
        throw "Frozen presentation live CUDA smoke is incomplete: $smokeDetails"
    }
    return $smokePayload
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
        [Parameter(Mandatory = $true)]
        [pscustomobject]$SmokePayload
    )

    $releaseManifest = Assert-ReleaseManifestCore `
        -ProjectRoot $ProjectRoot `
        -ManifestPath $ManifestPath `
        -DistributionPath $DistributionPath `
        -ExecutablePath $ExecutablePath `
        -SmokeReportPath $SmokeReportPath `
        -SmokePayload $SmokePayload
    if ($releaseManifest.schema_version -ne 2) {
        throw "Release manifest schema_version must be 2."
    }

    $expectedExecutablePath = "dist/GridGuard Khartoum/GridGuard Khartoum.exe"
    if ($releaseManifest.windows_build.executable_path -ne $expectedExecutablePath) {
        throw "Release manifest executable_path must be $expectedExecutablePath."
    }
    if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) {
        throw "Presentation executable was not created: $ExecutablePath"
    }
    $executableItem = Get-Item -LiteralPath $ExecutablePath
    $executableHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ExecutablePath).Hash
    if (
        $releaseManifest.windows_build.executable_size_bytes -ne $executableItem.Length -or
        -not [string]::Equals(
            [string]$releaseManifest.windows_build.executable_sha256,
            $executableHash,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw (
            "Release manifest does not bind the built executable: " +
            "bytes=$($executableItem.Length), sha256=$executableHash."
        )
    }

    if (-not (Test-Path -LiteralPath $DistributionPath -PathType Container)) {
        throw "Presentation distribution was not created: $DistributionPath"
    }
    $distributionFiles = @(Get-ChildItem -LiteralPath $DistributionPath -Recurse -File -Force)
    [int64]$distributionBytes = (
        $distributionFiles | Measure-Object -Property Length -Sum
    ).Sum
    if (
        $releaseManifest.windows_build.distribution_file_count -ne $distributionFiles.Count -or
        $releaseManifest.windows_build.distribution_size_bytes -ne $distributionBytes -or
        $releaseManifest.windows_build.distribution_tree_sha256 -ne
        $releaseManifest.windows_build.distribution.tree_sha256
    ) {
        throw (
            "Release manifest does not bind the built distribution: " +
            "files=$($distributionFiles.Count), bytes=$distributionBytes, " +
            "tree_sha256=$($releaseManifest.windows_build.distribution.tree_sha256)."
        )
    }

    $expectedSmokeReportPath = "build/gridguard/frozen-smoke.json"
    if (
        $releaseManifest.windows_build.frozen_smoke_report.path -ne
        $expectedSmokeReportPath
    ) {
        throw "Release manifest frozen_smoke_report.path must be $expectedSmokeReportPath."
    }
    $smokeReportItem = Get-Item -LiteralPath $SmokeReportPath
    $smokeReportHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SmokeReportPath).Hash
    if (
        $releaseManifest.windows_build.frozen_smoke_report.size_bytes -ne
        $smokeReportItem.Length -or
        -not [string]::Equals(
            [string]$releaseManifest.windows_build.frozen_smoke_report.sha256,
            $smokeReportHash,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw (
            "Release manifest does not bind the frozen smoke report: " +
            "bytes=$($smokeReportItem.Length), sha256=$smokeReportHash."
        )
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
}

if (-not $VerifyOnly) {
    $configuredSourceRepository = [Environment]::GetEnvironmentVariable(
        "GRIDGUARD_SOURCE_REPOSITORY", "Process"
    )
    if ([string]::IsNullOrWhiteSpace($configuredSourceRepository)) {
        $sourceRepositoryCandidate = Join-Path $repositoryParent "aipower_rl"
    }
    else {
        $sourceRepositoryCandidate = $configuredSourceRepository
    }
    if (-not (Test-Path -LiteralPath $sourceRepositoryCandidate -PathType Container)) {
        throw "Trusted source repository is unavailable: $sourceRepositoryCandidate"
    }
    $sourceRepository = (Resolve-Path -LiteralPath $sourceRepositoryCandidate).Path
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRepository ".git"))) {
        throw "Trusted source repository is not a Git repository: $sourceRepository"
    }

    $configuredSourceCheckpoint = [Environment]::GetEnvironmentVariable(
        "GRIDGUARD_SOURCE_CHECKPOINT", "Process"
    )
    if ([string]::IsNullOrWhiteSpace($configuredSourceCheckpoint)) {
        $sourceCheckpointCandidate = Join-Path $sourceRepository "artifacts\checkpoints\best.pt"
    }
    else {
        $sourceCheckpointCandidate = $configuredSourceCheckpoint
    }
    if (-not (Test-Path -LiteralPath $sourceCheckpointCandidate -PathType Leaf)) {
        throw "Trusted source checkpoint is unavailable: $sourceCheckpointCandidate"
    }
    $sourceCheckpoint = (Resolve-Path -LiteralPath $sourceCheckpointCandidate).Path

    $testReportDirectory = Join-Path $projectRoot "build"
    $testReport = Join-Path $testReportDirectory "release-tests.xml"
    New-Item -ItemType Directory -Path $testReportDirectory -Force | Out-Null
    if (Test-Path -LiteralPath $testReport) {
        Remove-Item -LiteralPath $testReport -Force
    }

    $previousSourceRepository = [Environment]::GetEnvironmentVariable(
        "GRIDGUARD_SOURCE_REPOSITORY", "Process"
    )
    $previousSourceCheckpoint = [Environment]::GetEnvironmentVariable(
        "GRIDGUARD_SOURCE_CHECKPOINT", "Process"
    )

    Push-Location $projectRoot
    try {
        [Environment]::SetEnvironmentVariable(
            "GRIDGUARD_SOURCE_REPOSITORY", $sourceRepository, "Process"
        )
        [Environment]::SetEnvironmentVariable(
            "GRIDGUARD_SOURCE_CHECKPOINT", $sourceCheckpoint, "Process"
        )
        uv sync --locked --extra dev --extra docs
        Assert-NativeCommandSucceeded -Operation "Locked dependency sync" -ExitCode $LASTEXITCODE
        uv run gridguard-doctor
        Assert-NativeCommandSucceeded -Operation "GPU preflight" -ExitCode $LASTEXITCODE
        $releaseTestArguments = @("run", "pytest", "-q", "--junitxml", $testReport)
        if ($RefreshManifest) {
            # The old manifest cannot validate a build that does not exist yet. This contract is
            # run explicitly after the freshly smoke-tested distribution is bound below.
            $releaseTestArguments += "--ignore=tests/test_release_manifest.py"
        }
        uv @releaseTestArguments
        $pytestExitCode = $LASTEXITCODE
        if (-not (Test-Path -LiteralPath $testReport -PathType Leaf)) {
            throw "Pytest did not create its JUnit report: $testReport"
        }
        [xml]$testResults = Get-Content -Raw -LiteralPath $testReport
        $testSuites = @($testResults.testsuites.testsuite)
        if ($testSuites.Count -eq 0) {
            throw "Pytest JUnit report does not contain a test suite: $testReport"
        }
        $testCount = 0
        $failureCount = 0
        $errorCount = 0
        $skippedCount = 0
        foreach ($testSuite in $testSuites) {
            $testCount += [int]$testSuite.tests
            $failureCount += [int]$testSuite.failures
            $errorCount += [int]$testSuite.errors
            $skippedCount += [int]$testSuite.skipped
        }
        Assert-NativeCommandSucceeded -Operation "Release test suite" -ExitCode $pytestExitCode
        if ($testCount -eq 0 -or $failureCount -ne 0 -or $errorCount -ne 0) {
            throw (
                "Release test report is incomplete: tests=$testCount, " +
                "failures=$failureCount, errors=$errorCount."
            )
        }
        if ($skippedCount -ne 0) {
            throw "Release test suite skipped $skippedCount test(s); all release tests are required."
        }
        uv run pyinstaller --noconfirm --clean packaging\gridguard.spec
        Assert-NativeCommandSucceeded -Operation "PyInstaller build" -ExitCode $LASTEXITCODE
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            "GRIDGUARD_SOURCE_REPOSITORY", $previousSourceRepository, "Process"
        )
        [Environment]::SetEnvironmentVariable(
            "GRIDGUARD_SOURCE_CHECKPOINT", $previousSourceCheckpoint, "Process"
        )
        Pop-Location
    }

    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Presentation executable was not created: $executable"
    }

    $previousSmokeReport = [Environment]::GetEnvironmentVariable(
        "GRIDGUARD_SMOKE_REPORT", "Process"
    )
    $previousQtPlatform = [Environment]::GetEnvironmentVariable(
        "QT_QPA_PLATFORM", "Process"
    )

    try {
        [Environment]::SetEnvironmentVariable(
            "GRIDGUARD_SMOKE_REPORT", $smokeReport, "Process"
        )
        [Environment]::SetEnvironmentVariable("QT_QPA_PLATFORM", "offscreen", "Process")
        if (Test-Path -LiteralPath $smokeReport) {
            Remove-Item -LiteralPath $smokeReport -Force
        }
        $smokeProcess = Start-Process `
            -FilePath $executable `
            -WorkingDirectory (Split-Path -Parent $executable) `
            -WindowStyle Hidden `
            -Wait `
            -PassThru
        if ($smokeProcess.ExitCode -ne 0) {
            $smokeDetails = if (Test-Path -LiteralPath $smokeReport) {
                Get-Content -Raw -LiteralPath $smokeReport
            }
            else {
                "smoke report missing"
            }
            throw (
                "Frozen presentation smoke failed with exit code " +
                "$($smokeProcess.ExitCode): $smokeDetails"
            )
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            "GRIDGUARD_SMOKE_REPORT", $previousSmokeReport, "Process"
        )
        [Environment]::SetEnvironmentVariable(
            "QT_QPA_PLATFORM", $previousQtPlatform, "Process"
        )
    }
}

$smokePayload = Get-ValidatedFrozenSmokePayload -ReportPath $smokeReport
if ($RefreshManifest) {
    Update-ReleaseManifestBindings `
        -ProjectRoot $projectRoot `
        -ManifestPath $releaseManifestPath `
        -DistributionPath $distribution `
        -ExecutablePath $executable `
        -SmokeReportPath $smokeReport
    uv run pytest -q tests/test_release_manifest.py
    Assert-NativeCommandSucceeded `
        -Operation "Refreshed release manifest contract" `
        -ExitCode $LASTEXITCODE
    Write-Host "Refreshed RELEASE_MANIFEST.json from the verified build outputs."
}
Assert-ReleaseSourceSnapshot `
    -ProjectRoot $projectRoot `
    -ManifestPath $releaseManifestPath
Assert-ReleaseManifestBinding `
    -ProjectRoot $projectRoot `
    -ManifestPath $releaseManifestPath `
    -DistributionPath $distribution `
    -ExecutablePath $executable `
    -SmokeReportPath $smokeReport `
    -SmokePayload $smokePayload

if ($VerifyOnly) {
    Write-Host "Existing presentation package matches RELEASE_MANIFEST.json: $executable"
}
else {
    Write-Host "Presentation package ready: $executable"
    Write-Host "Frozen presentation smoke: PASS ($smokeReport)"
}
