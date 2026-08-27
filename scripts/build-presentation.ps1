param(
    [switch]$VerifyOnly,
    [switch]$RefreshManifest,
    [switch]$StrictSourceParity,
    [string]$SourceRepositoryPath,
    [string]$SourceCheckpointPath
)

$ErrorActionPreference = "Stop"

if ($VerifyOnly -and $RefreshManifest) {
    throw "VerifyOnly and RefreshManifest cannot be used together."
}
if (
    $VerifyOnly -and (
        $StrictSourceParity -or
        -not [string]::IsNullOrWhiteSpace($SourceRepositoryPath) -or
        -not [string]::IsNullOrWhiteSpace($SourceCheckpointPath)
    )
) {
    throw "VerifyOnly cannot run source parity validation."
}
if (
    -not $StrictSourceParity -and (
        -not [string]::IsNullOrWhiteSpace($SourceRepositoryPath) -or
        -not [string]::IsNullOrWhiteSpace($SourceCheckpointPath)
    )
) {
    throw "source parity paths require -StrictSourceParity."
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distribution = Join-Path $projectRoot "dist\GridGuard Khartoum"
$executable = Join-Path $distribution "GridGuard Khartoum.exe"
$wheelArchive = Join-Path $projectRoot "dist\gridguard_khartoum-1.2.0-py3-none-any.whl"
$sourceArchive = Join-Path $projectRoot "dist\gridguard_khartoum-1.2.0.tar.gz"
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

function Assert-NoAmbientPytestControls {
    $controls = @(
        Get-ChildItem Env: |
            Where-Object {
                $_.Name.StartsWith(
                    "PYTEST_",
                    [System.StringComparison]::OrdinalIgnoreCase
                ) -and
                $_.Name -ne "PYTEST_CURRENT_TEST" -and
                $_.Name -ne "PYTEST_VERSION" -and
                -not [string]::IsNullOrWhiteSpace([string]$_.Value)
            } |
            ForEach-Object Name |
            Sort-Object -CaseSensitive
    )
    if ($controls.Count -ne 0) {
        throw (
            "Ambient pytest controls are forbidden for release validation: " +
            "$($controls -join ', ')."
        )
    }
}

function Assert-PytestJUnitPassedWithoutSkips {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportPath,
        [Parameter(Mandatory = $true)]
        [int]$ExitCode,
        [Parameter(Mandatory = $true)]
        [string]$Operation
    )

    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        throw "$Operation did not create its JUnit report: $ReportPath"
    }
    [xml]$testResults = Get-Content -Raw -LiteralPath $ReportPath
    $testSuites = @($testResults.testsuites.testsuite)
    if ($testSuites.Count -eq 0) {
        throw "$Operation JUnit report does not contain a test suite: $ReportPath"
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
    Assert-NativeCommandSucceeded -Operation $Operation -ExitCode $ExitCode
    if ($testCount -eq 0 -or $failureCount -ne 0 -or $errorCount -ne 0) {
        throw (
            "$Operation report is incomplete: tests=$testCount, " +
            "failures=$failureCount, errors=$errorCount."
        )
    }
    if ($skippedCount -ne 0) {
        throw "$Operation skipped $skippedCount test(s); every selected test is required."
    }
}

function Assert-CommittedReleaseValidationRecords {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [switch]$VerifyMachineInputs
    )

    Push-Location $ProjectRoot
    try {
        uv run --no-sync python tools/release_validation.py --verify
        Assert-NativeCommandSucceeded `
            -Operation "Committed visual-review validation" `
            -ExitCode $LASTEXITCODE
        if ($VerifyMachineInputs) {
            uv run --no-sync python tools/machine_validation.py --verify --verify-inputs
        }
        else {
            uv run --no-sync python tools/machine_validation.py --verify
        }
        Assert-NativeCommandSucceeded `
            -Operation "Committed machine-validation record" `
            -ExitCode $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
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
    if ($releaseManifest.schema_version -ne 3) {
        throw "Release manifest schema_version must be 3."
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
    return $releaseManifest
}

$releaseValidationDirectory = Join-Path $projectRoot "tmp\release-validation"
$testCollection = Join-Path $releaseValidationDirectory "release-tests.collection.json"
$testReport = Join-Path $releaseValidationDirectory "release-tests.xml"
$releaseArtifactReport = Join-Path `
    $releaseValidationDirectory `
    "release-artifact-tests.xml"
$sourceParityReport = Join-Path $releaseValidationDirectory "source-parity-tests.xml"
$pairedReproduction = Join-Path $releaseValidationDirectory "paired-benchmark-reproduced.json"
$robustnessReproduction = Join-Path `
    $releaseValidationDirectory `
    "robustness-supplement-reproduced.json"

if (-not $VerifyOnly) {
    $sourceRepository = $null
    $sourceCheckpoint = $null
    if ($StrictSourceParity) {
        $sourceRepositoryCandidate = $SourceRepositoryPath
        if ([string]::IsNullOrWhiteSpace($sourceRepositoryCandidate)) {
            $sourceRepositoryCandidate = [Environment]::GetEnvironmentVariable(
                "GRIDGUARD_SOURCE_REPOSITORY", "Process"
            )
        }
        if ([string]::IsNullOrWhiteSpace($sourceRepositoryCandidate)) {
            throw (
                "strict source parity requires GRIDGUARD_SOURCE_REPOSITORY " +
                "or -SourceRepositoryPath."
            )
        }
        if (-not (Test-Path -LiteralPath $sourceRepositoryCandidate -PathType Container)) {
            throw "Trusted source repository is unavailable: $sourceRepositoryCandidate"
        }
        $sourceRepository = (Resolve-Path -LiteralPath $sourceRepositoryCandidate).Path
        if (-not (Test-Path -LiteralPath (Join-Path $sourceRepository ".git"))) {
            throw "Trusted source repository is not a Git repository: $sourceRepository"
        }

        $sourceCheckpointCandidate = $SourceCheckpointPath
        if ([string]::IsNullOrWhiteSpace($sourceCheckpointCandidate)) {
            $sourceCheckpointCandidate = [Environment]::GetEnvironmentVariable(
                "GRIDGUARD_SOURCE_CHECKPOINT", "Process"
            )
        }
        if ([string]::IsNullOrWhiteSpace($sourceCheckpointCandidate)) {
            throw (
                "strict source parity requires GRIDGUARD_SOURCE_CHECKPOINT " +
                "or -SourceCheckpointPath."
            )
        }
        if (-not (Test-Path -LiteralPath $sourceCheckpointCandidate -PathType Leaf)) {
            throw "Trusted source checkpoint is unavailable: $sourceCheckpointCandidate"
        }
        $sourceCheckpoint = (Resolve-Path -LiteralPath $sourceCheckpointCandidate).Path
    }

    New-Item -ItemType Directory -Path $releaseValidationDirectory -Force | Out-Null
    foreach ($transientPath in @(
        $testCollection,
        $testReport,
        $releaseArtifactReport,
        $sourceParityReport,
        $pairedReproduction,
        $robustnessReproduction
    )) {
        if (Test-Path -LiteralPath $transientPath) {
            Remove-Item -LiteralPath $transientPath -Force
        }
    }
    foreach ($releaseArchive in @($wheelArchive, $sourceArchive)) {
        if (Test-Path -LiteralPath $releaseArchive) {
            Remove-Item -LiteralPath $releaseArchive -Force
        }
    }

    $previousSourceRepository = [Environment]::GetEnvironmentVariable(
        "GRIDGUARD_SOURCE_REPOSITORY", "Process"
    )
    $previousSourceCheckpoint = [Environment]::GetEnvironmentVariable(
        "GRIDGUARD_SOURCE_CHECKPOINT", "Process"
    )

    Push-Location $projectRoot
    try {
        if ($StrictSourceParity) {
            [Environment]::SetEnvironmentVariable(
                "GRIDGUARD_SOURCE_REPOSITORY", $sourceRepository, "Process"
            )
            [Environment]::SetEnvironmentVariable(
                "GRIDGUARD_SOURCE_CHECKPOINT", $sourceCheckpoint, "Process"
            )
        }
        else {
            [Environment]::SetEnvironmentVariable(
                "GRIDGUARD_SOURCE_REPOSITORY", $null, "Process"
            )
            [Environment]::SetEnvironmentVariable(
                "GRIDGUARD_SOURCE_CHECKPOINT", $null, "Process"
            )
        }
        uv sync --locked --extra dev --extra docs --offline
        Assert-NativeCommandSucceeded -Operation "Locked dependency sync" -ExitCode $LASTEXITCODE
        $testedSourceIdentity = Get-ReleaseSourceTreeIdentity -ProjectRoot $projectRoot
        uv run --no-sync ruff check .
        Assert-NativeCommandSucceeded -Operation "Ruff check" -ExitCode $LASTEXITCODE
        uv run --no-sync ruff format --check .
        Assert-NativeCommandSucceeded -Operation "Ruff format check" -ExitCode $LASTEXITCODE
        uv run --no-sync gridguard-doctor
        Assert-NativeCommandSucceeded -Operation "GPU preflight" -ExitCode $LASTEXITCODE
        uv run --no-sync gridguard-evaluate --verify-evidence
        Assert-NativeCommandSucceeded -Operation "Evidence verification" -ExitCode $LASTEXITCODE
        uv run --no-sync gridguard-evaluate --reproduce-paired --output "tmp/release-validation/paired-benchmark-reproduced.json"
        Assert-NativeCommandSucceeded `
            -Operation "Byte-identical paired evidence reproduction" `
            -ExitCode $LASTEXITCODE
        uv run --no-sync gridguard-evaluate --reproduce-robustness --output "tmp/release-validation/robustness-supplement-reproduced.json"
        Assert-NativeCommandSucceeded `
            -Operation "Byte-identical robustness evidence reproduction" `
            -ExitCode $LASTEXITCODE
        Assert-NoAmbientPytestControls
        uv run --no-sync python tools/machine_validation.py --collect-pytest-inventory --collection-path "tmp/release-validation/release-tests.collection.json"
        Assert-NativeCommandSucceeded `
            -Operation "Exact release pytest collection" `
            -ExitCode $LASTEXITCODE
        if (-not (Test-Path -LiteralPath $testCollection -PathType Leaf)) {
            throw "Pytest collection inventory was not created: $testCollection"
        }
        uv run --no-sync pytest -q --junitxml "tmp/release-validation/release-tests.xml" -m "not source_archive and not release_artifact"
        $pytestExitCode = $LASTEXITCODE
        Assert-PytestJUnitPassedWithoutSkips `
            -ReportPath $testReport `
            -ExitCode $pytestExitCode `
            -Operation "Release source test suite"
        if ($StrictSourceParity) {
            Assert-NoAmbientPytestControls
            uv run --no-sync pytest -q --junitxml "tmp/release-validation/source-parity-tests.xml" -m "source_archive"
            $sourcePytestExitCode = $LASTEXITCODE
            Assert-PytestJUnitPassedWithoutSkips `
                -ReportPath $sourceParityReport `
                -ExitCode $sourcePytestExitCode `
                -Operation "Strict source-archive parity tests"
        }
        Assert-ReleaseSourceTreeIdentity `
            -ProjectRoot $projectRoot `
            -Identity $testedSourceIdentity
        $buildTaskId = [guid]::NewGuid().ToString("N")
        $candidateRoot = Join-Path $projectRoot ("tmp\release-candidate-" + $buildTaskId)
        $pyinstallerWorkRoot = Join-Path `
            $projectRoot `
            ("tmp\release-pyinstaller-work-" + $buildTaskId)
        $resolvedCandidate = [System.IO.Path]::GetFullPath($candidateRoot)
        $resolvedPyinstallerWork = [System.IO.Path]::GetFullPath($pyinstallerWorkRoot)
        $candidatePrefix = Join-Path $projectRoot "tmp\release-candidate-"
        $pyinstallerWorkPrefix = Join-Path $projectRoot "tmp\release-pyinstaller-work-"
        if (-not $resolvedCandidate.StartsWith(
            $candidatePrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Invalid release candidate path: $resolvedCandidate"
        }
        if (-not $resolvedPyinstallerWork.StartsWith(
            $pyinstallerWorkPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )) {
            throw "Invalid PyInstaller work path: $resolvedPyinstallerWork"
        }
        $resolvedCandidateTreePrefix = `
            $resolvedCandidate + [System.IO.Path]::DirectorySeparatorChar
        if (
            $resolvedPyinstallerWork.Equals(
                $resolvedCandidate,
                [System.StringComparison]::OrdinalIgnoreCase
            ) -or
            $resolvedPyinstallerWork.StartsWith(
                $resolvedCandidateTreePrefix,
                [System.StringComparison]::OrdinalIgnoreCase
            )
        ) {
            throw "PyInstaller work path must be outside the release candidate source tree."
        }
        $previousBytecodeSetting = [Environment]::GetEnvironmentVariable(
            "PYTHONDONTWRITEBYTECODE", "Process"
        )
        try {
            [void](New-VerifiedReleaseCandidateSource `
                -ProjectRoot $projectRoot `
                -DestinationPath $resolvedCandidate `
                -Identity $testedSourceIdentity)
            New-Item -ItemType Directory -Path $resolvedPyinstallerWork -Force | Out-Null
            [Environment]::SetEnvironmentVariable(
                "PYTHONDONTWRITEBYTECODE", "1", "Process"
            )
            uv run --project $projectRoot --directory $candidateRoot --no-sync pyinstaller --noconfirm --clean --distpath $projectRoot\dist --workpath $pyinstallerWorkRoot $candidateRoot\packaging\gridguard.spec
            Assert-NativeCommandSucceeded -Operation "PyInstaller build" -ExitCode $LASTEXITCODE
            uv build $candidateRoot --out-dir $projectRoot\dist --offline --no-build-isolation --no-create-gitignore --sdist --wheel
            Assert-NativeCommandSucceeded `
                -Operation "Offline wheel and source-distribution build" `
                -ExitCode $LASTEXITCODE
            Assert-ReleaseCandidateSourceIdentity `
                -CandidateRoot $candidateRoot `
                -Identity $testedSourceIdentity
            Assert-ReleaseSourceTreeIdentity `
                -ProjectRoot $projectRoot `
                -Identity $testedSourceIdentity
        }
        finally {
            [Environment]::SetEnvironmentVariable(
                "PYTHONDONTWRITEBYTECODE", $previousBytecodeSetting, "Process"
            )
            try {
                if (Test-Path -LiteralPath $pyinstallerWorkRoot) {
                    if (-not $resolvedPyinstallerWork.StartsWith(
                        $pyinstallerWorkPrefix,
                        [System.StringComparison]::OrdinalIgnoreCase
                    )) {
                        throw (
                            "Refusing to remove an invalid PyInstaller work path: " +
                            $resolvedPyinstallerWork
                        )
                    }
                    Remove-Item -LiteralPath $resolvedPyinstallerWork -Recurse -Force
                }
            }
            finally {
                if (Test-Path -LiteralPath $candidateRoot) {
                    if (-not $resolvedCandidate.StartsWith(
                        $candidatePrefix,
                        [System.StringComparison]::OrdinalIgnoreCase
                    )) {
                        throw (
                            "Refusing to remove an invalid release candidate path: " +
                            $resolvedCandidate
                        )
                    }
                    Remove-Item -LiteralPath $resolvedCandidate -Recurse -Force
                }
            }
        }
        foreach ($releaseArchive in @($wheelArchive, $sourceArchive)) {
            if (-not (Test-Path -LiteralPath $releaseArchive -PathType Leaf)) {
                throw "Expected release archive was not created: $releaseArchive"
            }
        }
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
    $distributionPolicy = Assert-ReleaseDistributionPolicy -DistributionPath $distribution
    Write-Host (
        "Release distribution policy: PASS " +
        "($($distributionPolicy.file_count) files, $($distributionPolicy.size_bytes) bytes)."
    )

    New-Item -ItemType Directory -Path $smokeRoot -Force | Out-Null
    if (Test-Path -LiteralPath $smokeReport) {
        Remove-Item -LiteralPath $smokeReport -Force
    }
    $smokeArguments = @(
        "--_gridguard-frozen-smoke-report",
        ('"{0}"' -f $smokeReport)
    )
    $smokeProcess = Start-Process `
        -FilePath $executable `
        -ArgumentList $smokeArguments `
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

$smokePayload = Get-ValidatedFrozenSmokePayload -ReportPath $smokeReport
Assert-CommittedReleaseValidationRecords `
    -ProjectRoot $projectRoot `
    -VerifyMachineInputs:(-not $VerifyOnly)
if ($RefreshManifest) {
    Update-ReleaseManifestBindings `
        -ProjectRoot $projectRoot `
        -ManifestPath $releaseManifestPath `
        -DistributionPath $distribution `
        -ExecutablePath $executable `
        -SmokeReportPath $smokeReport
    Write-Host "Refreshed RELEASE_MANIFEST.json from the verified build outputs."
}
Assert-ReleaseSourceSnapshot `
    -ProjectRoot $projectRoot `
    -ManifestPath $releaseManifestPath `
    -RequireCleanWorktree:$VerifyOnly
$releaseManifest = Assert-ReleaseManifestBinding `
    -ProjectRoot $projectRoot `
    -ManifestPath $releaseManifestPath `
    -DistributionPath $distribution `
    -ExecutablePath $executable `
    -SmokeReportPath $smokeReport `
    -SmokePayload $smokePayload

New-Item -ItemType Directory -Path $releaseValidationDirectory -Force | Out-Null
if (Test-Path -LiteralPath $releaseArtifactReport) {
    Remove-Item -LiteralPath $releaseArtifactReport -Force
}
Push-Location $projectRoot
try {
    Assert-NoAmbientPytestControls
    uv run --no-sync pytest -q --junitxml "tmp/release-validation/release-artifact-tests.xml" -m "release_artifact"
    $releaseArtifactExitCode = $LASTEXITCODE
    Assert-PytestJUnitPassedWithoutSkips `
        -ReportPath $releaseArtifactReport `
        -ExitCode $releaseArtifactExitCode `
        -Operation "Release artifact test suite"
}
finally {
    Pop-Location
}
[void](Assert-FinalReleaseReadiness `
    -ProjectRoot $projectRoot `
    -ReleaseManifest $releaseManifest)

if ($VerifyOnly) {
    [void](Assert-ReleaseDistributionPolicy -DistributionPath $distribution)
    Write-Host "Existing presentation package matches RELEASE_MANIFEST.json: $executable"
}
else {
    Write-Host "Presentation package ready: $executable"
    Write-Host "Frozen presentation smoke: PASS ($smokeReport)"
}
