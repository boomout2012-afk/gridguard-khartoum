$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$distribution = Join-Path $projectRoot "dist\GridGuard Khartoum"
$executable = Join-Path $distribution "GridGuard Khartoum.exe"
$manifestPath = Join-Path $projectRoot "RELEASE_MANIFEST.json"
$smokeReport = Join-Path $projectRoot "build\gridguard\frozen-smoke.json"
$releaseIntegrityHelper = Join-Path $PSScriptRoot "release-integrity.ps1"
$diagnosticPath = Join-Path $projectRoot "build\gridguard\launch-diagnostic.json"

if (-not (Test-Path -LiteralPath $releaseIntegrityHelper -PathType Leaf)) {
    throw "Release-integrity helper is unavailable: $releaseIntegrityHelper"
}
. $releaseIntegrityHelper

function Get-ApplicationStartupFailure {
    param([string]$DiagnosticPath)

    try {
        $diagnostic = Get-Content -Raw -LiteralPath $DiagnosticPath | ConvertFrom-Json
        if (
            $diagnostic.source -ceq "application" -and
            $diagnostic.stage -ceq "application-startup" -and
            $diagnostic.status -ceq "FAILED"
        ) {
            return $diagnostic
        }
    }
    catch {
        # An unavailable record cannot establish readiness or replace a process failure.
    }
    return $null
}

$previousStartupDiagnostic = [Environment]::GetEnvironmentVariable(
    "GRIDGUARD_STARTUP_DIAGNOSTIC", "Process"
)
$currentStage = "integrity-verification"
$process = $null
try {
    [Environment]::SetEnvironmentVariable(
        "GRIDGUARD_STARTUP_DIAGNOSTIC", $diagnosticPath, "Process"
    )
    [void](Write-LaunchDiagnostic `
        -DiagnosticPath $diagnosticPath `
        -Stage $currentStage `
        -Status "STARTED" `
        -ExecutablePath $executable `
        -Message "Verifying every release-bound file before process creation.")

    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "GridGuard is not built. Run scripts\build-presentation.ps1 first."
    }
    $smokePayload = Get-ValidatedFrozenSmokePayload -ReportPath $smokeReport
    [void](Assert-ReleaseManifestBinding `
        -ProjectRoot $projectRoot `
        -ManifestPath $manifestPath `
        -DistributionPath $distribution `
        -ExecutablePath $executable `
        -SmokeReportPath $smokeReport `
        -SmokePayload $smokePayload)

    $currentStage = "process-startup"
    [void](Write-LaunchDiagnostic `
        -DiagnosticPath $diagnosticPath `
        -Stage $currentStage `
        -Status "STARTED" `
        -ExecutablePath $executable `
        -Message "Integrity passed; waiting briefly for an early loader or DLL failure.")

    $process = Start-Process `
        -FilePath $executable `
        -WorkingDirectory $distribution `
        -PassThru
    $exitedDuringStartup = $process.WaitForExit(8000)
    $applicationFailure = Get-ApplicationStartupFailure -DiagnosticPath $diagnosticPath
    if ($null -ne $applicationFailure) {
        throw "GridGuard blocked startup: $($applicationFailure.message)"
    }
    if ($exitedDuringStartup) {
        if ($process.ExitCode -ne 0) {
            throw (
                "GridGuard exited during startup with code $($process.ExitCode). " +
                "This can indicate a pre-import, native DLL, CUDA, or driver failure."
            )
        }
        [void](Write-LaunchDiagnostic `
            -DiagnosticPath $diagnosticPath `
            -Stage $currentStage `
            -Status "EXITED" `
            -ExecutablePath $executable `
            -ExitCode $process.ExitCode `
            -Message "GridGuard exited cleanly during the startup observation window.")
    }
    else {
        # Only the application knows whether its trust gate passed and its window opened.
        # Leave its atomic READY/FAILED record, or the pending STARTED record, untouched.
        # A later application failure must remain visible after this observation ends.
        Write-Host "GridGuard startup observation complete. Diagnostic: $diagnosticPath"
    }
}
catch {
    $exitCode = if ($null -ne $process -and $process.HasExited) {
        [Nullable[int]]$process.ExitCode
    }
    else {
        $null
    }
    $applicationFailure = Get-ApplicationStartupFailure -DiagnosticPath $diagnosticPath
    if ($null -eq $applicationFailure) {
        [void](Write-LaunchDiagnostic `
            -DiagnosticPath $diagnosticPath `
            -Stage $currentStage `
            -Status "FAILED" `
            -ExecutablePath $executable `
            -Message $_.Exception.ToString() `
            -ExitCode $exitCode)
    }
    throw "GridGuard presentation launch failed. Diagnostic: $diagnosticPath`n$($_.Exception.Message)"
}
finally {
    [Environment]::SetEnvironmentVariable(
        "GRIDGUARD_STARTUP_DIAGNOSTIC", $previousStartupDiagnostic, "Process"
    )
}
