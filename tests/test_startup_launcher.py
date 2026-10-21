from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("application_status", ["FAILED", "READY", "PENDING"])
def test_launcher_preserves_application_diagnostics_and_never_infers_readiness(
    tmp_path: Path,
    application_status: str,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1] / "scripts" / "run-presentation.ps1"
    shutil.copyfile(source, scripts / source.name)
    executable = tmp_path / "dist" / "GridGuard Khartoum" / "GridGuard Khartoum.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    helper = r"""
function Write-LaunchDiagnostic {
    param($DiagnosticPath, $Stage, $Status, $ExecutablePath, $Message, $ExitCode)
    [void](New-Item -ItemType Directory -Path (Split-Path $DiagnosticPath) -Force)
    @{ stage=$Stage; status=$Status; message=$Message } |
        ConvertTo-Json | Set-Content -LiteralPath $DiagnosticPath -Encoding UTF8
}
function Get-ValidatedFrozenSmokePayload { param($ReportPath); return @{} }
function Assert-ReleaseManifestBinding {
    param($ProjectRoot, $ManifestPath, $DistributionPath, $ExecutablePath,
          $SmokeReportPath, $SmokePayload)
}
function Start-Process {
    param($FilePath, $WorkingDirectory, [switch]$PassThru)
    if ('APPLICATION_STATUS' -ne 'PENDING') {
        @{ schema_version=1; source='application'; stage='application-startup';
           status='APPLICATION_STATUS'; message='CUDA unavailable';
           error_type='GpuPreflightError'; cause=@{message='driver failure'} } |
            ConvertTo-Json |
                Set-Content -LiteralPath $env:GRIDGUARD_STARTUP_DIAGNOSTIC -Encoding UTF8
    }
    $process = [pscustomobject]@{ HasExited=$false; ExitCode=$null }
    $process | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value { param($ms); $false }
    return $process
}
""".replace("APPLICATION_STATUS", application_status)
    (scripts / "release-integrity.ps1").write_text(helper, encoding="utf-8")
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    assert powershell is not None, "Windows presentation tests require PowerShell"
    result = subprocess.run(
        [powershell, "-NoProfile", "-File", str(scripts / source.name)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    payload = json.loads(
        (tmp_path / "build" / "gridguard" / "launch-diagnostic.json").read_text(
            encoding="utf-8-sig"
        )
    )
    if application_status == "FAILED":
        assert result.returncode != 0
        assert payload["status"] == "FAILED"
        assert payload["source"] == "application"
        assert payload["error_type"] == "GpuPreflightError"
        assert payload["cause"] == {"message": "driver failure"}
        assert "CUDA unavailable" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert payload["status"] == ("READY" if application_status == "READY" else "STARTED")
