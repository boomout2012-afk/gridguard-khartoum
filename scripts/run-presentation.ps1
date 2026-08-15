$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$executable = Join-Path $projectRoot "dist\GridGuard Khartoum\GridGuard Khartoum.exe"

if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
    throw "GridGuard is not built. Run scripts\build-presentation.ps1 first."
}

Start-Process -FilePath $executable -WorkingDirectory (Split-Path -Parent $executable)
