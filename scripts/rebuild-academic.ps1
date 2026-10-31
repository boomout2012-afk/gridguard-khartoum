param([switch]$InstallDependencies, [string]$PythonExecutable)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonExecutable) { $PythonExecutable = Join-Path $projectRoot '.venv/Scripts/python.exe' }
Push-Location -LiteralPath $projectRoot
try {
    if ($InstallDependencies) {
        & uv sync --locked --extra dev --extra docs
        if ($LASTEXITCODE -ne 0) { throw 'Locked Python setup failed.' }
        & uv pip install --python $PythonExecutable -r requirements-document-build.txt
        if ($LASTEXITCODE -ne 0) { throw 'Plotting dependencies failed.' }
        & npm install --prefix tmp/math-typesetting --no-audit --no-fund --save-exact mathjax-full@3.2.2 sharp@0.35.5
        if ($LASTEXITCODE -ne 0) { throw 'LaTeX rendering dependencies failed.' }
    }
    $env:GRIDGUARD_SHARP_PATH = Join-Path $projectRoot 'tmp/math-typesetting/node_modules/sharp'
    $trainingSha = (Get-FileHash -LiteralPath 'data/evidence/focused-ppo-training.json' -Algorithm SHA256).Hash.ToLower()
    $studySha = (Get-FileHash -LiteralPath 'data/evidence/focused-ppo-study.json' -Algorithm SHA256).Hash.ToLower()
    New-Item -ItemType Directory -Path 'tmp' -Force | Out-Null
    $steps = @(
        @('tools/export_training_telemetry.py'),
        @('tools/assemble_focused_thesis_content.py', '--training-sha256', $trainingSha, '--study-sha256', $studySha),
        @('tools/plot_focused_thesis.py'),
        @('tools/plot_training_diagnostics.py', '--render-only', '--training-sha256', $trainingSha),
        @('tools/build_focused_thesis.py'),
        @('tools/build_current_gui_pdf.py')
    )
    foreach ($step in $steps) {
        & $PythonExecutable @step
        if ($LASTEXITCODE -ne 0) { throw "Academic build failed: $($step[0])" }
    }
    Write-Host 'Thesis and GUI PDFs rebuilt. For slides use scripts/export-submission-presentation.ps1.'
} finally {
    Pop-Location
}
