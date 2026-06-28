param(
    [string]$Python = "python",
    [string]$Venv = ".venv-win",
    [string]$Extras = "dev,opt,plot,perf,bench",
    [switch]$NoVerify
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$VenvPath = if ([System.IO.Path]::IsPathRooted($Venv)) {
    $Venv
} else {
    Join-Path $Root $Venv
}

Write-Host "Creating or updating PythonSpinDynamics environment:"
Write-Host "  root:   $Root"
Write-Host "  venv:   $VenvPath"
Write-Host "  python: $Python"
Write-Host "  extras: $Extras"

Push-Location $Root
try {
    if (-not (Test-Path -LiteralPath $VenvPath)) {
        & $Python -m venv $VenvPath
    }

    $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -e ".[${Extras}]"

    if (-not $NoVerify) {
        & $VenvPython scripts\verify_dev_env.py --strict
    }

    Write-Host ""
    $ActivatePath = Join-Path $VenvPath "Scripts\Activate.ps1"
    Write-Host "Activate with:"
    Write-Host "  & `"$ActivatePath`""
    Write-Host ""
    Write-Host "Run smoke checks with:"
    Write-Host "  python -m unittest tests.smoke_tests"
    Write-Host "  python -m ruff check src tests examples"
}
finally {
    Pop-Location
}
