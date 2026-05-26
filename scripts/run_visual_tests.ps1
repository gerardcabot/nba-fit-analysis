# Run all NBA Fit visual validation tests from repo root.
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONPATH = $RepoRoot

Write-Host "==> visual_tests/01_endpoint_health.py"
python visual_tests/01_endpoint_health.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> visual_tests/02_pbp_schema.py"
python visual_tests/02_pbp_schema.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All visual tests finished. Figures: reports/figures/"
