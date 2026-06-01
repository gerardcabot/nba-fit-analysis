# Create external data layout for NBA Fit (Windows).
# Usage (current session):
#   $env:NBA_FIT_DATA_ROOT = 'E:\Nba Fit Data'
#   .\scripts\init_nba_fit_data_root.ps1

$ErrorActionPreference = "Stop"
if (-not $env:NBA_FIT_DATA_ROOT) {
    $env:NBA_FIT_DATA_ROOT = "E:\Nba Fit Data"
    Write-Host "NBA_FIT_DATA_ROOT not set; defaulting to $($env:NBA_FIT_DATA_ROOT)"
}
python "$PSScriptRoot\init_nba_fit_data_root.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Set permanently (User): [Environment]::SetEnvironmentVariable('NBA_FIT_DATA_ROOT', '$($env:NBA_FIT_DATA_ROOT)', 'User')"
