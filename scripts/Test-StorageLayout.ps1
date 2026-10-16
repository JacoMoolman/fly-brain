[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Set-FlyBrainEnv.ps1"

$required = @(
    $env:FLYBRAIN_PROJECT_ROOT,
    $env:FLYBRAIN_ASSET_ROOT,
    $env:FLYBRAIN_DATASET_ROOT,
    $env:FLYGYM_DATA_DIR,
    $env:FLYGYM_ASSET_CACHE_DIR,
    $env:PIP_CACHE_DIR,
    $env:UV_CACHE_DIR,
    $env:UV_PYTHON_INSTALL_DIR,
    $env:TORCH_HOME,
    $env:CUDA_CACHE_PATH,
    $env:MPLCONFIGDIR,
    $env:PYTHONPYCACHEPREFIX,
    $env:NUMBA_CACHE_DIR,
    $env:XDG_CACHE_HOME,
    $env:TEMP,
    $env:TMP
)

foreach ($path in $required) {
    $resolved = [System.IO.Path]::GetFullPath($path)
    if ($resolved.StartsWith('C:\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Storage guard failed, path points to C drive: $resolved"
    }
    if (-not (Test-Path -LiteralPath $resolved)) {
        throw "Storage guard failed, path does not exist: $resolved"
    }
}

$jDrive = Get-PSDrive -Name J
$gDrive = Get-PSDrive -Name G
Write-Output ('Storage guard passed. G free: {0:N1} GB, J free: {1:N1} GB' -f ($gDrive.Free / 1GB), ($jDrive.Free / 1GB))
