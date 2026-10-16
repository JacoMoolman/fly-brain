[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Set-FlyBrainEnv.ps1"
$python = Join-Path $env:FLYBRAIN_ASSET_ROOT 'envs\flybrain\Scripts\python.exe'
& $python "$env:FLYBRAIN_PROJECT_ROOT\scripts\verify_integrated.py"
if ($LASTEXITCODE -ne 0) { throw 'Integrated verification failed.' }

