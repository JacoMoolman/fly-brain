[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Set-FlyBrainEnv.ps1"

$python = Join-Path $env:FLYBRAIN_ASSET_ROOT 'envs\flybrain\Scripts\python.exe'
$viewer = Join-Path $env:FLYBRAIN_PROJECT_ROOT 'vendor\flygym\scripts\launch_interactive_viewer.py'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'FlyGym environment is missing. Run .\scripts\Setup-Python.ps1 first.'
}

Write-Output 'Starting the official FlyGym native MuJoCo viewer.'
& $python $viewer
