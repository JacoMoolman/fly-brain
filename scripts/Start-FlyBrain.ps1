[CmdletBinding()]
param(
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$url = "http://127.0.0.1:$Port"
$existing = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -match '-m\s+flybrain_local\.server(?:\s|$)'
})
if ($existing.Count) {
    Write-Output 'FlyBrain is already running. Use STOP-FLY-BRAIN.cmd to stop it.'
    if (-not $NoBrowser) { Start-Process $url }
    exit 0
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is in use by another process. No new server was started."
}
. "$PSScriptRoot\Set-FlyBrainEnv.ps1"
Set-Location -LiteralPath $env:FLYBRAIN_PROJECT_ROOT

$python = Join-Path $env:FLYBRAIN_ASSET_ROOT 'envs\flybrain\Scripts\python.exe'
$graph = Join-Path $env:FLYBRAIN_ASSET_ROOT 'prepared\malecns-v1.0\brain_graph.npz'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Python environment is missing. Run .\scripts\Setup-Python.ps1 first.'
}
if (-not (Test-Path -LiteralPath $graph)) {
    throw "Prepared MaleCNS graph is missing: $graph"
}

$url = "http://127.0.0.1:$Port"
if (-not $NoBrowser) {
    Start-Job -ScriptBlock {
        param($target)
        Start-Sleep -Seconds 7
        Start-Process $target
    } -ArgumentList $url | Out-Null
}

Write-Output "Starting the integrated MaleCNS and FlyGym GUI at $url"
Write-Output 'Keep this window open. Use STOP-FLY-BRAIN.cmd or press Ctrl+C here to stop.'
& $python -m flybrain_local.server --port $Port
exit $LASTEXITCODE
