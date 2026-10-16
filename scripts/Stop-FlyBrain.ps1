[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
function Get-FlyBrainProcesses {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match '-m\s+flybrain_local\.server(?:\s|$)'
    })
}

# Stop both the Python launcher and its server child, including older launches.
$targets = @(Get-FlyBrainProcesses)
foreach ($target in $targets) {
    try { Stop-Process -Id $target.ProcessId -Force -ErrorAction Stop }
    catch {
        if (Get-Process -Id $target.ProcessId -ErrorAction SilentlyContinue) { throw }
    }
}
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    $remaining = @(Get-FlyBrainProcesses)
    if (-not $remaining.Count) { break }
    Start-Sleep -Milliseconds 250
}
if ($remaining.Count) { throw 'FlyBrain is still running. Shutdown could not be verified.' }
if (Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue) {
    throw 'FlyBrain processes stopped, but port 8765 still has a listener. Check the process using that port.'
}
Write-Output 'FlyBrain is stopped. The browser tab can now be closed.'
