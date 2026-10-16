[CmdletBinding()]
param(
    [switch]$SkipStats
)

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Set-FlyBrainEnv.ps1"

$destination = Join-Path $env:FLYBRAIN_DATASET_ROOT 'flat-connectome'
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$baseUrl = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome'
$files = @(
    [pscustomobject]@{
        LocalName = 'annotations.feather'
        SourceName = 'body-annotations-male-cns-v1.0-minconf-0.5.feather'
        Bytes = 14483314
        Sha256 = '2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2'
    },
    [pscustomobject]@{
        LocalName = 'neurotransmitters.feather'
        SourceName = 'body-neurotransmitters-male-cns-v1.0.feather'
        Bytes = 43282834
        Sha256 = '95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621'
    },
    [pscustomobject]@{
        LocalName = 'edges.feather'
        SourceName = 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
        Bytes = 1051241946
        Sha256 = 'e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1'
    }
)

if (-not $SkipStats) {
    $files += [pscustomobject]@{
        LocalName = 'body-stats.feather'
        SourceName = 'body-stats-male-cns-v1.0-minconf-0.5.feather'
        Bytes = 0
        Sha256 = $null
    }
}

$manifest = @()
foreach ($file in $files) {
    $target = Join-Path $destination $file.LocalName
    $url = "$baseUrl/$($file.SourceName)"
    if ((Test-Path -LiteralPath $target) -and $file.Bytes -gt 0 -and (Get-Item -LiteralPath $target).Length -eq $file.Bytes) {
        Write-Output "Already downloaded: $($file.LocalName)"
    } else {
        Write-Output "Downloading $($file.SourceName) to $target"
        & curl.exe --fail --location --retry 5 --retry-delay 3 --continue-at - --output $target $url
        if ($LASTEXITCODE -ne 0) {
            throw "Download failed for $url"
        }
    }

    $item = Get-Item -LiteralPath $target
    if ($file.Bytes -gt 0 -and $item.Length -ne $file.Bytes) {
        throw "Unexpected byte count for $target, expected $($file.Bytes), received $($item.Length)"
    }
    $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    if ($file.Sha256 -and $digest -ne $file.Sha256) {
        throw "SHA-256 mismatch for $target"
    }
    $manifest += [ordered]@{
        local_name = $file.LocalName
        source_name = $file.SourceName
        url = $url
        bytes = $item.Length
        sha256 = $digest
    }
}

$manifestPath = Join-Path $env:FLYBRAIN_DATASET_ROOT 'download-manifest.json'
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8
Write-Output "Verified MaleCNS files. Manifest: $manifestPath"
