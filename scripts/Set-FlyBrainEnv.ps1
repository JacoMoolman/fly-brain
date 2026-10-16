[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = 'G:\FLYBRAIN'
$AssetRoot = 'J:\models\fruit fly'

$paths = @(
    $ProjectRoot,
    $AssetRoot,
    "$AssetRoot\datasets\malecns-v1.0\flat-connectome",
    "$AssetRoot\datasets\malecns-v1.0\skeletons",
    "$AssetRoot\assets\flygym",
    "$AssetRoot\envs",
    "$AssetRoot\python",
    "$AssetRoot\cache\pip",
    "$AssetRoot\cache\uv",
    "$AssetRoot\cache\torch",
    "$AssetRoot\cache\cuda",
    "$AssetRoot\cache\matplotlib",
    "$AssetRoot\cache\python",
    "$AssetRoot\cache\numba",
    "$AssetRoot\cache\warp",
    "$AssetRoot\cache\xdg",
    "$AssetRoot\temp",
    "$AssetRoot\checkpoints",
    "$AssetRoot\recordings",
    "$AssetRoot\results",
    "$AssetRoot\logs"
)

foreach ($path in $paths) {
    if ($path.StartsWith('C:\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing project path on C drive: $path"
    }
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

$env:FLYBRAIN_PROJECT_ROOT = $ProjectRoot
$env:FLYBRAIN_ASSET_ROOT = $AssetRoot
$env:FLYBRAIN_DATASET_ROOT = "$AssetRoot\datasets\malecns-v1.0"
$env:FLYGYM_DATA_DIR = "$AssetRoot\assets\flygym"
$env:FLYGYM_ASSET_CACHE_DIR = "$AssetRoot\assets\flygym"
$env:PIP_CACHE_DIR = "$AssetRoot\cache\pip"
$env:UV_CACHE_DIR = "$AssetRoot\cache\uv"
$env:UV_PYTHON_INSTALL_DIR = "$AssetRoot\python"
$env:TORCH_HOME = "$AssetRoot\cache\torch"
$env:CUDA_CACHE_PATH = "$AssetRoot\cache\cuda"
$env:MPLCONFIGDIR = "$AssetRoot\cache\matplotlib"
$env:PYTHONPYCACHEPREFIX = "$AssetRoot\cache\python"
$env:NUMBA_CACHE_DIR = "$AssetRoot\cache\numba"
$env:WARP_CACHE_PATH = "$AssetRoot\cache\warp"
$env:XDG_CACHE_HOME = "$AssetRoot\cache\xdg"
$env:TEMP = "$AssetRoot\temp"
$env:TMP = "$AssetRoot\temp"
$env:TMPDIR = "$AssetRoot\temp"

Write-Output "FlyBrain project root: $ProjectRoot"
Write-Output "FlyBrain asset root:   $AssetRoot"
