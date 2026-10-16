[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\Set-FlyBrainEnv.ps1"

$venvRoot = Join-Path $env:FLYBRAIN_ASSET_ROOT 'envs\flybrain'
$python = Join-Path $venvRoot 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    Write-Output 'Installing a project-local Python 3.13 runtime on J drive.'
    & uv python install 3.13
    if ($LASTEXITCODE -ne 0) { throw 'uv could not install Python 3.13.' }
    & uv venv --python 3.13 $venvRoot
    if ($LASTEXITCODE -ne 0) { throw 'uv could not create the FlyBrain environment.' }
}

Write-Output 'Installing FlyGym, the local brain-body application, and reinforcement-learning support.'
& uv pip install --python $python --editable "$env:FLYBRAIN_PROJECT_ROOT\vendor\flygym[examples,rl,warp]" --editable $env:FLYBRAIN_PROJECT_ROOT pyarrow psutil pytest
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }

& $python -c "from importlib.metadata import version; import flygym, mujoco, numpy, pyarrow, fastapi, torch; print('flygym', version('flygym')); print('mujoco', mujoco.__version__); print('numpy', numpy.__version__); print('pyarrow', pyarrow.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Installed Python environment failed its import check.' }

Write-Output "Python environment ready: $python"
