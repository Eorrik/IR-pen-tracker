Param()
$ErrorActionPreference = "Stop"

$env:HTTP_PROXY = "http://127.0.0.1:10809"
$env:HTTPS_PROXY = "http://127.0.0.1:10809"

function Fail($msg, $code=1) {
  Write-Host $msg
  exit $code
}

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
  Fail "未找到 uv，请先安装 https://github.com/astral-sh/uv"
}

$projRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projRoot

$env:PYTHONPATH = "src"

$py = Join-Path $projRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Fail "未找到虚拟环境 Python: $py"
}

uv pip install --python "$py" "pytest>=7.4.0" "numpy>=1.24.0" "opencv-python>=4.8.0" "mediapipe>=0.10.0"

& "$py" -c "import pkgutil, mediapipe as mp, pathlib; root = pathlib.Path(mp.__file__).resolve().parent; task_files = sorted([str(p.relative_to(root)) for p in root.rglob('*.task')])[:50]; print('mediapipe:', getattr(mp,'__version__',None), mp.__file__); print('has solutions:', hasattr(mp,'solutions')); print('submodules:', [m.name for m in pkgutil.iter_modules(mp.__path__)][:30]); print('task_files_sample:', task_files)"

& "$py" -m pytest -q
