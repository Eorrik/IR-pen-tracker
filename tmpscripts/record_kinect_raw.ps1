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
$scriptPath = Join-Path $projRoot "src\kinect_pen\tools\record_kinect_raw.py"

if (-not (Test-Path $scriptPath)) {
  Fail "脚本不存在: $scriptPath"
}

uv run --no-project --with "pyk4a>=1.5.0" --with "opencv-python>=4.8.0" --with "numpy>=1.24.0" python "$scriptPath"
