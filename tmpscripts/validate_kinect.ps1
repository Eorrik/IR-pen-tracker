Param()
$ErrorActionPreference = "Stop"

function Fail($msg, $code=1) {
  Write-Host $msg
  exit $code
}

$env:HTTP_PROXY = "http://127.0.0.1:10809"
$env:HTTPS_PROXY = "http://127.0.0.1:10809"

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
  Fail "未找到 uv，请先安装 https://github.com/astral-sh/uv"
}

$projRoot = Split-Path $PSScriptRoot -Parent
$scriptPath = Join-Path $projRoot "src\kinect_pen\tools\validate_kinect.py"

if (-not (Test-Path $scriptPath)) {
  Fail "脚本不存在: $scriptPath"
}

# 避免项目依赖冲突，仅为该脚本按需引入 pyk4a
uv run --no-project --with "pyk4a>=1.5.0" python "$scriptPath"
