$ErrorActionPreference = "Stop"

Write-Host "Running ruff..."
uv run --with ruff ruff check src tests

Write-Host "Running mypy..."
# Ignoring missing imports for cv2, mediapipe etc as stubs might be missing
uv run --with mypy --with numpy --with types-setuptools mypy src tests --ignore-missing-imports
