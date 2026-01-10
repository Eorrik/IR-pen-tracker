$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "src"

python -m compileall src\kinect_pen
python -c "import pathlib; t = pathlib.Path('pyproject.toml').read_text(encoding='utf-8'); assert 'mediapipe' in t; print('pyproject ok')"
