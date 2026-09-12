# Thin wrapper: all logic lives in start.py (cross-platform).
# Usage: ./start.ps1 [--skip-build] [--port 7860] [--host 0.0.0.0] [--token <t>]
param()
uv run python (Join-Path $PSScriptRoot "start.py") @args
exit $LASTEXITCODE
