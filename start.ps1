# Thin wrapper: all logic lives in start.py (cross-platform).
# Usage: ./start.ps1 [-SkipBuild] [-Port 7860] [-BindHost 0.0.0.0] [-Token <t>]
param()
uv run python (Join-Path $PSScriptRoot "start.py") @args
exit $LASTEXITCODE
