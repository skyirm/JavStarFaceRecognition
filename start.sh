#!/usr/bin/env bash
# Thin wrapper: all logic lives in start.py (cross-platform).
exec uv run python "$(dirname "$0")/start.py" "$@"
