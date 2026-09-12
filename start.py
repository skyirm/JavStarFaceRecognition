"""Unified cross-platform startup script: build frontend -> start backend server.

Usage:
    uv run python start.py [--skip-build] [--port 7860] [--host 0.0.0.0] [--token TOKEN]

Works identically on Windows / Linux / macOS (start.bat / start.sh are thin wrappers).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd: list, cwd: Path | None = None) -> int:
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frontend and start backend server")
    parser.add_argument("--skip-build", action="store_true", help="skip frontend build")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--token", default="", help="ADMIN_TOKEN for management API")
    args, extra = parser.parse_known_args()  # extra args are forwarded to uvicorn

    if shutil.which("uv") is None:
        sys.exit("uv not found. Install it first: https://docs.astral.sh/uv/")

    dist = ROOT / "web" / "dist" / "index.html"
    if not args.skip_build:
        npm = shutil.which("npm")
        if npm is None:
            sys.exit("npm not found (Node is required for the frontend build), or use --skip-build")
        if not (ROOT / "web" / "node_modules").is_dir():
            print("Installing frontend dependencies...", flush=True)
            if run([npm, "install"], cwd=ROOT / "web") != 0:
                sys.exit("npm install failed")
        print("Building frontend...", flush=True)
        if run([npm, "run", "build"], cwd=ROOT / "web") != 0:
            sys.exit("frontend build failed")
        print("Frontend build done")
    elif not dist.is_file():
        sys.exit("web/dist not found. Run once without --skip-build to build the frontend")

    if args.token:
        os.environ["ADMIN_TOKEN"] = args.token
        print("ADMIN_TOKEN set: management API protected")
    elif os.environ.get("ADMIN_TOKEN"):
        print("ADMIN_TOKEN already set in environment: management API protected")
    else:
        print("WARNING: ADMIN_TOKEN not set, management API is UNPROTECTED")

    print(f"Starting server: http://localhost:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        code = run(
            [
                "uv", "run", "uvicorn", "app:app",
                "--host", args.host,
                "--port", str(args.port),
                *extra,
            ]
        )
    except KeyboardInterrupt:
        print("\nStopping server...")
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
