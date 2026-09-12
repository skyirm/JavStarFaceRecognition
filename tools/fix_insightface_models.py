"""Fix insightface model pack layout so *.onnx files sit directly in <root>/models/<name>/.

Problem this solves: unzipping antelopev2.zip often creates an extra directory level,
e.g. ~/.insightface/models/antelopev2/antelopev2/det_10g.onnx, which makes
FaceAnalysis fail with "AssertionError: assert 'detection' in self.models".

The script flattens nested directories, can extract <name>.zip placed in the
models directory, and verifies the detection model (det_10g.onnx) is in place.

Usage:
    uv run python tools/fix_insightface_models.py
    uv run python tools/fix_insightface_models.py --name buffalo_l --root /root/.insightface
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

DOWNLOAD_URL = "https://github.com/deepinsight/insightface/releases/download/models/{name}.zip"


def collect_onnx(target: Path) -> list[Path]:
    return sorted(target.rglob("*.onnx")) if target.is_dir() else []


def extract_pack(zip_path: Path, target: Path) -> None:
    """Extract zip into target; if the zip has a single top folder, use its contents."""
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td)
        entries = list(Path(td).iterdir())
        src = entries[0] if len(entries) == 1 and entries[0].is_dir() else Path(td)
        target.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            shutil.move(str(f), str(target / f.name))


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 insightface 模型包目录（去掉多余嵌套/解压 zip）")
    parser.add_argument("--name", default="antelopev2", help="模型包名（默认 antelopev2）")
    parser.add_argument("--root", default="~/.insightface", help="insightface 根目录（默认 ~/.insightface）")
    args = parser.parse_args()

    models_dir = Path(args.root).expanduser() / "models"
    target = models_dir / args.name

    onnx_files = collect_onnx(target)
    if not onnx_files:
        zip_path = models_dir / f"{args.name}.zip"
        if zip_path.is_file():
            print(f"Extracting {zip_path} -> {target}")
            extract_pack(zip_path, target)
            onnx_files = collect_onnx(target)
        elif not target.is_dir():
            sys.exit(
                f"模型包不存在：{target}\n"
                f"请下载 {DOWNLOAD_URL.format(name=args.name)} 后把 zip 放到 {models_dir}/"
                f"（或解压到 {target}/），再运行本脚本"
            )

    if not list(target.glob("*.onnx")) and onnx_files:
        print(f"检测到嵌套目录，将 .onnx 上移到 {target} ...")
        for f in onnx_files:
            dst = target / f.name
            if dst.exists():
                print(f"  skip（已存在同名文件）: {f.relative_to(target)}")
                continue
            shutil.move(str(f), dst)
            print(f"  moved: {f.relative_to(target)} -> {f.name}")
        for d in sorted({f.parent for f in onnx_files if f.parent != target},
                        key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
                print(f"  removed empty dir: {d.relative_to(target)}")
            except OSError:
                pass

    final = list(target.glob("*.onnx"))
    if not final:
        sys.exit(f"{target} 下没有任何 .onnx 文件，模型包不完整，请重新下载")

    det = target / "det_10g.onnx"
    if det.is_file():
        print(f"OK: 检测模型就位 -> {det}")
    else:
        print(f"WARNING: 未找到 det_10g.onnx（检测模型），当前文件：{[f.name for f in final]}")


if __name__ == "__main__":
    main()
