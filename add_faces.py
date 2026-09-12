import argparse
import re
from pathlib import Path

from model import FaceVectorModel
from sqlite_connect import SqliteConnection
from vector_operation import get_face_vector_from_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# 文件夹名支持 "主名,别名1,别名2"（全/半角逗号或分号分隔）
NAME_SEP = re.compile(r"[,，;；]")


def register_person(db: SqliteConnection, raw_label: str, files, max_count: int, replace: bool) -> int:
    names = [n.strip() for n in NAME_SEP.split(raw_label) if n.strip()]
    primary = db.resolve_name(names[0])

    if replace:
        if primary == names[0]:
            removed = db.delete_one_by_name(primary)
            if removed:
                print(f"  replace: removed {removed} old vectors")
        else:
            print(f"  note: {names[0]} is an alias of {primary}, append only")

    count = 0
    for file in files:
        if count >= max_count:
            break
        vector = get_face_vector_from_file(str(file))
        if vector is None:
            print(f"  skip (no single face): {file.name}")
            continue
        db.insert_one(FaceVectorModel(label=primary, vector=vector, count=1))
        count += 1

    # 其余名字并入主名：已有向量则合并，没有则仅登记别名
    for name in names[1:]:
        if db.resolve_name(name) != name:
            print(f"  note: {name} is already an alias, skipped")
            continue
        if db.find_one_by_name(name):
            moved = db.merge_person(name, primary)
            print(f"  merged {name} -> {primary} ({moved} vectors)")
        else:
            db.add_alias(name, primary)

    return count


def build_library(source: Path, db: SqliteConnection, max_per_person: int, replace: bool):
    total = 0
    for person_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        label = person_dir.name
        files = sorted(f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        count = register_person(db, label, files, max_per_person, replace)
        print(f"{label}: {count} vectors added")
        total += count
    print(f"done. {total} vectors for {sum(1 for p in source.iterdir() if p.is_dir())} entries.")


def main():
    parser = argparse.ArgumentParser(description="从每人一个目录的图片建人脸库（目录名=人名，支持 '主名,别名'）")
    parser.add_argument("--source", default="original_images", help="图片根目录（默认 original_images）")
    parser.add_argument("--db", default=None, help="SQLite db 路径（默认 config.SQLITE_DB_PATH）")
    parser.add_argument("--max-per-person", type=int, default=60)
    parser.add_argument("--replace", action="store_true", help="先删除该人已有向量再入库（避免重复追加）")
    args = parser.parse_args()

    db = SqliteConnection(args.db) if args.db else SqliteConnection()
    build_library(Path(args.source), db, args.max_per_person, args.replace)


if __name__ == "__main__":
    main()
