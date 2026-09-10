import argparse
from pathlib import Path

from model import FaceVectorModel
from sqlite_connect import SqliteConnection
from vector_operation import get_face_vector_from_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def build_library(source: Path, db: SqliteConnection, max_per_person: int):
    total = 0
    for person_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        label = person_dir.name
        count = 0
        for file in sorted(person_dir.iterdir()):
            if count >= max_per_person:
                break
            if file.suffix.lower() not in IMAGE_EXTS:
                continue
            vector = get_face_vector_from_file(str(file))
            if vector is None:
                print(f"  skip (no single face): {file.name}")
                continue
            db.insert_one(FaceVectorModel(label=label, vector=vector, count=1))
            count += 1
        print(f"{label}: {count} vectors added")
        total += count
    print(f"done. {total} vectors for {len(list(source.iterdir()))} entries.")


def main():
    parser = argparse.ArgumentParser(description="从每人一个目录的图片重建人脸库（v2: 每图一条向量）")
    parser.add_argument("--source", default="original_images", help="图片根目录（默认 original_images）")
    parser.add_argument("--db", default=None, help="SQLite db 路径（默认 config.SQLITE_DB_PATH）")
    parser.add_argument("--max-per-person", type=int, default=60)
    args = parser.parse_args()

    db = SqliteConnection(args.db) if args.db else SqliteConnection()
    build_library(Path(args.source), db, args.max_per_person)


if __name__ == "__main__":
    main()
