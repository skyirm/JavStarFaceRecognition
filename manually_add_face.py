import argparse
from pathlib import Path
from random import shuffle

from logger import get_logger
from model import FaceVectorModel
from sqlite_connect import SqliteConnection
from vector_operation import get_face_vector_from_file


def main():
    parser = argparse.ArgumentParser(description="按文件名包含的人名，从图库目录批量注册向量")
    parser.add_argument("--source", default=r"E:\国产专区", help="图片库根目录")
    parser.add_argument("--db", default=None, help="SQLite db 路径（默认 config.SQLITE_DB_PATH）")
    parser.add_argument("--max-per-person", type=int, default=60)
    args = parser.parse_args()

    logger = get_logger(__name__)
    db = SqliteConnection(args.db) if args.db else SqliteConnection()
    file_list = [f for f in Path(args.source).rglob("*.jpg")]

    while True:
        label = input("Enter face label (留空回车退出): ").strip()
        if not label:
            break

        shuffle(file_list)
        count = 0
        for file in file_list:
            if count >= args.max_per_person:
                break
            if label not in str(file):
                continue
            vector = get_face_vector_from_file(str(file))
            if vector is None:
                continue
            db.insert_one(FaceVectorModel(label=label, vector=vector, count=1))
            count += 1
            logger.info("Add face for %s from %s", label, file)

        if count == 0:
            logger.warning("No face found for label %s", label)
        else:
            logger.info("Added %d vectors for label %s", count, label)


if __name__ == "__main__":
    main()
