import argparse
import re
from pathlib import Path
from random import shuffle

from logger import get_logger
from model import FaceVectorModel
from sqlite_connect import SqliteConnection
from vector_operation import get_face_data_from_file


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
        raw = input("Enter face label (支持逗号分隔多个名字，留空回车退出): ").strip()
        if not raw:
            break
        names = [n for n in re.split(r"[,，]", raw) if n.strip()]
        if not names:
            continue

        primary = db.resolve_name(names[0].strip())
        counts = {}
        for name in dict.fromkeys([primary] + [n.strip() for n in names[1:]]):
            if name != primary and db.resolve_name(name) != name:
                logger.warning("%s 已是「%s」的别名，跳过", name, db.resolve_name(name))
                continue
            shuffle(file_list)
            count = 0
            for file in file_list:
                if count >= args.max_per_person:
                    break
                if name not in str(file):
                    continue
                data = get_face_data_from_file(str(file))
                if data is None:
                    continue
                db.insert_one(
                    FaceVectorModel(label=name, vector=data[0], count=1, thumb=data[1])
                )
                count += 1
                logger.info("Add face for %s from %s", name, file)
            counts[name] = count
            if count == 0:
                logger.warning("No face found for label %s", name)
            else:
                logger.info("Added %d vectors for label %s", count, name)

        # 其余名字并入主名：有向量则合并（向量随之迁移），没有则仅登记别名
        for name in counts:
            if name == primary:
                continue
            if db.find_one_by_name(name):
                moved = db.merge_person(name, primary)
                logger.info("Merged %s into %s (%d vectors)", name, primary, moved)
            else:
                db.add_alias(name, primary)


if __name__ == "__main__":
    main()
