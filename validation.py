"""阈值扫描验证：以 validation_images/{人名}/xxx.jpg 为测试集，对 face_vector.db 扫描相似度阈值。

新语义：相似度越大越相似，predict = name if sim >= t else "Unknown"。
准确率 = (库内人员识别正确 + 库外人员判为 Unknown) / 总数。

可选 --baseline old.json（旧方案 glintr100+PG 的扫阈值结果），输出新旧对比表。
"""
import argparse
import json
from pathlib import Path

from sqlite_connect import SqliteConnection
from vector_operation import get_face_vector_from_file


def scan(db: SqliteConnection, image_root: Path, thresholds):
    # 预提取全部向量，避免每个阈值重复推理
    samples = []  # (true_name, best_name, best_sim)
    known = set(db.find_all_name())
    for file in sorted(image_root.rglob("*.jpg")):
        true_name = file.parent.name
        vector = get_face_vector_from_file(str(file))
        if vector is None:
            continue
        name, sim = db.find_most_similar(vector)
        samples.append((true_name, name, sim))
    print(f"evaluated {len(samples)} images ({len(known)} known persons)\n")

    rows = []
    for t in thresholds:
        right = 0
        for true_name, name, sim in samples:
            predict = name if sim >= t else "Unknown"
            if true_name in known and predict == true_name:
                right += 1
            elif true_name not in known and predict == "Unknown":
                right += 1
        acc = right / len(samples) if samples else 0.0
        rows.append((t, acc))
        print(f"threshold: {t:.2f}  accuracy: {acc:.4f}")
    best = max(rows, key=lambda r: r[1])
    print(f"\nbest threshold: {best[0]:.2f} (accuracy {best[1]:.4f})")
    return rows, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=None, help="SQLite db 路径（默认 config.SQLITE_DB_PATH）")
    parser.add_argument("--images", default="validation_images")
    parser.add_argument("--baseline", default=None, help="旧方案结果 json（threshold->accuracy）")
    args = parser.parse_args()

    db = SqliteConnection(args.db) if args.db else SqliteConnection()
    thresholds = [round(0.1 + 0.05 * i, 2) for i in range(19)]
    rows, best = scan(db, Path(args.images), thresholds)

    if args.baseline:
        old = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        print("\nthreshold   old(glintr100+PG)   new(AdaFace+SQLite)")
        for t, acc in rows:
            key = str(t)
            old_acc = old.get(key)
            old_str = f"{old_acc:.4f}" if old_acc is not None else "-"
            print(f"{t:>9.2f}   {old_str:>16}   {acc:>19.4f}")

    Path("validation_result.json").write_text(
        json.dumps({str(t): a for t, a in rows}, indent=2), encoding="utf-8"
    )
    print("\nsaved -> validation_result.json")


if __name__ == "__main__":
    main()
