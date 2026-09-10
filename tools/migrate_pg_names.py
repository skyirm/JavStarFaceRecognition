"""导出旧 PostgreSQL (pgvector) 中的 name 清单，并与 original_images 比对完整性。

用法（在能连到旧 PG 的机器上执行）：
  # 1) 从 PG 拉取 name 清单（凭据走环境变量或参数，避免硬编码）
  set PG_HOST=... PG_PORT=5432 PG_DB=... PG_USER=... PG_PASSWORD=...
  python tools/migrate_pg_names.py export --out pg_names.txt

  # 2) 与图片目录比对（可离线重跑）
  python tools/migrate_pg_names.py diff --names pg_names.txt --images original_images

结论：
  - "missing in images"：库里有人、缺注册图 → 按决策需补图后再建库
  - "extra in images"：有图但库里没人 → 新库会自然多收录
"""
import argparse
import os
import sys
from pathlib import Path


def cmd_export(args):
    import psycopg2

    conn = psycopg2.connect(
        host=args.host or os.environ["PG_HOST"],
        port=args.port or os.environ.get("PG_PORT", "5432"),
        dbname=args.db or os.environ["PG_DB"],
        user=args.user or os.environ["PG_USER"],
        password=args.password or os.environ["PG_PASSWORD"],
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT name FROM face_vector ORDER BY name")
            names = [row[0] for row in cur.fetchall()]
    finally:
        conn.close()

    Path(args.out).write_text("\n".join(names), encoding="utf-8")
    print(f"exported {len(names)} names -> {args.out}")


def cmd_diff(args):
    pg_names = [n for n in Path(args.names).read_text(encoding="utf-8").splitlines() if n.strip()]
    images = {p.name for p in Path(args.images).iterdir() if p.is_dir()}

    missing = sorted(set(pg_names) - images)
    extra = sorted(images - set(pg_names))

    print(f"pg names: {len(pg_names)}, image dirs: {len(images)}")
    print(f"\n[missing in images] 需补图后再建库（{len(missing)}）:")
    for n in missing:
        print(f"  - {n}")
    print(f"\n[extra in images] 新库将新增（{len(extra)}）:")
    for n in extra:
        print(f"  + {n}")

    Path(args.out or "missing_names.txt").write_text("\n".join(missing), encoding="utf-8")
    if missing:
        print(f"\nRESULT: INCOMPLETE - {len(missing)} person(s) need images (written to {args.out or 'missing_names.txt'})")
        return 1
    print("\nRESULT: COMPLETE - 可以直接重跑 add_faces.py 建库")
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="从 PG 导出 name 清单")
    p_exp.add_argument("--out", default="pg_names.txt")
    p_exp.add_argument("--host", default=None)
    p_exp.add_argument("--port", default=None)
    p_exp.add_argument("--db", default=None)
    p_exp.add_argument("--user", default=None)
    p_exp.add_argument("--password", default=None)
    p_exp.set_defaults(func=cmd_export)

    p_diff = sub.add_parser("diff", help="name 清单与图片目录比对")
    p_diff.add_argument("--names", required=True)
    p_diff.add_argument("--images", default="original_images")
    p_diff.add_argument("--out", default="missing_names.txt")
    p_diff.set_defaults(func=cmd_diff)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
