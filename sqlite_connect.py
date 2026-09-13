import os
import sqlite3
import tempfile
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from config import RECOGNIZE_LOG_MAX, SQLITE_DB_PATH
from logger import get_logger
from model import FaceVectorModel

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS face_vector (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name   TEXT NOT NULL,
    count  INTEGER NOT NULL DEFAULT 1,
    vector BLOB NOT NULL,
    thumb  BLOB
);
CREATE INDEX IF NOT EXISTS idx_face_vector_name ON face_vector(name);
CREATE TABLE IF NOT EXISTS alias (
    alias TEXT PRIMARY KEY,
    name  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alias_name ON alias(name);
CREATE TABLE IF NOT EXISTS recognize_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    name       TEXT,
    similarity REAL NOT NULL,
    thumb      BLOB
);
CREATE INDEX IF NOT EXISTS idx_recognize_log_name ON recognize_log(name);
"""


class SqliteConnection:
    """Local SQLite face vector store (v2: one person -> multiple vector rows).

    Similarity semantics: cosine similarity, higher = more similar.
    A (N, D) matrix cache is kept in memory; lookups are a single matmul,
    with per-name max reduction.

    Aliases: one person may have multiple names. The `alias` table maps
    alias -> canonical name; vectors are always stored under the canonical
    name, so recognition needs no change (resolve at write time).
    """

    def __init__(self, db_path: str = SQLITE_DB_PATH):
        import sqlite3

        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()
        self._lock = threading.Lock()
        self._ids: np.ndarray | None = None  # (N,) rowids, parallel to matrix rows
        self._names: np.ndarray | None = None  # (N,) str
        self._matrix: np.ndarray | None = None  # (N, D) float32, row-normalized
        self._reload_cache()

    def _migrate(self):
        """Add columns introduced after v1; thumb = WebP preview of the aligned crop."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(face_vector)")}
        if "thumb" not in cols:
            self._conn.execute("ALTER TABLE face_vector ADD COLUMN thumb BLOB")

    # ---- cache ----

    def _reload_cache(self):
        rows = self._conn.execute("SELECT id, name, vector FROM face_vector").fetchall()
        if rows:
            self._ids = np.array([r[0] for r in rows], dtype=np.int64)
            names = np.array([r[1] for r in rows])
            matrix = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        else:
            self._ids = np.array([], dtype=np.int64)
            names = np.array([], dtype=object)
            matrix = np.zeros((0, 0), dtype=np.float32)
        self._names = names
        self._matrix = matrix

    def _append_cache(self, name: str, vector, row_id: int):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        if self._matrix.size == 0:
            self._matrix = vector.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, vector.reshape(1, -1)])
        self._names = np.append(self._names, name)
        self._ids = np.append(self._ids, row_id)

    # ---- write ops (both append one row; v2 has no mean-merge) ----

    def insert_one(self, data: FaceVectorModel):
        self._add_row(data.label, data.vector, data.thumb)

    def update_one(self, data: FaceVectorModel):
        """Register one more vector for the person (v2 replaces mean-merge)."""
        self._add_row(data.label, data.vector, data.thumb)

    def _add_row(self, name: str, vector, thumb: bytes | None = None):
        canonical = self.resolve_name(name.strip())
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO face_vector (name, count, vector, thumb) VALUES (?, 1, ?, ?)",
                (canonical, blob, thumb),
            )
            self._conn.commit()
            self._append_cache(canonical, np.asarray(vector, dtype=np.float32), cur.lastrowid)
        if canonical != name.strip():
            logger.info("Resolved %s -> %s", name.strip(), canonical)
        logger.info("Insert vector for %s", canonical)

    def delete_one_by_name(self, name: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM face_vector WHERE name = ?", (name,))
            self._conn.execute(
                "DELETE FROM alias WHERE name = ? OR alias = ?", (name, name)
            )
            self._conn.commit()
            removed = cur.rowcount
        if removed:
            self._reload_cache()
        logger.info("Delete %s (%d vectors)", name, removed)
        return removed

    def delete_vector_by_id(self, name: str, vector_id: int) -> bool:
        """Delete one vector row by id; `name` guard prevents cross-person deletion."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM face_vector WHERE id = ? AND name = ?", (vector_id, name)
            )
            self._conn.commit()
            removed = cur.rowcount
        if removed:
            self._reload_cache()
            logger.info("Delete vector #%d of %s", vector_id, name)
        return bool(removed)

    # ---- read ops ----

    def find_one_by_name(self, name: str) -> dict | None:
        row = self._conn.execute(
            "SELECT name, count FROM face_vector WHERE name = ? LIMIT 1", (name,)
        ).fetchone()
        if not row:
            return None
        vec_count = self._conn.execute(
            "SELECT COUNT(*) FROM face_vector WHERE name = ?", (name,)
        ).fetchone()[0]
        return {"name": row[0], "count": vec_count, "aliases": self.find_aliases(name)}

    def find_most_similar(self, vector) -> tuple[str | None, float]:
        """Return (name, max_cosine_similarity) over all vectors, grouped by name."""
        top = self.find_top_similar(vector, k=1)
        return top[0] if top else (None, -1.0)

    def find_top_similar(self, vector, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k [(name, max_cosine_similarity), ...], best first, grouped by name."""
        with self._lock:
            matrix, names = self._matrix, self._names
        if matrix is None or matrix.size == 0 or k <= 0:
            return []
        v = np.asarray(vector, dtype=np.float32)
        sims = matrix @ v
        uniq, inverse = np.unique(names, return_inverse=True)
        best = np.full(len(uniq), -np.inf, dtype=np.float32)
        np.maximum.at(best, inverse, sims)
        k = min(k, len(uniq))
        top = np.argsort(-best)[:k]
        return [(str(uniq[i]), float(best[i])) for i in top]

    def find_all_name(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT name FROM face_vector ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def find_all(self) -> list[dict]:
        """name -> aggregated vector count + aliases, for the management page."""
        rows = self._conn.execute(
            "SELECT name, COUNT(*) FROM face_vector GROUP BY name ORDER BY name"
        ).fetchall()
        aliases: dict[str, list[str]] = {}
        for name, alias in self._conn.execute("SELECT name, alias FROM alias"):
            aliases.setdefault(name, []).append(alias)
        return [
            {"name": name, "count": count, "aliases": aliases.get(name, [])}
            for name, count in rows
        ]

    def find_page(self, q: str = "", page: int = 1, page_size: int = 50) -> dict:
        """Paginated person list with search (name or alias, case-insensitive)."""
        page = max(1, page)
        page_size = max(1, min(page_size, 200))
        rows = self._conn.execute(
            "SELECT name, COUNT(*) FROM face_vector GROUP BY name ORDER BY name"
        ).fetchall()
        aliases: dict[str, list[str]] = {}
        for name, alias in self._conn.execute("SELECT name, alias FROM alias"):
            aliases.setdefault(name, []).append(alias)
        needle = q.strip().lower()
        if needle:
            rows = [
                (name, count)
                for name, count in rows
                if needle in name.lower()
                or any(needle in a.lower() for a in aliases.get(name, []))
            ]
        total = len(rows)
        start = (page - 1) * page_size
        items = [
            {"name": name, "count": count, "aliases": aliases.get(name, [])}
            for name, count in rows[start : start + page_size]
        ]
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    def stats(self) -> dict:
        """Aggregate counts for monitoring."""
        vectors = self._conn.execute("SELECT COUNT(*) FROM face_vector").fetchone()[0]
        persons = self._conn.execute(
            "SELECT COUNT(DISTINCT name) FROM face_vector"
        ).fetchone()[0]
        return {"persons": persons, "vectors": vectors}

    def find_thumbs(self, name: str) -> list[dict]:
        """Per-vector preview thumbs: [{"id": rowid, "thumb": webp bytes | None}, ...]."""
        rows = self._conn.execute(
            "SELECT id, thumb FROM face_vector WHERE name = ? ORDER BY id", (name,)
        ).fetchall()
        return [{"id": r[0], "thumb": r[1]} for r in rows]

    # ---- recognition history ----

    def log_recognition(self, name: str | None, similarity: float, thumb: bytes | None):
        """Append one recognize result (name=None for Unknown) with FIFO pruning."""
        ts = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                "INSERT INTO recognize_log (ts, name, similarity, thumb) VALUES (?, ?, ?, ?)",
                (ts, name, similarity, thumb),
            )
            self._conn.execute(
                "DELETE FROM recognize_log WHERE id <= "
                "(SELECT id FROM recognize_log ORDER BY id DESC LIMIT 1 OFFSET ?)",
                (RECOGNIZE_LOG_MAX,),
            )
            self._conn.commit()

    def find_history(
        self, q: str = "", unknown_only: bool = False, page: int = 1, page_size: int = 50
    ) -> dict:
        page = max(1, page)
        page_size = max(1, min(page_size, 200))
        where, args = [], []
        if unknown_only:
            where.append("name IS NULL")
        elif q.strip():
            where.append("name LIKE ?")
            args.append(f"%{q.strip()}%")
        tail = (" WHERE " + " AND ".join(where)) if where else ""
        total = self._conn.execute(
            "SELECT COUNT(*) FROM recognize_log" + tail, args
        ).fetchone()[0]
        rows = self._conn.execute(
            "SELECT id, ts, name, similarity, thumb FROM recognize_log"
            + tail
            + " ORDER BY id DESC LIMIT ? OFFSET ?",
            [*args, page_size, (page - 1) * page_size],
        ).fetchall()
        return {
            "items": [
                {"id": r[0], "ts": r[1], "name": r[2], "similarity": r[3], "thumb": r[4]}
                for r in rows
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    def delete_history(self, record_id: int | None = None) -> int:
        with self._lock:
            if record_id is None:
                cur = self._conn.execute("DELETE FROM recognize_log")
            else:
                cur = self._conn.execute(
                    "DELETE FROM recognize_log WHERE id = ?", (record_id,)
                )
            self._conn.commit()
            return cur.rowcount

    # ---- library health (duplicate / outlier suggestions) ----

    def find_issues(
        self, merge_threshold: float, outlier_threshold: float, max_per_type: int = 50
    ) -> dict:
        """Cross-name high-similarity pairs + same-name outlier vectors.

        Uses the in-memory cache; pair enumeration is chunked to bound memory.
        """
        with self._lock:
            matrix, names, ids = self._matrix, self._names, self._ids
        if matrix is None or matrix.size == 0:
            return {"cross_name": [], "outliers": []}
        n = matrix.shape[0]

        cross_name: list[dict] = []
        for start in range(0, n, 1024):
            block = matrix[start : start + 1024]
            sims = block @ matrix.T
            for bi in range(block.shape[0]):
                i = start + bi
                row = sims[bi]
                cand = np.where(row >= merge_threshold)[0]
                cand = cand[cand > i]
                for j in cand:
                    if names[i] != names[j]:
                        cross_name.append(
                            {
                                "a": int(ids[i]),
                                "a_name": str(names[i]),
                                "b": int(ids[j]),
                                "b_name": str(names[j]),
                                "sim": round(float(row[j]), 4),
                            }
                        )
        cross_name.sort(key=lambda p: -p["sim"])
        cross_name = cross_name[:max_per_type]

        outliers: list[dict] = []
        for name in np.unique(names):
            idx = np.where(names == name)[0]
            if len(idx) < 2:
                continue
            sub = matrix[idx]
            mean = sub.mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm == 0:
                continue
            sims = sub @ (mean / norm)
            for k, s in zip(idx, sims):
                if s < outlier_threshold:
                    outliers.append(
                        {
                            "id": int(ids[k]),
                            "name": str(name),
                            "sim_to_mean": round(float(s), 4),
                        }
                    )
        outliers.sort(key=lambda o: o["sim_to_mean"])
        outliers = outliers[:max_per_type]
        return {"cross_name": cross_name, "outliers": outliers}

    # ---- aliases ----

    def resolve_name(self, name: str) -> str:
        """Return the canonical name if `name` is a registered alias, else itself."""
        row = self._conn.execute(
            "SELECT name FROM alias WHERE alias = ?", (name,)
        ).fetchone()
        return row[0] if row else name

    def add_alias(self, alias: str, name: str):
        """Register `alias` as an additional name of person `name`."""
        alias, name = alias.strip(), name.strip()
        if not alias:
            raise ValueError("别名不能为空")
        if alias == name:
            raise ValueError("别名不能与主名相同")
        owner = self.resolve_name(alias)
        if owner != alias:
            raise ValueError(f"别名「{alias}」已被「{owner}」使用")
        if self._name_exists(alias):
            raise ValueError(f"「{alias}」已是其他人员的主名，请改用合并功能")
        if not self._name_exists(name):
            raise ValueError(f"人员「{name}」不存在")
        with self._lock:
            self._conn.execute(
                "INSERT INTO alias (alias, name) VALUES (?, ?)", (alias, name)
            )
            self._conn.commit()
        logger.info("Add alias %s -> %s", alias, name)

    def remove_alias(self, alias: str, name: str | None = None) -> bool:
        with self._lock:
            if name is None:
                cur = self._conn.execute("DELETE FROM alias WHERE alias = ?", (alias,))
            else:
                cur = self._conn.execute(
                    "DELETE FROM alias WHERE alias = ? AND name = ?", (alias, name)
                )
            self._conn.commit()
            removed = cur.rowcount
        if removed:
            logger.info("Remove alias %s", alias)
        return bool(removed)

    def find_aliases(self, name: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT alias FROM alias WHERE name = ? ORDER BY alias", (name,)
        ).fetchall()
        return [r[0] for r in rows]

    def merge_person(self, from_name: str, into_name: str) -> int:
        """Move all vectors of `from_name` into `into_name`, keeping it as alias."""
        from_name, into_name = from_name.strip(), into_name.strip()
        if from_name == into_name:
            raise ValueError("不能合并到自身")
        resolved_from = self.resolve_name(from_name)
        if resolved_from != from_name:
            raise ValueError(f"「{from_name}」已是「{resolved_from}」的别名，无需合并")
        resolved_into = self.resolve_name(into_name)
        if resolved_into != into_name:
            raise ValueError(f"「{into_name}」是「{resolved_into}」的别名，请使用主名")
        if not self._name_exists(from_name):
            raise ValueError(f"人员「{from_name}」不存在")
        if not self._name_exists(into_name):
            raise ValueError(f"人员「{into_name}」不存在")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE face_vector SET name = ? WHERE name = ?", (into_name, from_name)
            )
            moved = cur.rowcount
            self._conn.execute(
                "INSERT OR REPLACE INTO alias (alias, name) VALUES (?, ?)",
                (from_name, into_name),
            )
            self._conn.commit()
        self._reload_cache()
        logger.info("Merge %s -> %s (%d vectors)", from_name, into_name, moved)
        return moved

    def _name_exists(self, name: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM face_vector WHERE name = ? LIMIT 1", (name,)
            ).fetchone()
            is not None
        )

    # ---- library export / import (machine migration) ----

    REQUIRED_COLUMNS = {"id", "name", "count", "vector"}

    def export_summary(self) -> dict:
        persons = self._conn.execute(
            "SELECT COUNT(DISTINCT name) FROM face_vector"
        ).fetchone()[0]
        vectors = self._conn.execute("SELECT COUNT(*) FROM face_vector").fetchone()[0]
        aliases = self._conn.execute("SELECT COUNT(*) FROM alias").fetchone()[0]
        return {"persons": persons, "vectors": vectors, "aliases": aliases}

    def snapshot_bytes(self) -> bytes:
        """Consistent snapshot of the whole database via the SQLite backup API.

        Covers face_vector (vectors + WebP thumbs), alias, and recognize_log.
        """
        fd, tmp = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            with self._lock:
                dst = sqlite3.connect(tmp)
                try:
                    self._conn.backup(dst)
                finally:
                    dst.close()
            return Path(tmp).read_bytes()
        finally:
            Path(tmp).unlink(missing_ok=True)

    def import_database(self, path: str) -> dict:
        """Replace library contents (vectors + aliases) with those of the db at `path`.

        Row ids are preserved, so thumbs/ids referenced elsewhere stay stable.
        recognize_log is NOT touched. Raises ValueError on an invalid file.
        """
        try:
            src = sqlite3.connect(path)
            try:
                cols = {r[1] for r in src.execute("PRAGMA table_info(face_vector)")}
                if not self.REQUIRED_COLUMNS <= cols:
                    raise ValueError("不是有效的人脸库导出文件（缺少 face_vector 表）")
                n_vectors = src.execute("SELECT COUNT(*) FROM face_vector").fetchone()[0]
                n_aliases = src.execute("SELECT COUNT(*) FROM alias").fetchone()[0]
                n_persons = src.execute(
                    "SELECT COUNT(DISTINCT name) FROM face_vector"
                ).fetchone()[0]
            finally:
                src.close()
        except sqlite3.DatabaseError as e:
            raise ValueError(f"不是有效的 SQLite 数据库文件: {e}") from e

        with self._lock:
            self._conn.execute("ATTACH DATABASE ? AS import_src", (path,))
            try:
                self._conn.execute("DELETE FROM face_vector")
                self._conn.execute("DELETE FROM alias")
                if "thumb" in cols:
                    self._conn.execute(
                        "INSERT INTO main.face_vector (id, name, count, vector, thumb) "
                        "SELECT id, name, count, vector, thumb FROM import_src.face_vector"
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO main.face_vector (id, name, count, vector) "
                        "SELECT id, name, count, vector FROM import_src.face_vector"
                    )
                self._conn.execute(
                    "INSERT INTO main.alias (alias, name) "
                    "SELECT alias, name FROM import_src.alias"
                )
                self._conn.execute(
                    "UPDATE sqlite_sequence SET seq = COALESCE("
                    "(SELECT MAX(id) FROM main.face_vector), 0) WHERE name = 'face_vector'"
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                self._conn.execute("DETACH DATABASE import_src")
        self._reload_cache()
        logger.info(
            "Imported library: %d persons / %d vectors / %d aliases",
            n_persons,
            n_vectors,
            n_aliases,
        )
        return {"persons": n_persons, "vectors": n_vectors, "aliases": n_aliases}

    def check_connection(self, func):
        """Compat no-op: local SQLite needs no reconnect logic."""
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        db = SqliteConnection(str(Path(td) / "test.db"))
        db.insert_one(FaceVectorModel(label="alice", vector=[1.0, 0.0, 0.0], count=1))
        db.update_one(FaceVectorModel(label="alice", vector=[0.0, 1.0, 0.0], count=1))
        db.insert_one(FaceVectorModel(label="bob", vector=[0.9, 0.1, 0.0], count=1))
        print(db.find_all())
        print(db.find_most_similar([1.0, 0.0, 0.0]))
        db.delete_one_by_name("alice")
        print(db.find_all_name(), db.find_most_similar([1.0, 0.0, 0.0]))
