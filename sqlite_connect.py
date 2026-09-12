import threading
from pathlib import Path

import numpy as np

from config import SQLITE_DB_PATH
from logger import get_logger
from model import FaceVectorModel

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS face_vector (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name   TEXT NOT NULL,
    count  INTEGER NOT NULL DEFAULT 1,
    vector BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_face_vector_name ON face_vector(name);
CREATE TABLE IF NOT EXISTS alias (
    alias TEXT PRIMARY KEY,
    name  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alias_name ON alias(name);
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
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()
        self._lock = threading.Lock()
        self._names: np.ndarray | None = None  # (N,) str
        self._matrix: np.ndarray | None = None  # (N, D) float32, row-normalized
        self._reload_cache()

    # ---- cache ----

    def _reload_cache(self):
        rows = self._conn.execute("SELECT name, vector FROM face_vector").fetchall()
        if rows:
            names = np.array([r[0] for r in rows])
            matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        else:
            names = np.array([], dtype=object)
            matrix = np.zeros((0, 0), dtype=np.float32)
        self._names = names
        self._matrix = matrix

    def _append_cache(self, name: str, vector: np.ndarray):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        if self._matrix.size == 0:
            self._matrix = vector.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, vector.reshape(1, -1)])
        self._names = np.append(self._names, name)

    # ---- write ops (both append one row; v2 has no mean-merge) ----

    def insert_one(self, data: FaceVectorModel):
        self._add_row(data.label, data.vector)

    def update_one(self, data: FaceVectorModel):
        """Register one more vector for the person (v2 replaces mean-merge)."""
        self._add_row(data.label, data.vector)

    def _add_row(self, name: str, vector):
        canonical = self.resolve_name(name.strip())
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT INTO face_vector (name, count, vector) VALUES (?, 1, ?)",
                (canonical, blob),
            )
            self._conn.commit()
            self._append_cache(canonical, np.asarray(vector, dtype=np.float32))
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
