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
"""


class SqliteConnection:
    """Local SQLite face vector store (v2: one person -> multiple vector rows).

    Similarity semantics: cosine similarity, higher = more similar.
    A (N, D) matrix cache is kept in memory; lookups are a single matmul,
    with per-name max reduction.
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
        logger.info("Insert vector for %s", data.label)

    def update_one(self, data: FaceVectorModel):
        """Register one more vector for the person (v2 replaces mean-merge)."""
        self._add_row(data.label, data.vector)
        logger.info("Append vector for %s", data.label)

    def _add_row(self, name: str, vector):
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT INTO face_vector (name, count, vector) VALUES (?, 1, ?)",
                (name, blob),
            )
            self._conn.commit()
            self._append_cache(name, np.asarray(vector, dtype=np.float32))

    def delete_one_by_name(self, name: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM face_vector WHERE name = ?", (name,))
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
        return {"name": row[0], "count": vec_count}

    def find_most_similar(self, vector) -> tuple[str | None, float]:
        """Return (name, max_cosine_similarity) over all vectors, grouped by name."""
        with self._lock:
            matrix, names = self._matrix, self._names
        if matrix is None or matrix.size == 0:
            return None, -1.0
        v = np.asarray(vector, dtype=np.float32)
        sims = matrix @ v
        uniq, inverse = np.unique(names, return_inverse=True)
        best = np.full(len(uniq), -np.inf, dtype=np.float32)
        np.maximum.at(best, inverse, sims)
        idx = int(np.argmax(best))
        return str(uniq[idx]), float(best[idx])

    def find_all_name(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT name FROM face_vector ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def find_all(self) -> list[dict]:
        """name -> aggregated vector count, for the management page."""
        rows = self._conn.execute(
            "SELECT name, COUNT(*) FROM face_vector GROUP BY name ORDER BY name"
        ).fetchall()
        return [{"name": r[0], "count": r[1]} for r in rows]

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
