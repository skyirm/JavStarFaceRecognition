import numpy as np
import pytest

from model import FaceVectorModel
from sqlite_connect import SqliteConnection


@pytest.fixture()
def db(tmp_path):
    return SqliteConnection(str(tmp_path / "test.db"))


def _v(*components, dim=512):
    vec = np.zeros(dim, dtype=np.float32)
    vec[: len(components)] = components
    return vec


def test_insert_and_find(db):
    db.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1))
    db.update_one(FaceVectorModel(label="alice", vector=_v(0.0, 1.0), count=1))
    db.insert_one(FaceVectorModel(label="bob", vector=_v(0.9, 0.1), count=1))

    assert db.find_all_name() == ["alice", "bob"]
    assert db.find_all() == [
        {"name": "alice", "count": 2, "aliases": []},
        {"name": "bob", "count": 1, "aliases": []},
    ]
    assert db.find_one_by_name("alice") == {"name": "alice", "count": 2, "aliases": []}
    assert db.find_one_by_name("nobody") is None


def test_find_most_similar_group_max(db):
    # alice: two rows; query close to alice's second row -> alice with its max sim
    db.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1))
    db.insert_one(FaceVectorModel(label="alice", vector=_v(0.0, 1.0), count=1))
    db.insert_one(FaceVectorModel(label="bob", vector=_v(0.9, 0.1), count=1))

    name, sim = db.find_most_similar(_v(0.0, 1.0))
    assert name == "alice"
    assert sim == pytest.approx(1.0, abs=1e-6)

    name, sim = db.find_most_similar(_v(1.0))
    assert name == "alice"
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_delete_removes_all_rows_and_updates_cache(db):
    db.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1))
    db.insert_one(FaceVectorModel(label="alice", vector=_v(0.0, 1.0), count=1))
    db.insert_one(FaceVectorModel(label="bob", vector=_v(1.0), count=1))

    assert db.delete_one_by_name("alice") == 2
    assert db.delete_one_by_name("alice") == 0
    assert db.find_all_name() == ["bob"]
    name, sim = db.find_most_similar(_v(0.0, 1.0))
    assert name == "bob" and sim == pytest.approx(0.0, abs=1e-6)


def test_persistence_and_cache_reload(tmp_path):
    path = str(tmp_path / "persist.db")
    db1 = SqliteConnection(path)
    db1.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1))
    db1.insert_one(FaceVectorModel(label="bob", vector=_v(0.0, 1.0), count=1))
    del db1

    db2 = SqliteConnection(path)
    assert db2.find_all_name() == ["alice", "bob"]
    name, sim = db2.find_most_similar(_v(0.0, 1.0))
    assert name == "bob" and sim == pytest.approx(1.0, abs=1e-6)


def test_empty_db(db):
    assert db.find_all_name() == []
    assert db.find_most_similar(_v(1.0)) == (None, -1.0)


def test_unnormalized_input_is_normalized(db):
    db.insert_one(FaceVectorModel(label="alice", vector=_v(3.0), count=1))
    name, sim = db.find_most_similar(_v(1.0))
    assert name == "alice" and sim == pytest.approx(1.0, abs=1e-5)


def test_delete_vector_by_id(db):
    db.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1))
    db.insert_one(FaceVectorModel(label="alice", vector=_v(0.0, 1.0), count=1))
    db.insert_one(FaceVectorModel(label="bob", vector=_v(1.0), count=1))

    assert db.delete_vector_by_id("alice", 1) is True
    assert db.find_one_by_name("alice")["count"] == 1

    # name guard: cannot delete another person's row by id, or a nonexistent id
    assert db.delete_vector_by_id("alice", 3) is False
    assert db.delete_vector_by_id("bob", 999) is False
    assert db.find_one_by_name("bob")["count"] == 1

    # cache reflects the deletion
    name, sim = db.find_most_similar(_v(1.0))
    assert name == "bob" and sim == pytest.approx(1.0, abs=1e-6)


def test_history_log_filter_and_prune(db, monkeypatch):
    monkeypatch.setattr("sqlite_connect.RECOGNIZE_LOG_MAX", 3)
    for i in range(5):
        db.log_recognition(
            "alice" if i % 2 == 0 else None,
            0.9 if i % 2 == 0 else -1.0,
            b"w" if i == 3 else None,
        )
    r = db.find_history()
    assert r["total"] == 3  # FIFO pruned to RECOGNIZE_LOG_MAX
    ids = [i["id"] for i in r["items"]]
    assert ids == sorted(ids, reverse=True) and ids[0] == 5
    assert r["items"][1]["thumb"] == b"w"  # id 4
    assert db.find_history(unknown_only=True)["total"] == 1
    assert db.find_history(q="ali")["total"] == 2
    assert db.find_history(q="nobody")["total"] == 0

    assert db.delete_history(4) == 1
    assert db.delete_history(4) == 0
    assert db.delete_history() == 2
    assert db.find_history()["total"] == 0


def test_find_issues(db):
    e0 = _v(1.0)
    e1 = _v(0.0, 1.0)
    e2 = _v(0.0, 0.0, 1.0)
    e3 = _v(0.0, 0.0, 0.0, 1.0)
    mid = (e0 + e1) / np.linalg.norm(e0 + e1)
    db.insert_one(FaceVectorModel(label="alice", vector=e0, count=1))
    db.insert_one(FaceVectorModel(label="alice", vector=mid, count=1))
    db.insert_one(FaceVectorModel(label="alice", vector=e2, count=1))
    db.insert_one(FaceVectorModel(label="bob", vector=e2, count=1))
    db.insert_one(FaceVectorModel(label="carol", vector=e3, count=1))

    issues = db.find_issues(merge_threshold=0.6, outlier_threshold=0.5)

    # cross-name: alice's e2 vector vs bob's identical vector (sim=1.0)
    pairs = [
        (p["a_name"], p["b_name"], p["a"], p["b"], p["sim"])
        for p in issues["cross_name"]
    ]
    assert len(pairs) == 1
    a_name, b_name, a, b, sim = pairs[0]
    assert {a_name, b_name} == {"alice", "bob"} and sim >= 0.99
    assert isinstance(a, int) and isinstance(b, int)

    # outlier: alice's e2 vector is far from alice's mean direction
    out = [(o["name"], o["sim_to_mean"], o["id"]) for o in issues["outliers"]]
    assert len(out) == 1 and out[0][0] == "alice" and out[0][1] < 0.5

    # single-vector persons produce nothing
    assert db.find_issues(merge_threshold=0.99, outlier_threshold=0.01)["outliers"] == []


def test_library_export_import_roundtrip(tmp_path):
    db1 = SqliteConnection(str(tmp_path / "a.db"))
    db1.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1, thumb=b"w1"))
    db1.insert_one(FaceVectorModel(label="alice", vector=_v(0.0, 1.0), count=1))
    db1.insert_one(FaceVectorModel(label="bob", vector=_v(0.0, 0.0, 1.0), count=1, thumb=b"w2"))
    db1.add_alias("爱丽丝", "alice")
    payload = db1.snapshot_bytes()
    assert payload[:16].startswith(b"SQLite format 3")

    db2 = SqliteConnection(str(tmp_path / "b.db"))
    db2.insert_one(FaceVectorModel(label="junk", vector=_v(0.9, 0.1), count=1))
    db2.add_alias("垃圾别名", "junk")

    dump = tmp_path / "dump.db"
    dump.write_bytes(payload)
    summary = db2.import_database(str(dump))
    assert summary == {"persons": 2, "vectors": 3, "aliases": 1}

    # old contents fully replaced
    assert {p["name"] for p in db2.find_all()} == {"alice", "bob"}
    assert db2.find_one_by_name("alice")["count"] == 2
    assert db2.find_thumbs("alice")[0]["thumb"] == b"w1"
    assert db2.find_thumbs("bob")[0]["thumb"] == b"w2"
    assert db2.resolve_name("爱丽丝") == "alice"

    # recognition works on the imported cache
    name, sim = db2.find_most_similar(_v(1.0))
    assert name == "alice" and sim == pytest.approx(1.0, abs=1e-6)

    # id continuity: next insert continues after imported max id (3)
    db2.insert_one(FaceVectorModel(label="carol", vector=_v(0.0, 0.0, 0.0, 1.0), count=1))
    assert db2.find_thumbs("carol")[0]["id"] == 4


def test_import_rejects_invalid_file(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"this is not a sqlite database" * 10)
    db = SqliteConnection(str(tmp_path / "lib.db"))
    with pytest.raises(ValueError):
        db.import_database(str(bad))
    assert db.find_all_name() == []  # library untouched after failed import


def test_stats(db):
    db.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1))
    db.insert_one(FaceVectorModel(label="alice", vector=_v(0.0, 1.0), count=1))
    db.insert_one(FaceVectorModel(label="bob", vector=_v(1.0), count=1))
    assert db.stats() == {"persons": 2, "vectors": 3}
    db.delete_one_by_name("alice")
    assert db.stats() == {"persons": 1, "vectors": 1}


def test_find_page_search_and_paginate(db):
    db.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1))
    db.insert_one(FaceVectorModel(label="bob", vector=_v(0.0, 1.0), count=1))
    db.insert_one(FaceVectorModel(label="carol", vector=_v(1.0), count=1))
    db.add_alias("卡罗尔", "carol")

    r = db.find_page(page_size=2)
    assert r["total"] == 3 and len(r["items"]) == 2 and r["page"] == 1
    assert db.find_page(page=2, page_size=2)["page"] == 2

    r = db.find_page(q="BOB")
    assert r["total"] == 1 and r["items"][0]["name"] == "bob"
    r = db.find_page(q="卡罗尔")
    assert r["total"] == 1 and r["items"][0]["name"] == "carol"
    assert db.find_page(q="nobody")["total"] == 0

    # clamping
    assert db.find_page(page=0)["page"] == 1
    assert db.find_page(page_size=999)["page_size"] == 200


def test_thumb_storage_and_find(db):
    thumb = b"fake-webp-bytes"
    db.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1, thumb=thumb))
    db.insert_one(FaceVectorModel(label="alice", vector=_v(0.0, 1.0), count=1))

    items = db.find_thumbs("alice")
    assert [i["id"] for i in items] == [1, 2]
    assert items[0]["thumb"] == thumb
    assert items[1]["thumb"] is None
    assert db.find_thumbs("nobody") == []

    # thumbs travel with rows on merge/delete
    db.insert_one(FaceVectorModel(label="bob", vector=_v(1.0), count=1, thumb=thumb))
    db.merge_person("bob", "alice")
    assert db.find_thumbs("alice")[2]["thumb"] == thumb
    db.delete_one_by_name("alice")
    assert db.find_thumbs("alice") == []


def test_migration_adds_thumb_column(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS face_vector ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  name TEXT NOT NULL,"
        "  count INTEGER NOT NULL DEFAULT 1,"
        "  vector BLOB NOT NULL"
        ");"
    )
    conn.execute(
        "INSERT INTO face_vector (name, count, vector) VALUES ('alice', 1, ?)",
        (b"\x00" * 2048,),
    )
    conn.commit()
    conn.close()

    db = SqliteConnection(path)
    assert db.find_all_name() == ["alice"]
    assert db.find_thumbs("alice")[0]["thumb"] is None
    db.insert_one(FaceVectorModel(label="alice", vector=_v(1.0), count=1, thumb=b"x"))
    assert db.find_thumbs("alice")[1]["thumb"] == b"x"
