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
    assert db.find_all() == [{"name": "alice", "count": 2}, {"name": "bob", "count": 1}]
    assert db.find_one_by_name("alice") == {"name": "alice", "count": 2}
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
