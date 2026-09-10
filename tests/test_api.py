import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ["SQLITE_DB_PATH"] = str(Path(tempfile.mkdtemp()) / "api_test.db")

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

IMG_ROOT = Path(__file__).parent.parent / "validation_images"


def _img_bytes(person: str, index: int = 0) -> bytes:
    files = sorted((IMG_ROOT / person).glob("*.jpg"))
    return files[index].read_bytes()


def _noise_bytes() -> bytes:
    rng = np.random.RandomState(0)
    ok, buf = cv2.imencode(".jpg", rng.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def _multi_face_bytes() -> bytes:
    a = cv2.imdecode(np.frombuffer(_img_bytes("天使もえ", 0), np.uint8), 1)
    b = cv2.imdecode(np.frombuffer(_img_bytes("早坂ひめ", 0), np.uint8), 1)
    h = min(a.shape[0], b.shape[0])
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    ok, buf = cv2.imencode(".jpg", np.hstack([a, b]))
    assert ok
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_list_faces_empty(client):
    assert client.get("/api/faces").json() == []


def test_upload_and_list_and_suggest(client):
    r = client.post(
        "/api/faces", params={"name": "测试人物"}, files={"file": ("a.jpg", _img_bytes("天使もえ", 0))}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"name": "测试人物", "count": 1}

    faces = client.get("/api/faces").json()
    assert {"name": "测试人物", "count": 1} in faces

    r = client.get("/api/faces/suggest", params={"q": "测试"})
    assert "测试人物" in r.json()["suggestions"]


def test_recognize_registered_person(client):
    r = client.post("/api/recognize", files={"file": ("a.jpg", _img_bytes("天使もえ", 1))})
    assert r.status_code == 200, r.text
    hits = r.json()
    assert len(hits) == 1
    assert hits[0]["name"] == "测试人物"
    assert hits[0]["similarity"] > 0.5
    x1, y1, x2, y2 = hits[0]["bbox"]
    assert x2 > x1 and y2 > y1


def test_recognize_unknown_person(client):
    r = client.post("/api/recognize", files={"file": ("b.jpg", _img_bytes("早坂ひめ", 0))})
    assert r.status_code == 200, r.text
    hits = r.json()
    assert hits[0]["name"] == "Unknown"


def test_upload_rejections(client):
    r = client.post("/api/faces", params={"name": ""}, files={"file": ("a.jpg", _img_bytes("天使もえ", 0))})
    assert r.status_code == 400
    r = client.post("/api/faces", params={"name": "x"}, files={"file": ("n.jpg", _noise_bytes())})
    assert r.status_code == 400
    r = client.post("/api/faces", params={"name": "x"}, files={"file": ("m.jpg", _multi_face_bytes())})
    assert r.status_code == 400
    assert "多个" in r.json()["detail"]


def test_compare(client):
    r = client.post(
        "/api/compare",
        files={"file1": ("a.jpg", _img_bytes("天使もえ", 0)), "file2": ("b.jpg", _img_bytes("天使もえ", 2))},
    )
    same = r.json()["similarity"]
    assert same > 0.5
    r = client.post(
        "/api/compare",
        files={"file1": ("a.jpg", _img_bytes("天使もえ", 0)), "file2": ("b.jpg", _img_bytes("早坂ひめ", 1))},
    )
    diff = r.json()["similarity"]
    assert diff < same
    print(f"compare same={same:.4f} diff={diff:.4f}")


def test_delete_face(client):
    assert client.delete("/api/faces/测试人物").status_code == 200
    assert client.delete("/api/faces/测试人物").status_code == 404
    assert client.get("/api/faces").json() == []
