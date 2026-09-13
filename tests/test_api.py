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


def _zip_without_db() -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no database here")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_list_faces_empty(client):
    r = client.get("/api/faces").json()
    assert r["items"] == [] and r["total"] == 0


def test_health(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok"
    assert r["models"] == {"detector": True, "adaface": True}
    assert set(r["library"]) == {"persons", "vectors"}
    assert r["uptime_seconds"] >= 0


def test_upload_and_list_and_suggest(client):
    r = client.post(
        "/api/faces", params={"name": "测试人物"}, files={"file": ("a.jpg", _img_bytes("天使もえ", 0))}
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"name": "测试人物", "count": 1, "aliases": []}

    faces = client.get("/api/faces").json()["items"]
    assert {"name": "测试人物", "count": 1, "aliases": []} in faces

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
    assert client.get("/api/faces").json()["total"] == 0


def test_faces_pagination_and_search(client):
    for name in ("张三", "李四", "王五"):
        r = client.post(
            "/api/faces", params={"name": name}, files={"file": ("a.jpg", _img_bytes("天使もえ", 0))}
        )
        assert r.status_code == 200, r.text
    client.post("/api/faces/王五/aliases", json={"alias": "王五别名"})

    r = client.get("/api/faces", params={"page_size": 2}).json()
    assert r["total"] == 3 and len(r["items"]) == 2 and r["page"] == 1
    r = client.get("/api/faces", params={"page": 2, "page_size": 2}).json()
    assert r["page"] == 2 and len(r["items"]) == 1
    r = client.get("/api/faces", params={"q": "李四"}).json()
    assert r["total"] == 1 and r["items"][0]["name"] == "李四"
    r = client.get("/api/faces", params={"q": "王五别名"}).json()
    assert r["total"] == 1 and r["items"][0]["name"] == "王五"
    r = client.get("/api/faces", params={"q": "不存在的名字"}).json()
    assert r["total"] == 0 and r["items"] == []

    for name in ("张三", "李四", "王五"):
        assert client.delete(f"/api/faces/{name}").status_code == 200
    assert client.get("/api/faces").json()["total"] == 0


def test_thumbs_after_upload(client):
    r = client.post(
        "/api/faces", params={"name": "测试人物"}, files={"file": ("a.jpg", _img_bytes("天使もえ", 0))}
    )
    assert r.status_code == 200, r.text

    thumbs = client.get("/api/faces/测试人物/thumbs").json()
    assert len(thumbs) == 1
    assert thumbs[0]["thumb"].startswith("data:image/webp;base64,")

    assert client.delete("/api/faces/测试人物").status_code == 200
    assert client.get("/api/faces/测试人物/thumbs").json() == []
    assert client.get("/api/faces").json()["total"] == 0


def test_delete_single_vector(client):
    r = client.post(
        "/api/faces", params={"name": "测试人物"}, files={"file": ("a.jpg", _img_bytes("天使もえ", 0))}
    )
    assert r.status_code == 200, r.text
    vid = client.get("/api/faces/测试人物/thumbs").json()[0]["id"]

    assert client.delete("/api/faces/测试人物/vectors/9999").status_code == 404
    r = client.delete(f"/api/faces/测试人物/vectors/{vid}")
    assert r.status_code == 200
    assert r.json() == {
        "deleted": vid,
        "name": "测试人物",
        "remaining": 0,
        "person_gone": True,
    }
    assert client.get("/api/faces/测试人物/thumbs").json() == []
    assert client.get("/api/faces").json()["total"] == 0


def test_recognition_history(client):
    client.post(
        "/api/faces", params={"name": "测试人物"}, files={"file": ("a.jpg", _img_bytes("天使もえ", 0))}
    )
    assert client.post("/api/recognize", files={"file": ("a.jpg", _img_bytes("天使もえ", 1))}).status_code == 200
    assert client.post("/api/recognize", files={"file": ("b.jpg", _img_bytes("早坂ひめ", 0))}).status_code == 200

    h = client.get("/api/history").json()
    assert h["total"] == 2
    ids = [i["id"] for i in h["items"]]
    assert ids == sorted(ids, reverse=True)
    assert {i["name"] for i in h["items"]} == {None, "测试人物"}
    for i in h["items"]:
        assert i["thumb"] is None or i["thumb"].startswith("data:image/webp;base64,")

    unknown = client.get("/api/history", params={"unknown": "true"}).json()
    assert unknown["total"] == 1 and unknown["items"][0]["name"] is None

    first = h["items"][0]["id"]
    assert client.delete(f"/api/history/{first}").status_code == 200
    assert client.delete(f"/api/history/{first}").status_code == 404
    assert client.delete("/api/history").json()["deleted"] == 1
    assert client.get("/api/history").json()["total"] == 0

    assert client.delete("/api/faces/测试人物").status_code == 200
    assert client.get("/api/faces").json()["total"] == 0


def test_faces_issues_endpoint(client):
    # empty library -> no issues
    r = client.get("/api/faces/issues").json()
    assert r == {"cross_name": [], "outliers": []}


def test_library_export_import(client):
    client.post(
        "/api/faces", params={"name": "测试人物"}, files={"file": ("a.jpg", _img_bytes("天使もえ", 0))}
    )
    r = client.get("/api/library/export")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    assert r.content[:2] == b"PK"
    zip_bytes = r.content

    # wiping the library then importing restores it
    assert client.delete("/api/faces/测试人物").status_code == 200
    assert client.get("/api/faces").json()["total"] == 0
    r = client.post("/api/library/import", files={"file": ("lib.zip", zip_bytes)})
    assert r.status_code == 200
    assert r.json() == {"persons": 1, "vectors": 1, "aliases": 0}
    faces = client.get("/api/faces").json()["items"]
    assert [f["name"] for f in faces] == ["测试人物"] and faces[0]["count"] == 1
    thumbs = client.get("/api/faces/测试人物/thumbs").json()
    assert thumbs[0]["thumb"].startswith("data:image/webp;base64,")

    # recognize still works after import
    r = client.post("/api/recognize", files={"file": ("a.jpg", _img_bytes("天使もえ", 1))})
    assert r.json()[0]["name"] == "测试人物"

    # invalid uploads rejected
    r = client.post("/api/library/import", files={"file": ("bad.zip", b"not a zip")})
    assert r.status_code == 400
    r = client.post(
        "/api/library/import",
        files={"file": ("empty.zip", _zip_without_db())},
    )
    assert r.status_code == 400

    assert client.delete("/api/faces/测试人物").status_code == 200
    assert client.delete("/api/history").status_code == 200
