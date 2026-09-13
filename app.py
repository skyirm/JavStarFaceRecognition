import asyncio
import base64
import io
import json
import os
import secrets
import tempfile
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path

import cv2
import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Response,
    Security,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    CLUSTER_MERGE_THRESHOLD,
    CLUSTER_OUTLIER_THRESHOLD,
    FACE_DETECT_THRESHOLD,
    RECOGNIZE_TOP_K,
    SIMILARITY_THRESHOLD,
    SQLITE_DB_PATH,
)
from logger import get_logger
from model import FaceVectorModel
from sqlite_connect import SqliteConnection
from vector_operation import (
    adaface,
    compare_vector,
    detector,
    get_face_data_from_faces,
    get_result_from_array,
    get_vectors_from_faces,
)

logger = get_logger(__name__)

MAX_IMAGE_EDGE = 1280
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMPORT_BYTES = 500 * 1024 * 1024
START_TIME = time.monotonic()

db = SqliteConnection(os.environ.get("SQLITE_DB_PATH", SQLITE_DB_PATH))
_infer_semaphore = asyncio.Semaphore(1)

_admin_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


async def require_admin(token: str | None = Security(_admin_header)):
    """库管理接口鉴权：设置 ADMIN_TOKEN 后必须携带匹配的 X-Admin-Token 头。"""
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected:
        return
    if not token or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="需要管理员令牌（X-Admin-Token 请求头）")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading detection model...")
    from vector_operation import adaface, detector  # noqa: F401  (loads at import/lifespan)

    adaface.warmup()

    if not os.environ.get("ADMIN_TOKEN"):
        logger.warning("ADMIN_TOKEN not set: face library management API is UNPROTECTED")
    logger.info("Models ready")
    yield


app = FastAPI(title="FaceFind Face Recognition", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class Candidate(BaseModel):
    name: str
    similarity: float


class FaceHit(BaseModel):
    name: str
    similarity: float
    bbox: list[float]
    candidates: list[Candidate] = []


class CompareResult(BaseModel):
    similarity: float


class NameSuggestion(BaseModel):
    suggestions: list[str]


class AliasInput(BaseModel):
    alias: str


class MergeInput(BaseModel):
    source: str
    target: str


class PersonInfo(BaseModel):
    name: str
    count: int
    aliases: list[str] = []


class FaceList(BaseModel):
    items: list[PersonInfo]
    total: int
    page: int
    page_size: int


class ThumbInfo(BaseModel):
    id: int
    thumb: str | None


class Health(BaseModel):
    status: str
    uptime_seconds: float
    models: dict[str, bool]
    library: dict[str, int]


class HistoryItem(BaseModel):
    id: int
    ts: str
    name: str | None
    similarity: float
    thumb: str | None


class HistoryList(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int


class CrossPair(BaseModel):
    a: int
    a_name: str
    b: int
    b_name: str
    sim: float


class Outlier(BaseModel):
    id: int
    name: str
    sim_to_mean: float


class Issues(BaseModel):
    cross_name: list[CrossPair]
    outliers: list[Outlier]


async def _read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="图片过大")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解析图片")
    return img


def _downscale(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Limit long edge to MAX_IMAGE_EDGE; returns (image, scale_applied)."""
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= MAX_IMAGE_EDGE:
        return img, 1.0
    scale = MAX_IMAGE_EDGE / long_edge
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


@app.post("/api/recognize", response_model=list[FaceHit])
async def recognize(file: UploadFile = File(...)):
    img = await _read_image(file)
    scaled, scale = _downscale(img)
    async with _infer_semaphore:
        faces = get_result_from_array(scaled)
        if not faces:
            raise HTTPException(status_code=400, detail="未检测到人脸")
        pairs = get_face_data_from_faces(scaled, faces)
    hits = []
    for face, (vector, thumb) in zip(faces, pairs):
        top = db.find_top_similar(vector, k=RECOGNIZE_TOP_K)
        best_name, best_sim = top[0] if top else (None, -1.0)
        name = best_name if best_sim >= SIMILARITY_THRESHOLD else "Unknown"
        x1, y1, x2, y2 = (float(v) / scale for v in face.bbox)
        hits.append(
            FaceHit(
                name=name,
                similarity=round(float(best_sim), 4),
                bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                candidates=[
                    Candidate(name=n, similarity=round(float(s), 4)) for n, s in top
                ],
            )
        )
        db.log_recognition(
            None if name == "Unknown" else name, round(float(best_sim), 4), thumb
        )
        logger.info("Recognize %s with similarity %.4f", name, best_sim)
    return hits


@app.get("/api/health", response_model=Health)
async def health():
    """Liveness + model/library stats for monitoring. No auth required."""
    models = {
        "detector": len(getattr(detector, "models", {})) > 0,
        "adaface": adaface.is_loaded(),
    }
    return Health(
        status="ok" if all(models.values()) else "degraded",
        uptime_seconds=round(time.monotonic() - START_TIME, 1),
        models=models,
        library=db.stats(),
    )


@app.get("/api/faces", response_model=FaceList, dependencies=[Depends(require_admin)])
async def list_faces(q: str = "", page: int = 1, page_size: int = 50):
    return db.find_page(q=q, page=page, page_size=page_size)


@app.get("/api/faces/suggest", response_model=NameSuggestion)
async def suggest(q: str = ""):
    if not q:
        return NameSuggestion(suggestions=[])
    # 候选包含别名，命中别名时返回主名
    lookup: dict[str, str] = {}
    for p in db.find_all():
        lookup[p["name"]] = p["name"]
        for alias in p["aliases"]:
            lookup[alias] = p["name"]
    matches = get_close_matches(q, lookup.keys(), n=3, cutoff=0.2)
    suggestions = list(dict.fromkeys(lookup[m] for m in matches))
    return NameSuggestion(suggestions=suggestions)


@app.delete("/api/faces/{name}", dependencies=[Depends(require_admin)])
async def delete_face(name: str):
    removed = db.delete_one_by_name(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{name} 不存在")
    return {"deleted": name, "vectors": removed}


@app.delete(
    "/api/faces/{name}/vectors/{vector_id}",
    dependencies=[Depends(require_admin)],
)
async def delete_vector(name: str, vector_id: int):
    """Delete one registered vector (a bad sample) of a person."""
    if not db.delete_vector_by_id(name, vector_id):
        raise HTTPException(status_code=404, detail=f"向量 #{vector_id} 不存在")
    person = db.find_one_by_name(name)
    if person is None:
        return {"deleted": vector_id, "name": name, "remaining": 0, "person_gone": True}
    return {
        "deleted": vector_id,
        "name": name,
        "remaining": person["count"],
        "person_gone": False,
    }


@app.post(
    "/api/faces/{name}/aliases",
    response_model=PersonInfo,
    dependencies=[Depends(require_admin)],
)
async def add_alias(name: str, body: AliasInput):
    try:
        db.add_alias(body.alias, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PersonInfo(**db.find_one_by_name(name))


@app.delete(
    "/api/faces/{name}/aliases/{alias}", dependencies=[Depends(require_admin)]
)
async def remove_alias(name: str, alias: str):
    if not db.remove_alias(alias, name):
        raise HTTPException(status_code=404, detail=f"别名「{alias}」不存在")
    return {"removed": alias, "name": name}


@app.post(
    "/api/faces/merge", response_model=PersonInfo, dependencies=[Depends(require_admin)]
)
async def merge_faces(body: MergeInput):
    try:
        db.merge_person(body.source, body.target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PersonInfo(**db.find_one_by_name(body.target))


@app.get(
    "/api/faces/{name}/thumbs",
    response_model=list[ThumbInfo],
    dependencies=[Depends(require_admin)],
)
async def face_thumbs(name: str):
    """Per-vector WebP previews as data URIs (null = registered before thumbs)."""
    items = []
    for t in db.find_thumbs(name):
        uri = (
            "data:image/webp;base64," + base64.b64encode(t["thumb"]).decode("ascii")
            if t["thumb"]
            else None
        )
        items.append(ThumbInfo(id=t["id"], thumb=uri))
    return items


@app.get(
    "/api/faces/issues",
    response_model=Issues,
    dependencies=[Depends(require_admin)],
)
async def face_issues(
    merge_threshold: float = CLUSTER_MERGE_THRESHOLD,
    outlier_threshold: float = CLUSTER_OUTLIER_THRESHOLD,
):
    """Duplicate/outlier suggestions computed from the in-memory vector cache."""
    return db.find_issues(
        merge_threshold=merge_threshold, outlier_threshold=outlier_threshold
    )


@app.get("/api/history", response_model=HistoryList, dependencies=[Depends(require_admin)])
async def history(q: str = "", unknown: bool = False, page: int = 1, page_size: int = 50):
    data = db.find_history(q=q, unknown_only=unknown, page=page, page_size=page_size)
    for item in data["items"]:
        if item["thumb"]:
            item["thumb"] = (
                "data:image/webp;base64," + base64.b64encode(item["thumb"]).decode("ascii")
            )
    return data


@app.delete("/api/history", dependencies=[Depends(require_admin)])
async def clear_history():
    return {"deleted": db.delete_history()}


@app.delete("/api/history/{record_id}", dependencies=[Depends(require_admin)])
async def delete_history(record_id: int):
    if not db.delete_history(record_id):
        raise HTTPException(status_code=404, detail=f"记录 #{record_id} 不存在")
    return {"deleted": record_id}


@app.get(
    "/api/library/export", dependencies=[Depends(require_admin)]
)
async def export_library():
    """Download the whole library as a zip (manifest.json + face_vector.db)."""
    payload = db.snapshot_bytes()
    manifest = {
        "app": "facefind",
        "format": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **db.export_summary(),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("face_vector.db", payload)
    filename = f"face_library_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/library/import", dependencies=[Depends(require_admin)])
async def import_library(file: UploadFile = File(...)):
    """Replace the library with the zip produced by /api/library/export.

    recognize_log (recognition history) is not part of the library and is kept.
    """
    data = await file.read()
    if len(data) > MAX_IMPORT_BYTES:
        raise HTTPException(status_code=413, detail="导入文件过大")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="不是有效的 zip 文件")
    if "face_vector.db" not in zf.namelist():
        raise HTTPException(status_code=400, detail="压缩包中缺少 face_vector.db")

    fd, tmp = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        Path(tmp).write_bytes(zf.read("face_vector.db"))
        try:
            summary = db.import_database(tmp)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp).unlink(missing_ok=True)
    logger.info("Library imported from %s", file.filename)
    return summary


@app.post("/api/faces", response_model=PersonInfo)
async def upload_face(name: str, file: UploadFile = File(...)):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="名字不能为空")
    img = await _read_image(file)
    async with _infer_semaphore:
        faces = get_result_from_array(img)
        if faces is None or len(faces) == 0:
            raise HTTPException(status_code=400, detail="检测不到人脸")
        if len(faces) > 1:
            raise HTTPException(status_code=400, detail="检测到多个人脸，无法上传")
        if faces[0]["det_score"] < FACE_DETECT_THRESHOLD:
            raise HTTPException(status_code=400, detail="人脸置信度较低")
        vector, thumb = get_face_data_from_faces(img, faces)[0]
    db.update_one(FaceVectorModel(label=name, vector=vector, count=1, thumb=thumb))
    logger.info("Upload face %s", name)
    return PersonInfo(**db.find_one_by_name(db.resolve_name(name)))


@app.post("/api/compare", response_model=CompareResult)
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1, img2 = await _read_image(file1), await _read_image(file2)
    async with _infer_semaphore:
        vectors1 = get_vectors_from_faces(img1, get_result_from_array(img1) or [])
        if not vectors1:
            raise HTTPException(status_code=400, detail="图片1检测不到人脸")
        if len(vectors1) > 1:
            raise HTTPException(status_code=400, detail="图片1有多个人脸")
        vectors2 = get_vectors_from_faces(img2, get_result_from_array(img2) or [])
        if not vectors2:
            raise HTTPException(status_code=400, detail="图片2检测不到人脸")
        if len(vectors2) > 1:
            raise HTTPException(status_code=400, detail="图片2有多个人脸")
        similarity = compare_vector(vectors1[0], vectors2[0])
    return CompareResult(similarity=round(similarity, 4))


_web_dist = Path(__file__).parent / "web" / "dist"
if _web_dist.is_dir():
    app.mount("/", StaticFiles(directory=_web_dist, html=True), name="web")
