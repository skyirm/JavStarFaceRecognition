import asyncio
import os
import secrets
from contextlib import asynccontextmanager
from difflib import get_close_matches
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    FACE_DETECT_THRESHOLD,
    RECOGNIZE_TOP_K,
    SIMILARITY_THRESHOLD,
    SQLITE_DB_PATH,
)
from logger import get_logger
from model import FaceVectorModel
from sqlite_connect import SqliteConnection
from vector_operation import (
    compare_vector,
    get_result_from_array,
    get_vectors_from_faces,
)

logger = get_logger(__name__)

MAX_IMAGE_EDGE = 1280
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

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


app = FastAPI(title="JavStar Face Recognition", lifespan=lifespan)
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
        vectors = get_vectors_from_faces(scaled, faces)
    hits = []
    for face, vector in zip(faces, vectors):
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
        logger.info("Recognize %s with similarity %.4f", name, best_sim)
    return hits


@app.get("/api/faces", response_model=list[PersonInfo], dependencies=[Depends(require_admin)])
async def list_faces():
    return [PersonInfo(**item) for item in db.find_all()]


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
        vector = get_vectors_from_faces(img, faces)[0]
    db.update_one(FaceVectorModel(label=name, vector=vector, count=1))
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
