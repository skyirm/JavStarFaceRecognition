import asyncio
import os
from contextlib import asynccontextmanager
from difflib import get_close_matches
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    FACE_DETECT_THRESHOLD,
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading detection model...")
    from vector_operation import detector  # noqa: F401  (SCRFD loads at import)

    logger.info("Models ready")
    yield


app = FastAPI(title="JavStar Face Recognition", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class FaceHit(BaseModel):
    name: str
    similarity: float
    bbox: list[float]


class CompareResult(BaseModel):
    similarity: float


class NameSuggestion(BaseModel):
    suggestions: list[str]


class PersonInfo(BaseModel):
    name: str
    count: int


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
        name, similarity = db.find_most_similar(vector)
        if similarity < SIMILARITY_THRESHOLD:
            name = "Unknown"
        x1, y1, x2, y2 = (float(v) / scale for v in face.bbox)
        hits.append(
            FaceHit(
                name=name,
                similarity=round(float(similarity), 4),
                bbox=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            )
        )
        logger.info("Recognize %s with similarity %.4f", name, similarity)
    return hits


@app.get("/api/faces", response_model=list[PersonInfo])
async def list_faces():
    return [PersonInfo(**item) for item in db.find_all()]


@app.get("/api/faces/suggest", response_model=NameSuggestion)
async def suggest(q: str = ""):
    if not q:
        return NameSuggestion(suggestions=[])
    matches = get_close_matches(q, db.find_all_name(), n=3, cutoff=0.2)
    return NameSuggestion(suggestions=matches)


@app.delete("/api/faces/{name}")
async def delete_face(name: str):
    removed = db.delete_one_by_name(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"{name} 不存在")
    return {"deleted": name, "vectors": removed}


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
    return PersonInfo(**db.find_one_by_name(name))


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
