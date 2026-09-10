import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

import adaface
from config import FACE_DETECT_THRESHOLD

_available = set(ort.get_available_providers())
_providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in _available]

detector = FaceAnalysis(name="antelopev2", allowed_modules=["detection"], providers=_providers)
detector.prepare(ctx_id=0, det_size=(640, 640))


def _face_vector(img: np.ndarray, face) -> np.ndarray:
    aligned = norm_crop(img, face.kps, image_size=112)
    return adaface.get(aligned)


def get_face_vector_from_file(path: str) -> np.ndarray | None:
    img = _imread_unicode(path)
    if img is None:
        return None
    faces = detector.get(img)
    if len(faces) != 1 or faces[0]["det_score"] < FACE_DETECT_THRESHOLD:
        return None
    return _face_vector(img, faces[0])


def get_face_vector_from_array(img) -> list[np.ndarray] | None:
    if img is None:
        return None
    img = np.ascontiguousarray(img, dtype=np.uint8)
    faces = detector.get(img)
    return [_face_vector(img, face) for face in faces]


def compare_vector(vector1, vector2) -> float:
    """Cosine similarity (higher = more similar). Vectors must be L2-normalized."""
    return float(np.dot(vector1, vector2))


def get_result_from_array(img):
    if img is None:
        return None
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return detector.get(img)


def _imread_unicode(path: str):
    """imdecode-based imread that supports non-ASCII paths on Windows."""
    data = np.fromfile(path, np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)
