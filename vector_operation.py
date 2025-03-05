import insightface
import cv2
import numpy as np
from insightface.app.common import Face

from mongo_connect import FaceVectorModel

detector = insightface.app.FaceAnalysis(
    name="buffalo_l", providers=["CUDAExecutionProvider","CPUExecutionProvider"]
)
detector.prepare(ctx_id=0, det_size=(640, 640))


def get_face_vector(path: str) -> list[float]|None:
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    faces = detector.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

def get_face_vector_from_array(img:list[list[list[int]]]) -> list[list[float]]|None:
    img = np.array(img)
    if img is None:
        return None
    faces = detector.get(img)
    return [face.normed_embedding.tolist() for face in faces]


def compare_vector(vector1: list, vector2: list) -> float:
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return 1 - np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )


def get_face_label(faces: list[FaceVectorModel], vector: list) -> tuple[str, float]:
    max_similarity = 0
    max_label = ""
    for face in faces:
        similarity = compare_vector(face.vector, vector)
        if similarity > max_similarity:
            max_similarity = similarity
            max_label = face.label
    return max_label, max_similarity

def get_result_from_array(img:list[list[list[int]]])->list[Face]|None:
    img = np.array(img)
    return detector.get(img)

# if __name__ == "__main__":
#     img = cv2.imdecode(np.fromfile("./original_images/七嶋舞/wKGkQD_l_2.jpg", np.uint8), cv2.IMREAD_COLOR)
#     result = detector.get(img)
#     print(result)
