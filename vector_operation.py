import insightface
import cv2
import numpy as np

from database_connect import FaceVectorModel

detector = insightface.app.FaceAnalysis(name="buffalo_l",providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=0)

def get_face_vector(path:str):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    faces = detector.get(img)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding

def compare_vector(vector1:list, vector2:list):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.dot(vector1, vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))


def get_face_label(faces:list[FaceVectorModel], vector:list) -> str:
    max_similarity = 0
    max_label = ""
    for face in faces:
        similarity = compare_vector(face.vector, vector)
        if similarity > max_similarity:
            max_similarity = similarity
            max_label = face.label
    return max_label

if __name__ == "__main__":
    img = cv2.imdecode(np.fromfile("./original_images/七嶋舞/wKGkQD_l_2.jpg", np.uint8), cv2.IMREAD_COLOR)
    result = detector.get(img)
    print(result)