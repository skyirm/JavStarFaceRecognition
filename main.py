import numpy as np

from vector_operation import get_face_vector, compare_vector, get_face_label
from database_connect import DatabaseConnect, FaceVectorModel
from pathlib import Path
from database_connect import DatabaseConnect

db = DatabaseConnect()


p = Path.cwd() / "original_images"

face_models = []
for path in p.iterdir():
    label = path.name
    vector = np.zeros(512)
    count = 0
    for file in path.iterdir():
        face_vector = get_face_vector(file)
        if face_vector is None:
            continue
        count += 1
        vector += np.array(face_vector)
    mean_vector = vector / count
    face_models.append(FaceVectorModel(mean_vector, label, count))
    db.replace_one(FaceVectorModel(mean_vector.tolist(), label, count))
